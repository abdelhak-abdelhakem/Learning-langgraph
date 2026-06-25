from typing import TypedDict, List
from langgraph.graph import StateGraph , START , END

from dotenv import load_dotenv
from pydantic import BaseModel , Field , field_validator
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredMarkdownLoader 
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from telegram_job_digest_format import send_message

load_dotenv()
llm = ChatOpenAI(model="gpt-5-nano", temperature=0) 
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

chroma_index_path = "notebooks/docs/jobradar_mini_pipeline/chroma_db"
chroma_collection_name = "jobradar_inria_jobs"
chunk_size = 300
chunk_overlap = 50
profile_path = "notebooks/docs/my_profile.md"
vector_k = 3
bm25_k = 3
retriever_weights = [0.5,0.5]

loader = UnstructuredMarkdownLoader(profile_path)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
chunks = splitter.split_documents(docs)
def get_or_create_vector_store(doc_chunks) -> Chroma:
    vectorstore = Chroma(
        persist_directory=chroma_index_path,
        embedding_function=embeddings,
        collection_name=chroma_collection_name
    )

    if doc_chunks:
        vectorstore.add_documents(doc_chunks)

    return vectorstore


vectorstore = get_or_create_vector_store(chunks)

chroma_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": vector_k}
)

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = bm25_k

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever],
    weights=retriever_weights
)





class JobMatch (BaseModel):
    match_score: int = Field(...,
                             description="the macth score must be between 0 to 100")
    rationale: str 
    matched_skills: list[str] 
    missing_skills: list[str] 

    @field_validator("match_score")
    @classmethod
    def score_must_be_valid(cls, score:int)-> int:
        if not 0 <= score <= 100:
            raise ValueError(f"match_score must be between 0 and 100, got {score}")
        return score
    
    @field_validator("matched_skills", "missing_skills")
    @classmethod
    def no_empty_strings(cls, v):
        cleaned = [s for s in v if s.strip() != ""]
        return cleaned


structured_llm = llm.with_structured_output(JobMatch)


class JobRadarState(TypedDict):
    job_listings: list      # written by scrape_node
    new_jobs: list          # written by dedup_node
    scored_jobs: list       # written by retrieve_profile_node + score_job_node
    cover_letters: dict     # written by draft_letter_node
    digest_sent: bool       # written by notify_telegram_node

def scrape_job_details(url, headers):
    """Fetches the individual job page and extracts the full description."""
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        main_content = soup.find('div', class_='contenu-offre') or soup.find('main')
        if not main_content:
            main_content = soup.body

        for element in main_content(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        full_text = main_content.get_text(separator='\n\n', strip=True)
        return full_text

    except requests.RequestException as e:
        print(f"  [!] Error fetching details for {url}: {e}")
        return "Error fetching description."


def scrape_node(state:JobRadarState)->JobRadarState:
    """scrape jobs"""
    base_url = "https://jobs.inria.fr"
    search_url = f"{base_url}/public/classic/en/offres"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    print(f"Fetching job listings from: {search_url}...\n")
    try:
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching the main page: {e}")
        state["job_listings"] = []
        return state

    soup = BeautifulSoup(response.text, 'html.parser')
    job_listings = []

    # 1. Gather all unique job URLs and map them to their titles
    job_dict = {}
    for link in soup.find_all('a', href=True):
        href = link['href']
        if "/public/classic/en/offres/" in href and any(char.isdigit() for char in href):
            job_url = urljoin(base_url, href)
            job_title = link.get_text(strip=True)
            if job_url not in job_dict:
                job_dict[job_url] = job_title

    print(f"Found {len(job_dict)} unique jobs. Starting deep scrape...\n")

    # 2. Scrape full details for each job
    for idx, (job_url, job_title) in enumerate(job_dict.items(), 1):
        print(f"[{idx}/{len(job_dict)}] Scraping: {job_title}")
        job_id = job_url.split('/')[-1]
        full_description = scrape_job_details(job_url, headers)

        job_listings.append({
            'id': job_id,
            'title': job_title,
            'url': job_url,
            'description': full_description
        })

        time.sleep(0.5)  # be polite to the server
    state["job_listings"] = job_listings
    return state

def dedup_node(state:JobRadarState)->JobRadarState:
    """
    For each scraped job, checks if a near-duplicate already exists in Chroma.
    Returns only the genuinely new jobs.
    """
    job_listings = state["job_listings"]
    documents = []
    for item in job_listings:
        combined_text = f"{item['title']}\n\n{item['description']}"
        doc = Document(
            page_content=combined_text,
            metadata={"url": item["url"], "job_id": item["id"]}
        )
        documents.append(doc)
    vector_db = Chroma(
        persist_directory="notebooks/docs/scraper_to_Chroma-mini-project/chroma_db",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name="inria_jobs"
    )
    existing = vector_db.get()  
    existing_ids = {meta.get("job_id") for meta in existing["metadatas"]}

    new_jobs = [doc for doc in documents if doc.metadata["job_id"] not in existing_ids]
    print(f"\n{len(new_jobs)} new jobs found, {len(documents) - len(new_jobs)} duplicates skipped (by ID).")
    state["new_jobs"] = new_jobs
    return state

def retrieve_profile_node(state : JobRadarState) -> JobRadarState:
    """"""
    scored_jobs = []
    list_of_new_jobs = state["new_jobs"]
    for new_job in list_of_new_jobs:
        chunks = ensemble_retriever.invoke(new_job.page_content[:500])
        scored_jobs.append({"doc":new_job,# original job document
                            "chunks":chunks # retrieved profile chunks
                            })
    state["scored_jobs"] = scored_jobs
    return state



def score_job_node(state : JobRadarState)-> JobRadarState:
    """"""
    updated_scored_jobs = []
    for item in state["scored_jobs"]:
        # item has doc + chunks already
        # invoke structured_llm, write job_match back into item

        context =  item["chunks"]
        question = item["doc"]
        template = """You are a technical recruiter. Evaluate this candidate profile against the job description and return a structured assessment.
    
        Candidate Profile and Project Readme:
        {context}
    
        Question: {question}
        """
        prompt = PromptTemplate.from_template(template)


        formatted_context = "\n\n".join(doc.page_content for doc in context)
        rag_chain = prompt| structured_llm
        answer = rag_chain.invoke({"context": formatted_context, "question": question.page_content})
        updated_scored_jobs.append({"doc":question,# original job document
                            "chunks":context, # retrieved profile chunks
                            "job_match":answer #structured_llm answer
                            })
    state["scored_jobs"] = updated_scored_jobs
    return state


def draft_letter_node(state : JobRadarState)-> JobRadarState:
    """"""
    cover_letters = {}
    for item in state["scored_jobs"]:
        context =  item["chunks"]
        question = item["doc"]
        job_id = question.metadata["job_id"]
        if item["job_match"].match_score < 70 : 
            cover_letters[job_id] = {}
        else : 
            template = """You are an expert Career Coach, HR Specialist, and Professional Cover Letter Writer.

            Your task is to generate a highly personalized and professional motivation letter (cover letter) based on:

            1. The Job Description provided by the user.
            2. The Candidate Profile retrieved from the vector database (CV, skills, education, projects, experience, certifications, languages, achievements).

            ## Objective

            Create a motivation letter that maximizes the alignment between the candidate's profile and the job requirements.

            The letter must:

            * Highlight the most relevant skills and experiences.
            * Demonstrate understanding of the company's needs.
            * Explain why the candidate is a strong fit.
            * Remain truthful and only use information available in the candidate profile.
            * Be professional, persuasive, and human-like.
            * Avoid generic statements whenever possible.

            ## Writing Guidelines

            1. Analyze the job description carefully:

            * Required skills
            * Preferred skills
            * Responsibilities
            * Technologies
            * Industry/domain
            * Education requirements

            2. Analyze the candidate profile:

            * Technical skills
            * Projects
            * Professional experience
            * Academic background
            * Certifications
            * Languages
            * Achievements

            3. Identify the strongest matches between the candidate and the position.

            4. Generate a motivation letter with the following structure:

            ### Introduction

            * Mention the position.
            * Express enthusiasm for the opportunity.

            ### Body Paragraph 1

            * Present the candidate's educational background and relevant expertise.

            ### Body Paragraph 2

            * Highlight the most relevant skills, projects, and experiences that match the job description.

            ### Body Paragraph 3

            * Explain why the candidate is interested in the company or role.
            * Show how the candidate can contribute.

            ### Conclusion

            * Reaffirm motivation.
            * Thank the recruiter.
            * Express availability for an interview.

            ## Constraints

            * Do NOT invent experiences, degrees, projects, companies, certifications, or skills.
            * Do NOT mention information that does not exist in the retrieved profile.
            * If some job requirements are missing from the profile, focus on transferable skills and learning ability.
            * Keep a professional and confident tone.
            * Avoid repeating the same information.
            * Use clear and concise language.
            * Generate between 300 and 500 words unless the user requests otherwise.

            ## Output Format

            Return only the final motivation letter in well-formatted text.

            ### Input

            Job Description:
            {question}

            ### Candidate Profile:

            {context}

            Generate the motivation letter.

                """
            prompt = PromptTemplate.from_template(template)

            formatted_context = "\n\n".join(doc.page_content for doc in context)
            rag_chain = prompt | llm | StrOutputParser()
            cover_letter = rag_chain.invoke({"context": formatted_context, "question": question.page_content})
            cover_letters[job_id] = cover_letter
    state["cover_letters"]= cover_letters
    return state


def route_by_score(state: JobRadarState) -> str:
    """The conditional edge after score_job_node: if match_score >= 70, route to draft_letter_node; otherwise skip straight to notify_telegram_node with just the score (no letter for weak matches)"""
    top_score = max(item["job_match"].match_score for item in state["scored_jobs"])
    if top_score >= 70:
        return "draft_letter_node"
    return "notify_telegram_node"

def notify_telegram_node(state:JobRadarState)->JobRadarState:
    """send notification in telegram"""
    for item in state["scored_jobs"]:
        job_id = item["doc"].metadata["job_id"]
        score = item["job_match"].match_score
        title = item["doc"].metadata.get("title", "Unknown")
        url = item["doc"].metadata["url"]
        cover_letter = state["cover_letters"].get(job_id, "") 
        if len(cover_letter) > 10:#check if the cover letter exist 
            send_message(f"""*🎯 {title}*

*Score:* {score}/100
*cover letter: * {cover_letter}  
[View Job]({url})""")
        else : 
            send_message(f"""*🎯 {title}*

*Score:* {score}/100

[View Job]({url})""")
    state["digest_sent"] = True
    return state


graph = StateGraph(JobRadarState)

graph.add_node("scrape_node",scrape_node)
graph.add_node("dedup_node",dedup_node)
graph.add_node("retrieve_profile_node",retrieve_profile_node)
graph.add_node("score_job_node",score_job_node)
graph.add_node("draft_letter_node",draft_letter_node)
graph.add_node("notify_telegram_node",notify_telegram_node)


graph.add_edge(START,"scrape_node")
graph.add_edge("scrape_node","dedup_node")
graph.add_edge("dedup_node","retrieve_profile_node")
graph.add_edge("retrieve_profile_node","score_job_node")
graph.add_conditional_edges(
    "score_job_node",
    route_by_score,
    {
        "draft_letter_node":"draft_letter_node",
        "notify_telegram_node":"notify_telegram_node"

    }
)

"""The conditional edge after score_job_node: if match_score >= 70, route to draft_letter_node; otherwise skip straight to notify_telegram_node with just the score (no letter for weak matches)"""


graph.add_edge("draft_letter_node", "notify_telegram_node")  
graph.add_edge("notify_telegram_node",END)

if __name__ == "__main__":
    app = graph.compile()
    app.invoke({
        "job_listings": [],
        "new_jobs": [],
        "scored_jobs": [],
        "cover_letters": {},
        "digest_sent": False
    })