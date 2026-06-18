from dotenv import load_dotenv
import os
import glob
from pydantic import BaseModel , Field , field_validator
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredMarkdownLoader 
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from typing import TypedDict , List
from langgraph.graph import StateGraph , START , END

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0) 
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

directory_path = "notebooks/docs/cv-matcher-mini-project"
chroma_index_path = "notebooks/docs/cv-matcher-mini-project/chroma_db"
chroma_collection_name = "cv_matcher"
chunk_size = 300
chunk_overlap = 50
vector_k = 3
bm25_k = 3
retriever_weights = [0.5,0.5]

#Define the structured Output 

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


#RAG

def load_and_split_docs(directory_path: str, chunk_size: int, chunk_overlap: int):
    all_files = glob.glob(f"{directory_path}/**/*", recursive=True)
    all_files = [f for f in all_files if os.path.isfile(f)]

    all_chunks = []

    for file_path in all_files:
        loader = UnstructuredMarkdownLoader(file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)

    return all_chunks

def get_or_create_vector_store(doc_chunks) -> Chroma:
    vectorstore = Chroma(
        persist_directory=chroma_index_path,
        embedding_function=embeddings,
        collection_name=chroma_collection_name
    )

    if doc_chunks:
        vectorstore.add_documents(doc_chunks)

    return vectorstore


#graph
    
    
class CVMatcherState (TypedDict):
    question : str
    context :  List
    answer : JobMatch

chunks = load_and_split_docs(directory_path, chunk_size, chunk_overlap)
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


def retrieve_profile_node(state : CVMatcherState) -> CVMatcherState:
    """"""
    question = state["question"]
    docs = ensemble_retriever.invoke(question)
    state["context"] = docs
    return state



def score_job_node(state : CVMatcherState)-> CVMatcherState:
    """"""
    question = state["question"]
    context = state["context"]

    template = """You are a technical recruiter. Evaluate this candidate profile against the job description and return a structured assessment.
    
    Candidate Profile and Project Readme:
    {context}
    
    Question: {question}
    """
    prompt = PromptTemplate.from_template(template)


    formatted_context = "\n\n".join(doc.page_content for doc in context)
    rag_chain = prompt| structured_llm
    state["answer"]= rag_chain.invoke({"context": formatted_context, "question": question})
    return state

graph  = StateGraph(CVMatcherState)
graph.add_node("retrieve_profile_node",retrieve_profile_node)
graph.add_node("score_job_node",score_job_node)

graph.add_edge(START,"retrieve_profile_node")
graph.add_edge("retrieve_profile_node","score_job_node")
graph.add_edge("score_job_node",END)

app = graph.compile()



job_postings = []
for i in range(1, 4):
    file_name = f"notebooks/docs/job{i}.md"
    
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            content = file.read()
            
            
            job_postings.append(content)
            print(f"Successfully added {file_name} to job_postings.")
            
    except FileNotFoundError:
        print(f"Warning: {file_name} was not found.")
        job_postings.append(None)

for i, job in enumerate(job_postings, 1):
    final_state = app.invoke({"question": job})
    print(final_state["answer"])