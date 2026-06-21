import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

chroma_index_path = "notebooks/docs/scraper_to_Chroma-mini-project/chroma_db"
chroma_collection_name = "inria_jobs"
chunk_size = 1000
chunk_overlap = 150
similarity_threshold = 0.95


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


def scrape_inria_jobs():
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
        return []

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

    return job_listings


def jobs_to_documents(job_listings: list) -> list[Document]:
    """Converts raw scraped job dicts into LangChain Document objects."""
    documents = []
    for item in job_listings:
        combined_text = f"{item['title']}\n\n{item['description']}"
        doc = Document(
            page_content=combined_text,
            metadata={"url": item["url"], "job_id": item["id"]}
        )
        documents.append(doc)
    return documents




def filter_new_jobs(vector_db, documents):
    """
    For each scraped job, checks if a near-duplicate already exists in Chroma.
    Returns only the genuinely new jobs.
    """
    existing = vector_db.get()  
    existing_ids = {meta.get("job_id") for meta in existing["metadatas"]}

    new_jobs = [doc for doc in documents if doc.metadata["job_id"] not in existing_ids]
    print(f"\n{len(new_jobs)} new jobs found, {len(documents) - len(new_jobs)} duplicates skipped (by ID).")
    return new_jobs

def store_documents(vector_db: Chroma, documents: list[Document], chunk_size: int, chunk_overlap: int):
    """Chunks and stores documents into the Chroma collection."""
    if not documents:
        print("Nothing new to store.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    vector_db.add_documents(chunks)
    print(f"Stored {len(chunks)} chunks from {len(documents)} new jobs.")


if __name__ == "__main__":
    # 1. Scrape
    job_listings = scrape_inria_jobs()
    if not job_listings:
        print("No jobs scraped, exiting.")
        exit()

    documents = jobs_to_documents(job_listings)

    # 2. Load or create the persistent Chroma collection
    vector_db = Chroma(
        persist_directory=chroma_index_path,
        embedding_function=embeddings,
        collection_name=chroma_collection_name
    )

    # 3. Dedup against what's already stored
    new_jobs = filter_new_jobs(vector_db, documents)

    # 4. Store only the genuinely new jobs
    store_documents(vector_db, new_jobs, chunk_size, chunk_overlap)