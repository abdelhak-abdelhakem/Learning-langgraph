import glob
import os
from langchain_community.document_loaders import UnstructuredMarkdownLoader 
from langchain_openai import  OpenAIEmbeddings
from langchain_community.utils.math import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

directory_path = "notebooks/docs/"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

all_files = glob.glob(f"{directory_path}/job*.md", recursive=True)
all_files = [f for f in all_files if os.path.isfile(f)]

all_docs = []
texts = []
sources = []

for file_path in all_files:
        loader = UnstructuredMarkdownLoader(file_path)
        docs = loader.load()
        all_docs.extend(docs)

        for doc in docs:
            texts.append(doc.page_content)
            sources.append(file_path)


vectors = embeddings.embed_documents(texts)

similarity_matrix = cosine_similarity(vectors, vectors)

threshold = 0.95

duplicated_job = []

print(similarity_matrix)


for i in range(len(texts)):
    for j in range(i + 1, len(texts)):

        score = similarity_matrix[i][j]

        if score > threshold:
            duplicated_job.append((sources[i], sources[j], score))

print(duplicated_job)