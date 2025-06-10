# ecommbot/.env  
# ✅ (Keep this in your root project folder, DO NOT upload to GitHub)
# Example:
# HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
# ASTRA_DB_API_ENDPOINT=https://your-db-id-us-east1.apps.astra.datastax.com
# ASTRA_DB_APPLICATION_TOKEN=your_astra_token_here
# ASTRA_DB_KEYSPACE=your_keyspace_name


# ecommbot/ingest.py ✅
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from ecommbot.data_converter import dataconveter

load_dotenv()

ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def ingestdata(status):
    vstore = AstraDBVectorStore(
        embedding=embedding,
        collection_name="chatbotecomm",
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
        namespace=ASTRA_DB_KEYSPACE,
    )

    if status is None:
        docs = dataconveter()
        inserted_ids = vstore.add_documents(docs)
        return vstore, inserted_ids
    else:
        return vstore

if __name__ == '__main__':
    vstore_result = ingestdata(None)

    if isinstance(vstore_result, tuple):
        vstore, inserted_ids = vstore_result
        print(f"\nInserted {len(inserted_ids)} documents.")
    else:
        vstore = vstore_result

    results = vstore.similarity_search("can you tell me the low budget sound basshead.")
    for res in results:
        print(f"* {res.page_content} [{res.metadata}]")


# ecommbot/data_converter.py ✅
import json
from langchain_core.documents import Document

def dataconveter():
    try:
        with open("data/raw_reviews.json", "r", encoding="utf-8") as file:
            data = json.load(file)

        documents = []
        for item in data:
            metadata = {"product": item.get("product", "N/A")}
            doc = Document(page_content=item.get("review", ""), metadata=metadata)
            documents.append(doc)

        return documents
    except Exception as e:
        print("❌ Error in dataconveter:", e)
        return []


# ecommbot/retrieval_generation.py ✅
import os
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub
from ecommbot.ingest import ingestdata

load_dotenv()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    huggingfacehub_api_token=token,
    model_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 512
    }
)

print("\U0001F916 Running query...")
vstore = ingestdata("use_existing")
query = "Suggest me the best mobile under 15000"
docs = vstore.similarity_search(query, k=3)

for doc in docs:
    print("\n📄 Matched Review:", doc.page_content)

response = llm.invoke(query)
print("\n🤖 Bot Response:", response)
