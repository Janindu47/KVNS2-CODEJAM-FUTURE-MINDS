import os
import json
import csv
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class PatchedGeminiEmbeddings(GoogleGenerativeAIEmbeddings):
    def __init__(self, **kwargs):
        if "google_api_key" not in kwargs:
            load_dotenv()
            kwargs["google_api_key"] = os.getenv("GOOGLE_API_KEY")
        super().__init__(**kwargs)

PDF_PATH = "grade-11-history-text-book.pdf"
QUERIES_PATH = "queries.json"
CSV_OUTPUT = "submission.csv"
WORKFLOW_JSON = "workflow.json"

def load_textbook(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_info"] = f"Page {doc.metadata.get('page', 'N/A')}"
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    for doc in split_docs:
        doc.metadata["source_info"] = f"Page {doc.metadata.get('page', 'N/A')}"
    return split_docs

def build_vectorstore(docs):
    embeddings = PatchedGeminiEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(docs, embedding=embeddings)
    return vectorstore.as_retriever()

def create_rag_chain(retriever):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, max_tokens=512)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful history assistant. Use the following context to answer the question. "
         "Cite the page number and section if possible. Keep it short and accurate.\n\n{context}"),
        ("human", "{input}")
    ])
    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)

def process_queries(rag_chain, queries):
    filtered_queries = [q for q in queries if q.get("include", False)]
    results = []

    for i, query in enumerate(filtered_queries, start=1):
        qid = query.get("query_id", f"Q{i}")
        question = query.get("question", "")
        print(f"[{i}/{len(filtered_queries)}] 🧠 Processing: {question}")

        try:
            response = rag_chain.invoke({"input": question})
            answer = response["answer"]
            context_docs = response.get("context", [])

            raw_context = "\n\n".join(doc.page_content for doc in context_docs[:3])
            pages = list({doc.metadata.get("page", "N/A") for doc in context_docs})
            sections = list({doc.metadata.get("section", "N/A") for doc in context_docs if doc.metadata.get("section")})

            results.append({
                "ID": qid,
                "Context": raw_context,
                "Answer": answer,
                "Sections": "; ".join(sections) if sections else "N/A",
                "Pages": "; ".join(str(p) for p in pages),
                "Error": ""
            })
            print("✅ Success!")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            results.append({
                "ID": qid,
                "Context": "",
                "Answer": "",
                "Sections": "N/A",
                "Pages": "N/A",
                "Error": str(e)
            })

        time.sleep(4.5)

    return results

def write_csv(results, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Context", "Answer", "Sections", "Pages", "Error"])
        writer.writeheader()
        writer.writerows(results)

def export_workflow_json(path):
    workflow = {
        "agents": [
            {"name": "RAG Retriever", "task": "Fetch relevant chunks from the textbook"},
            {"name": "Gemini 1.5 Flash Generator", "task": "Generate answers using retrieved context"}
        ],
        "flow": [
            "User input -> Retriever -> Top context -> Gemini -> Final Answer (with reference)"
        ]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(workflow, f, indent=4)

if __name__ == "__main__":
    print("📖 Loading textbook...")
    documents = load_textbook(PDF_PATH)

    print("📚 Building vectorstore...")
    retriever = build_vectorstore(documents)

    print("🤖 Initializing chatbot...")
    rag_chain = create_rag_chain(retriever)

    print("📥 Loading questions...")
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print("💬 Generating answers for included questions...")
    results = process_queries(rag_chain, queries)

    print(f"📄 Saving results to {CSV_OUTPUT}")
    write_csv(results, CSV_OUTPUT)

    print(f"🧠 Exporting workflow to {WORKFLOW_JSON}")
    export_workflow_json(WORKFLOW_JSON)

    print("✅ Done! All selected questions processed.")
