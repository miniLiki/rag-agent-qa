from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from services.parser import parse_document
from services.chunker import chunk_text
from services.hybrid_retriever import HybridRetriever
from services.rag_pipeline import RAGPipeline
from utils.file_utils import save_uploaded_file
from agents.rewrite_query_agent import RewriteQueryAgent
from agents.review_answer_agent import ReviewAnswerAgent

app = FastAPI(title="RAG + Agent QA System")

retriever = None
rag_pipeline = None
current_chunks = []

rewrite_query_agent = RewriteQueryAgent()
review_answer_agent = ReviewAnswerAgent()


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


class AskRequest(BaseModel):
    question: str
    top_k: int = 3

def get_retriever():
    global retriever
    if retriever is None:
        print("[APP] importing HybridRetriever...")
        from services.hybrid_retriever import HybridRetriever
        print("[APP] HybridRetriever imported")

        print("[APP] creating HybridRetriever...")
        retriever = HybridRetriever()
        print("[APP] HybridRetriever created")

    return retriever


def get_rag_pipeline():
    global rag_pipeline
    if rag_pipeline is None:
        from services.rag_pipeline import RAGPipeline
        rag_pipeline = RAGPipeline()
    return rag_pipeline

@app.get("/")
def health():
    return {"status": "ok", "message": "项目后端启动成功"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global current_chunks

    print("[UPLOAD] start")
    file_bytes = await file.read()
    print(f"[UPLOAD] read file bytes: {len(file_bytes)}")

    saved_path = save_uploaded_file(file.filename, file_bytes)
    print(f"[UPLOAD] saved_path: {saved_path}")

    text = parse_document(saved_path)
    print(f"[UPLOAD] parsed text length: {len(text)}")

    chunks = chunk_text(text, chunk_size=500, overlap=100)
    print(f"[UPLOAD] chunk count: {len(chunks)}")

    current_chunks = chunks

    print("[UPLOAD] rebuilding retriever...")
    get_retriever().rebuild(chunks)
    print("[UPLOAD] rebuild done")

    return {
        "message": "文件上传解析成功",
        "file_name": file.filename,
        "saved_path": saved_path,
        "text_length": len(text),
        "chunk_count": len(chunks),
        "preview": text[:500],
        "chunk_preview": chunks[:3],
    }


@app.post("/search")
def search_chunks(req: SearchRequest):
    if not current_chunks:
        return {"message": "请先上传并解析文档", "results": []}

    results = get_retriever().retrieve(
        req.query,
        recall_k=max(8, req.top_k * 3),
        final_k=req.top_k,
    )

    return {
        "message": "检索成功",
        "query": req.query,
        "top_k": req.top_k,
        "results": [item.__dict__ for item in results],
    }


@app.post("/ask")
def ask_question(req: AskRequest):
    if not current_chunks:
        return {
            "message": "请先上传并解析文档",
            "question": req.question,
            "rewritten_question": "",
            "answer": "",
            "reviewed_answer": "",
            "sources": [],
        }

    rewritten_question = rewrite_query_agent.run(req.question)

    retrieved = get_retriever().retrieve(
        query=rewritten_question,
        recall_k=max(8, req.top_k * 3),
        final_k=req.top_k,
    )

    retrieved_dicts = [item.__dict__ for item in retrieved]

    result = get_rag_pipeline().answer_question(rewritten_question, retrieved_dicts)

    reviewed_answer = review_answer_agent.run(
        question=rewritten_question,
        answer=result["answer"],
        sources=result["sources"],
    )

    return {
        "message": "问答成功",
        "question": req.question,
        "rewritten_question": rewritten_question,
        "top_k": req.top_k,
        "answer": result["answer"],
        "reviewed_answer": reviewed_answer,
        "sources": result["sources"],
    }

