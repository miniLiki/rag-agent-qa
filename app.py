from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from services.parser import parse_document
from services.chunker import chunk_text
from services.vector_store import SimpleVectorStore
from services.rag_pipeline import RAGPipeline
from utils.file_utils import save_uploaded_file
from agents.rewrite_query_agent import RewriteQueryAgent
from agents.review_answer_agent import ReviewAnswerAgent

app = FastAPI(title="RAG + Agent QA System")

vector_store = SimpleVectorStore()
rag_pipeline = RAGPipeline()
current_chunks = []
rewrite_query_agent = RewriteQueryAgent()
review_answer_agent = ReviewAnswerAgent()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


class AskRequest(BaseModel):
    question: str
    top_k: int = 3


@app.get("/")
def health():
    return {"status": "ok", "message": "项目2后端启动成功"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global current_chunks

    file_bytes = await file.read()
    saved_path = save_uploaded_file(file.filename, file_bytes)

    text = parse_document(saved_path)
    chunks = chunk_text(text, chunk_size=500, overlap=100)

    current_chunks = chunks
    vector_store.build_index(chunks)

    return {
        "message": "文件上传解析成功",
        "file_name": file.filename,
        "saved_path": saved_path,
        "text_length": len(text),
        "chunk_count": len(chunks),
        "preview": text[:500],
        "chunk_preview": chunks[:3]
    }


@app.post("/search")
def search_chunks(req: SearchRequest):
    if not current_chunks:
        return {
            "message": "请先上传并解析文档",
            "results": []
        }

    results = vector_store.search(req.query, top_k=req.top_k)

    return {
        "message": "检索成功",
        "query": req.query,
        "top_k": req.top_k,
        "results": results
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
            "sources": []
        }

    rewritten_question = rewrite_query_agent.run(req.question)
    retrieved = vector_store.search(rewritten_question, top_k=req.top_k)
    result = rag_pipeline.answer_question(rewritten_question, retrieved)
    reviewed_answer = review_answer_agent.run(
        question=rewritten_question,
        answer=result["answer"],
        sources=result["sources"]
    )

    return {
        "message": "问答成功",
        "question": req.question,
        "rewritten_question": rewritten_question,
        "top_k": req.top_k,
        "answer": result["answer"],
        "reviewed_answer": reviewed_answer,
        "sources": result["sources"]
    }