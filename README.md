# RAG + Agent 文档问答系统

一个基于 RAG 和 Agent 的文档问答原型系统，支持文档上传、文本解析、chunk 切分、向量检索、基于证据回答，以及问题改写与答案审核。

## 功能模块

- 文档上传：支持 TXT / PDF 文件上传
- 文本解析：提取文档文本内容
- 文本切分：将长文本切分为多个 chunks
- 向量检索：基于 FAISS 返回最相关的 top-k 片段
- RAG 问答：基于检索到的证据生成回答
- RewriteQueryAgent：将用户问题改写为更适合检索的形式
- ReviewAnswerAgent：对生成答案进行审核，减少超出证据范围的回答

## 技术栈

- Python
- FastAPI
- Streamlit
- PyMuPDF
- FAISS
- Requests
- LLM API
- Prompt Engineering

## 项目结构

```text
rag-agent-qa/
├─ app.py
├─ requirements.txt
├─ .env.example
├─ .gitignore
├─ README.md
├─ agents/
│  ├─ base.py
│  ├─ rewrite_query_agent.py
│  └─ review_answer_agent.py
├─ services/
│  ├─ llm.py
│  ├─ parser.py
│  ├─ chunker.py
│  ├─ embedder.py
│  ├─ vector_store.py
│  └─ rag_pipeline.py
├─ ui/
│  └─ streamlit_app.py
├─ utils/
│  └─ file_utils.py
└─ data/
   └─ uploads/
```

## 系统流程

1. 上传文档
2. 解析文本
3. 切分为多个 chunks
4. 建立向量索引
5. 用户提问
6. RewriteQueryAgent 改写问题
7. 检索 top-k 证据片段
8. 基于证据生成回答
9. ReviewAnswerAgent 审核答案
10. 返回最终答案与证据来源

## 运行方式

### 1. 创建虚拟环境并安装依赖

```bash
uv venv --python /usr/bin/python3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，并填入你自己的模型接口配置。

### 3. 启动后端

```bash
uvicorn app:app --reload
```

### 4. 启动前端

```bash
streamlit run ui/streamlit_app.py
```

## 当前说明

当前版本为本地原型，重点在于 RAG 主流程、Agent 增强流程、前后端联调与可解释性展示。  
后续可继续扩展：

- 真实 embedding 模型替换
- 多文档支持
- PDF 复杂版式解析
- 持久化索引
- 历史记录与问答缓存