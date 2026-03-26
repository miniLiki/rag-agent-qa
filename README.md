# RAG + Agent 问答系统

一个基于 RAG（检索增强生成）的轻量级问答系统，使用 FastAPI 和简化的 Agent 流水线构建。

该项目正在从基于 FAISS 的教学原型，升级为更接近生产级的检索架构，包含混合检索、重排以及 Milvus 集成。

---

## 当前状态

当前仓库处于 **升级 / 重构阶段**。

已完成：

- 将检索层重构为模块化组件
- 用真实的 `sentence-transformers` 向量接口替换原有的示例 embedding 逻辑
- 添加 BM25 稀疏检索支持
- 添加混合检索（Hybrid Retrieval）框架（含结果融合）
- 添加 Cross-Encoder 重排接口
- 重构 FastAPI 初始化流程，使用懒加载避免启动阻塞
- 已验证后端健康检查与上传流程可正常运行

当前阻塞问题：

- Milvus 在本地环境中尚未完全运行
- 问题已定位到稠密检索路径中的 `MilvusClient(...)` 初始化

当前阶段总结：

系统结构与检索层重构已基本完成，但 Milvus 稠密检索路径仍需调试。

---

## 项目目标

构建一个模块化 RAG 问答系统，包含：

- 查询改写（Query Rewriting）
- 稠密检索（Dense Retrieval）
- 稀疏检索（Sparse Retrieval）
- 混合融合（Hybrid Fusion）
- 重排（Reranking）
- 基于证据的答案生成（Evidence-grounded Answer）
- 答案审查 / 评估（Answer Review / Critique）

---

## 系统架构

目标流水线：

```text
User Question
→ Rewrite Query Agent
→ Hybrid Retrieval
   → Dense Retrieval (Milvus)
   → Sparse Retrieval (BM25)
→ Fusion (RRF)
→ Reranker
→ RAG Answer Generation
→ Review Answer Agent
→ Final Response
```

当前稳定路径：

```
User Question
→ Rewrite Query Agent
→ Retriever 初始化
→ 上传 / 解析 / 切分
→ 检索流水线重构中
```

---

## 项目结构

```
rag-agent-qa/
├── app.py
├── requirements.txt
├── services/
│   ├── embedder.py
│   ├── bm25_store.py
│   ├── milvus_store.py
│   ├── hybrid_retriever.py
│   ├── reranker.py
│   ├── retrieval_types.py
│   ├── rag_pipeline.py
│   └── ...
├── agents/
│   ├── rewrite_query_agent.py
│   ├── review_answer_agent.py
│   └── ...
└── data/
```

---

## 关键重构

### 1. Embedding 升级

原始原型使用的是用于演示的简单 embedding 方法。

现已替换为基于 `sentence-transformers` 的真实向量接口。

---

### 2. 检索层模块化

原先单路径的 FAISS 检索逻辑被替换为模块化组件：

- `MilvusVectorStore`
- `BM25Store`
- `HybridRetriever`
- `CrossEncoderReranker`

优势：

- 更易扩展
- 更易调试

---

### 3. FastAPI 懒加载初始化

原始实现中，模型与检索器在导入时初始化，可能导致启动阻塞。

现已改为按需加载（lazy loading），仅在需要时初始化组件。

---

## 已实现组件

### 查询改写 Agent

在检索前对用户问题进行改写。

---

### 稀疏检索

基于 BM25 实现。

---

### 稠密检索

基于 Milvus，目前正在集成中。

---

### 混合检索

已实现基于 RRF 的混合检索框架。

---

### 重排（Reranking）

已添加 Cross-Encoder 重排接口。

---

### 答案审查

系统中包含用于答案校验的 Review Agent。

---

## API 接口

### 健康检查

```bash
GET /
```

预期返回：

```json
{"status":"ok","message":"项目后端启动成功"}
```

---

### 上传文档

```bash
POST /upload
```

---

### 搜索

```bash
POST /search
```

---

### 提问

```bash
POST /ask
```

---

## 本地运行

安装依赖：

```bash
pip install -r requirements.txt
```

启动后端：

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

测试健康检查：

```bash
curl --noproxy '*' http://127.0.0.1:8000/
```

---

## 已知问题

- Milvus 在本地环境初始化存在问题
- 稠密检索路径尚未完全验证
- 重排模块已搭建，但端到端验证依赖检索稳定性
- 当前分支属于重构 / 升级过程，不是最终生产版本

---

## 路线图

下一步计划：

1. 解决 Milvus 客户端初始化问题
2. 完整验证稠密检索
3. 打通混合检索 + 重排的端到端流程
4. 增加检索评估指标
5. 优化文档持久化与元数据支持
6. 增加更完善的前端 / Demo 流程

---

## 总结

该项目最初是一个基于 FAISS 的最小化 RAG 教学原型。

当前正在升级为更接近生产环境的检索系统，具备：

- 真实向量表示（embedding）
- 稀疏 + 稠密检索
- 混合融合
- 重排机制
- 模块化检索架构

目前后端结构与重构基础已完成，主要剩余工作是稳定 Milvus 稠密检索集成。
