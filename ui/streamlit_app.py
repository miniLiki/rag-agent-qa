import streamlit as st
import requests

st.set_page_config(page_title="RAG + Agent 文档问答系统", layout="wide")

st.title("RAG + Agent 文档问答系统")
st.write("上传 PDF 或 TXT 文档，完成文档解析、切分与检索。")

uploaded_file = st.file_uploader("上传文档", type=["pdf", "txt"])

if uploaded_file is not None:
    if st.button("开始解析"):
        with st.spinner("解析中，请稍等..."):
            files = {
                "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }
            response = requests.post("http://127.0.0.1:8000/upload", files=files, timeout=300)
            data = response.json()

        st.success("解析完成")
        st.write(f"文件名：{data['file_name']}")
        st.write(f"保存路径：{data['saved_path']}")
        st.write(f"文本长度：{data['text_length']}")

        st.subheader("文本预览")
        st.text_area("预览内容", data["preview"], height=220)

        st.subheader("Chunk 信息")
        st.write(f"Chunk 数量：{data['chunk_count']}")

        if data["chunk_preview"]:
            for i, chunk in enumerate(data["chunk_preview"], start=1):
                st.text_area(f"Chunk {i}", chunk, height=150)

st.divider()

st.subheader("文档检索测试")
query = st.text_input("输入你的问题")
top_k = st.slider("返回前几个相关片段", min_value=1, max_value=5, value=3)

if st.button("开始检索"):
    if not query.strip():
        st.warning("请输入问题")
    else:
        with st.spinner("检索中，请稍等..."):
            response = requests.post(
                "http://127.0.0.1:8000/search",
                json={"query": query, "top_k": top_k},
                timeout=300
            )
            data = response.json()

        st.success(data["message"])

        if data["results"]:
            for i, item in enumerate(data["results"], start=1):
                st.markdown(f"### 检索结果 {i}")
                st.write(f"相似度分数：{item['score']:.4f}")
                st.write(f"Chunk 索引：{item['index']}")
                st.text_area(f"Chunk 内容 {i}", item["chunk"], height=160)
        else:
            st.info("没有检索到结果，请先上传并解析文档。")

st.divider()

st.subheader("RAG 问答测试")
question = st.text_input("输入你的问答问题")
ask_top_k = st.slider("问答使用前几个证据片段", min_value=1, max_value=5, value=3, key="ask_top_k")

if st.button("开始问答"):
    if not question.strip():
        st.warning("请输入问题")
    else:
        with st.spinner("问答中，请稍等..."):
            response = requests.post(
                "http://127.0.0.1:8000/ask",
                json={"question": question, "top_k": ask_top_k},
                timeout=300
            )
            data = response.json()

        st.success(data["message"])

        st.subheader("改写后的检索问题")
        st.write(data["rewritten_question"])

        st.subheader("原始答案")
        st.write(data["answer"])

        st.subheader("审核后答案")
        st.write(data["reviewed_answer"])

        st.subheader("证据来源")
        if data["sources"]:
            for i, item in enumerate(data["sources"], start=1):
                st.markdown(f"### 证据 {i}")
                st.write(f"相似度分数：{item['score']:.4f}")
                st.write(f"Chunk 索引：{item['index']}")
                st.text_area(f"证据内容 {i}", item["chunk"], height=160, key=f"source_{i}")
        else:
            st.info("没有可展示的证据来源。")