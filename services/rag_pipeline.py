from services.llm import LLMClient


class RAGPipeline:
    def __init__(self):
        self.llm = LLMClient()

    def answer_question(self, question: str, retrieved_chunks: list[dict]) -> dict:
        if not retrieved_chunks:
            return {
                "answer": "未检索到相关证据，无法回答该问题。",
                "sources": [],
            }

        evidence_text = []
        for i, item in enumerate(retrieved_chunks, start=1):
            evidence_text.append(
                f"[证据{i} | idx={item['index']} | score={item['score']:.4f} | source={item['source']}]\n{item['chunk']}"
            )

        evidence_block = "\n\n".join(evidence_text)

        system_prompt = (
            "你是一个文档问答助手。"
            "请严格基于给定证据回答问题。"
            "如果证据不足，请明确说明“根据当前检索到的证据无法确定”，不要编造。"
        )

        user_prompt = f"""
请根据以下证据回答问题。

〖问题〗
{question}

〖证据〗
{evidence_block}

要求：
1. 只能基于证据回答
2. 不要使用证据之外的外部知识
3. 如果证据不足，明确说明证据不足
4. 回答尽量简洁清晰
"""

        answer = self.llm.chat(system_prompt=system_prompt, user_prompt=user_prompt)

        return {
            "answer": answer,
            "sources": retrieved_chunks,
        }