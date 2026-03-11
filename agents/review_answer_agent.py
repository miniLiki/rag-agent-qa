from agents.base import BaseAgent


class ReviewAnswerAgent(BaseAgent):
    def run(self, question: str, answer: str, sources: list[dict]) -> str:
        evidence_text = []
        for i, item in enumerate(sources, start=1):
            evidence_text.append(f"[证据{i}]\n{item['chunk']}")

        evidence_block = "\n\n".join(evidence_text) if evidence_text else "无可用证据"

        system_prompt = (
            "你是一个答案审核助手。"
            "请判断回答是否严格基于证据。"
            "如果回答超出了证据范围，请改写成更保守、更忠于证据的版本。"
        )

        user_prompt = f"""
请审核下面这个回答。

要求：
1. 判断回答是否严格基于证据
2. 如果回答与证据一致，可直接保留原回答
3. 如果回答超出证据，请改写成更保守、更准确的版本
4. 只输出最终审核后的回答，不要解释

【问题】
{question}

【原回答】
{answer}

【证据】
{evidence_block}
"""
        return self.run_llm(system_prompt=system_prompt, user_prompt=user_prompt).strip()