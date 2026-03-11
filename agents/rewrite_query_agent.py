from agents.base import BaseAgent


class RewriteQueryAgent(BaseAgent):
    def run(self, question: str) -> str:
        system_prompt = (
            "你是一个文档检索问题改写助手。"
            "请将用户问题改写成更适合文档检索的表达。"
            "要求保留原意，不要扩展不存在的信息。"
        )

        user_prompt = f"""
请将下面的问题改写成更适合文档检索的形式。

要求：
1. 保留原意
2. 更清晰、更完整
3. 不要引入原问题中没有的新事实
4. 只输出改写后的问题，不要解释

原问题：
{question}
"""
        return self.run_llm(system_prompt=system_prompt, user_prompt=user_prompt).strip()