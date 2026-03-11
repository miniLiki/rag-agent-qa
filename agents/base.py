from services.llm import LLMClient


class BaseAgent:
    def __init__(self):
        self.llm = LLMClient()

    def run_llm(self, system_prompt: str, user_prompt: str) -> str:
        return self.llm.chat(system_prompt=system_prompt, user_prompt=user_prompt)