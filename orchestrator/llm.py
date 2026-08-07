import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """
You are an autonomous Python data analyst.

Rules:
- Return ONLY executable Python code.
- DO NOT use markdown.
- DO NOT wrap the code in ``` or ```python.
- DO NOT add explanations.
- The first character of your response must be valid Python code.
"""

class LLM:
    def __init__(self):
        self.client = Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

    def generate_code(self, task: str) -> str:
        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            temperature=0,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": task
                }
            ]
        )

        return response.content[0].text.strip()