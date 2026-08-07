import os
from groq import Groq
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
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        
    def generate_code(self, task: str) -> str:
        response = self.client.chat.completions.create(
            model="qwen/qwen3.6-27b",   # or another Groq-supported model
            temperature=0,
            max_completion_tokens=1500,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": task
                }
            ]
        )

        return response.choices[0].message.content.strip()