import os
from dotenv import load_dotenv

try:
    from groq import Groq
except ImportError as e:
    print(f"Error importing groq: {e}")
    print("Please install with: pip install groq")
    raise

from vectorstore.hf_store import HuggingFaceVectorStore

load_dotenv()

class ResumeRAG:
    def __init__(self, vectorstore: HuggingFaceVectorStore, reasoning_model: str = "llama-3.3-70b-versatile"):
        self.vs = vectorstore
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required. Please set it in your .env file.")
        self.client = Groq(api_key=groq_api_key)
        self.model = reasoning_model
        # Fallback model used automatically on rate-limit or capacity errors
        self.fallback_model = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")

    def answer_query(self, job_description: str, top_k: int = 3):
        docs = self.vs.similarity_search(job_description, top_k=top_k)
        context = "\n\n".join([f"Resume: {d['resume_id']}\n{d['content']}" for d in docs])

        prompt = f"""
You are a recruiter assistant. You have the following resume excerpts:

{context}

Here is the job description / search query:

{job_description}

Please:
- Rank the candidates (by resume_id).
- Explain why they match or do not match.
"""

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.2,
                max_tokens=768
            )
            return response.choices[0].message.content
        except Exception as e:
            # Auto-fallback on common rate limit/capacity errors
            err_text = str(e).lower()
            should_fallback = any(s in err_text for s in ["rate limit", "429", "capacity", "overloaded"]) 
            if not should_fallback:
                raise
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model=self.fallback_model,
                    temperature=0.1,
                    max_tokens=640
                )
                return response.choices[0].message.content
            except Exception:
                # Re-raise original error if fallback also fails
                raise
