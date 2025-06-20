import requests
from crewai.llms.base_llm import BaseLLM


# Class to connect CrewAI with a local Ollama model via its REST API.
class OllamaLLM(BaseLLM):
    # Initialize wrapper
    def __init__(self, model_name="llama3", base_url="http://localhost:11434", temperature=0.7):
        self.model = model_name
        self.base_url = base_url
        self.temperature = temperature

    # Send prompt to local llm and generate response
    def call(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "stream": False
            }
        )

        # If request is successful, return API response
        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            # Raise error if POST request is unsuccessful
            raise RuntimeError(f"Ollama generation failed: {response.text}")
