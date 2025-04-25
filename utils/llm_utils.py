import os
import requests
import json
from dotenv import load_dotenv
from typing import List, Dict, Any
import concurrent.futures

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def generate_rollout(prompts: List[str], model_id: str, reasoning=False) -> List[str]:
    """
    Generate n rollout responses for a given prompt using the specified model.
    Sends requests in parallel for efficiency.
    
    Args:
        prompts: List of prompts to generate rollouts for
        model_id: The model identifier for OpenRouter
    Returns:
        List of CoT responses
    """

    n = len(prompts)
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/your-username/your-repo",  # Replace with your actual site
        "X-Title": "CoT Generator",
        "Content-Type": "application/json"
    }
    
    # Add CoT instruction to the prompt
    # cot_prompt = f"{prompt}\n\nLet's think through this step by step."
    
    def make_single_request(prompt: str):
        data = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,  # Adjust as needed for diversity
            "max_tokens": 4096   # Adjust as needed for response length
        }
        response = requests.post(
            url=url,
            headers=headers,
            data=json.dumps(data)
        )
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
        
        result = response.json()
        if reasoning:
            return result["choices"][0]["message"]["reasoning"]
        else:
            return result["choices"][0]["message"]["content"]
    
    # Use ThreadPoolExecutor to make parallel requests
    cots = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(n, 10)) as executor:
        # Submit n requests - FIXED: Use executor.submit correctly
        future_to_request = {executor.submit(make_single_request, prompts[i]): i for i in range(n)}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_request):
            result = future.result()
            if result:
                cots.append(result)
    
    return cots
