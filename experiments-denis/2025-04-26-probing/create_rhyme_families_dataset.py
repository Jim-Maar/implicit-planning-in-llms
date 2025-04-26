import os
import requests
import tqdm
import json
from dotenv import load_dotenv
from typing import List, Dict, Any
import concurrent.futures

# Load environment variables from .env file
load_dotenv("hf.env")

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

seed_words=["night","quick","room","shore","pain","unfold","sing","sleep","sand","eyes","bake","call"]

prompts=[f"list 10 words that rhyme with the word \"{w}\". Only list the words, separate them by commas. Do not uppercase the first word." for w in seed_words]
#rhymelists=generate_rollout(prompts,"anthropic/claude-3.7-sonnet")

    
rhymelists=['slick, brick, stick, trick, pick, kick, sick, tick, lick, flick',
 'night, bright, light, sight, fight, tight, might, right, flight, height',
 'doom, gloom, bloom, zoom, loom, tomb, groom, boom, flume, broom',
 'sleep, deep, heap, keep, leap, creep, reap, seep, steep, weep',
 'band, bland, brand, canned, fanned, gland, grand, hand, land, planned',
 'sing, ring, wing, sting, king, thing, bring, ding, fling, swing',
 'unfold, uphold, behold, foretold, retold, enrolled, cajoled, extolled, controlled, paroled',
 'shore, more, bore, core, door, floor, four, gore, lore, pour',
 'pain, cane, drain, gain, main, plain, rain, stain, train, vain',
 'skies, lies, size, wise, highs, flies, cries, ties, buys, rise',
 'bake, cake, fake, lake, make, rake, sake, take, wake, flake',
 'call, ball, fall, hall, wall, mall, tall, small, stall, crawl']

rhyme_lists=[[w.strip(',') for w in l.split()] for l in rhymelists]

labels=[]

for rhyme_list in rhyme_lists:
    labels.append(rhyme_list[0]+"_rhymes")

line_catalog={}
for label in labels: line_catalog[label]=[]

for rhyme_list in tqdm.tqdm(rhyme_lists):
    label=rhyme_list[0]+"_rhymes"
    line_prompts=[f"Write 10 short poetry lines end with the word \"{w}\". Each line should be at most ten words long. Do not end lines with a dot. Only list the lines, separate them by caret return." for w in rhyme_list]
    line_lists=generate_rollout(line_prompts,"anthropic/claude-3.7-sonnet")
    for lines in line_lists:
        lines=lines.split("\n")
        line_catalog[label]+=lines
        
with open("line_catalog.json",'w') as f:
    json.dump(line_catalog,f)
