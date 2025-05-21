# %%
rhyme_families = {
    "ing": [
        "bring", "cling", "ding", "fling", "king", "ping", "ring", "sing", 
        "sling", "spring", "sting", "string", "swing", "thing", "wing", "zing", 
        "bling", "wring", "sping"
    ],
    
    "air": [
        "air", "bare", "bear", "blare", "care", "chair", "dare", "fair", 
        "fare", "flare", "glare", "hair", "hare", "lair", "mare", "pair", 
        "pare", "rare", "scare", "share", "snare", "spare", "square", "stare", 
        "swear", "tear", "wear", "where", "stair", "prayer", "there", "their"
    ],
    
    "ip": [
        "blip", "chip", "clip", "dip", "drip", "flip", "grip", "hip", 
        "kip", "lip", "nip", "pip", "quip", "rip", "ship", "sip", 
        "skip", "slip", "snip", "strip", "tip", "trip", "whip", "zip", 
        "equip", "flip", "gyp", "script"
    ],
    
    "oat": [
        "boat", "coat", "dote", "float", "gloat", "goat", "moat", "note", 
        "oat", "quote", "rote", "stoat", "throat", "tote", "vote", "wrote", 
        "bloat", "gloat", "promote", "remote", "denote", "devote"
    ],
    
    "ird": [
        "bird", "curd", "gird", "herd", "nerd", "slurred", "spurred", "stirred", 
        "third", "word", "absurd", "blurred", "deferred", "deterred", "inferred", 
        "occurred", "preferred", "referred", "transferred", "concurred"
    ],
    
    "ee": [
        "bee", "brie", "chi", "cree", "dee", "fee", "flee", "free", 
        "glee", "gree", "he", "key", "knee", "lee", "me", "pea", 
        "plea", "quay", "sea", "see", "she", "ski", "spree", "tea", 
        "tee", "thee", "three", "tree", "we", "wee", "ye", "agree", 
        "decree", "degree", "foresee", "trainee", "trustee", "jubilee"
    ],
    
    "ight": [
        "bight", "bite", "blight", "bright", "cite", "dight", "fight", "flight", 
        "fright", "height", "kite", "knight", "light", "might", "mite", "night", 
        "plight", "quite", "right", "rite", "sight", "site", "slight", "spite", 
        "tight", "trite", "white", "write", "alight", "contrite", "delight", "excite", 
        "ignite", "incite", "indite", "invite", "polite", "recite", "unite"
    ],
    
    "ake": [
        "ache", "bake", "brake", "break", "cake", "drake", "fake", "flake", 
        "jake", "lake", "make", "quake", "rake", "sake", "shake", "slake", 
        "snake", "stake", "take", "wake", "awake", "betake", "forsake", "mistake", 
        "partake", "retake"
    ],
    
    "ow": [
        "blow", "bow", "crow", "flow", "glow", "grow", "know", "low", 
        "mow", "row", "show", "slow", "snow", "sow", "stow", "throw", 
        "tow", "bestow", "below", "elbow", "fellow", "follow", "hollow",
        "mellow", "narrow", "shadow", "shallow", "window", "winnow", "yellow"
    ],
    
    "it": [
        "it", "bit", "chit", "fit", "flit", "grit", "hit", "kit", "knit", 
        "lit", "mitt", "nit", "pit", "quit", "sit", "skit", "slit", 
        "spit", "split", "tit", "twit", "whit", "wit", "writ", "admit", 
        "commit", "emit", "habit", "hermit", "omit", "permit", "rabbit", 
        "remit", "submit", "transmit"
    ]
}

good_word_pairs = {
    "ing" : ["king", "ring"],
    "air" : ["bear", "chair"],
    "ip" : ["ship", "chip"],
    "oat" : ["goat", "boat"],
    "ird" : ["bird", "word"],
    "ee" : ["bee", "tree"],
    "ight" : ["light", "night"],
    "ake" : ["snake", "rake"],
    "ow" : ["snow", "crow"],
    "it" : ["rabbit", "habit"]
}

good_word_pairs_different_rhyme_family = {
    ('ing', 'oat') : ["king", "goat"],
    ('ee', 'ow'): ["bee", "crow"],
    ('oat', 'ake') : ["boat", "rake"],
    ('ing', 'ake') : ["king", "snake"],
    ('oat', 'ight') : ["goat", "light"],
    ('ird', 'it') : ["bird", "rabbit"],
    ('air', 'ip') : ["chair", "ship"],
    ('ip', 'it') : ["chip", "habit"],
    ('ow', 'it') : ["crow", "rabbit"],
    ('ing', 'it') : ["ring", "rabbit"],
}

# %%

# choose n random k tuples from the rhyme family keys
import random
import itertools

def random_k_tuples_from_all_combinations(items, k, n):
    """
    Generate n random k-tuples by sampling from all possible k-combinations.
    
    Args:
        items: The source list to sample from
        k: The size of each tuple
        n: The number of tuples to generate
    
    Returns:
        List of n random k-tuples
    """
    # Generate all possible k-combinations
    all_combinations = list(itertools.combinations(items, k))
    
    # Ensure we're not asking for more tuples than possible
    max_possible = len(all_combinations)
    if n > max_possible:
        raise ValueError(f"Cannot generate {n} unique tuples; only {max_possible} are possible")
    
    # Sample n combinations without replacement
    return random.sample(all_combinations, n)

# Example usage
rhyme_family_names = list(rhyme_families.keys())
random_tuples = random_k_tuples_from_all_combinations(rhyme_family_names, k=2, n=20)
random_pairs = [('ing', 'oat'), ('ee', 'ow'), ('oat', 'ake'), ('ing', 'ake'), ('oat', 'ight'), ('ird', 'it'), ('air', 'ip'), ('ip', 'it'), ('ow', 'it'), ('ing', 'it'), ('oat', 'it'), ('air', 'it'), ('ake', 'ow'), ('oat', 'ow'), ('ip', 'ow'), ('air', 'oat'), ('air', 'ird'), ('oat', 'ee'), ('ee', 'ake'), ('ird', 'ee')]
# random_pairs = [('ight', 'ick'), ('ate', 'ack'), ('ate', 'ound'), ('ick', 'all'), ('ight', 'ow'), ('ay', 'ight'), ('ick', 'ear'), ('ack', 'all'), ('ain', 'ack'), ('ow', 'all'), ('ain', 'all'), ('ate', 'ick'), ('ate', 'all'), ('ight', 'all'), ('ay', 'ick'), ('ay', 'ate'), ('ate', 'ight'), ('ound', 'all'), ('ain', 'ear'), ('ow', 'ick')]
print(random_pairs)
# %%

"""
The list should have the following format:

1. <first line>\n<second line>\n\n
2. <first line>\n<second line>\n\n
...\n\n
105. <first line>\n<second line>\n\n

Remember the couplet should be a rhyming couplet!

List:
"""

import os
import json

from utils.llm_utils import generate_rollout

model_id = "anthropic/claude-3-7-sonnet"
generation_prompt_template = """Please generate exactly 105 diverse two-line rhyming couplets, using the -{sound} rhyme family (e.g. {words}). Place your rhymes at the end of the lines.
"""

dataset_cfgs = {}
for rhyme_family_name in rhyme_families:
    dataset_cfgs[rhyme_family_name] = {
        "file_path": f"{rhyme_family_name}.json",
        "generation_prompt": generation_prompt_template.format(sound=rhyme_family_name, words=good_word_pairs[rhyme_family_name])
    }

# %%

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

def get_list(rollout):
    l = rollout.split("\n\n")[1:]
    l = [line.split("\n")[0] + "\n" for line in l]
    for i in range(len(l)):
        line = l[i]
        # check if line[0] is nubmer
        if line[0].isdigit():
            l[i] = l[i].split(". ")[1]
    return l

datasets = {}
# Define file paths for saving/loading the data using absolute paths
for dataset_name in dataset_cfgs:
    file_path = os.path.join(script_dir, dataset_cfgs[dataset_name]["file_path"])
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            dataset = json.load(f)
    else:
        print(f"Generating {dataset_name} data")
        rollout = generate_rollout([dataset_cfgs[dataset_name]["generation_prompt"]], model_id)[0]
        print(rollout[:300])
        dataset = get_list(rollout)
        with open(file_path, 'w') as f:
            json.dump(dataset, f)
    datasets[dataset_name] = dataset

# %%
for dataset_name in datasets:
    print(len(datasets[dataset_name]))

# %%
# save datasets
with open("rhyme_families.json", "w") as f:
    json.dump(datasets, f)

# %% 
model_id = "anthropic/claude-3-7-sonnet"
generation_prompt_template = """Please generate exactly 105 diverse two-line rhyming couplets ending with the word \"{word}\". Place your rhymes at the end of the lines. The first line should already heavily imply that the second line will end in \"{word}\".
"""

dataset_cfgs = {}
for rhyme_family_name in rhyme_families:
    for word in good_word_pairs[rhyme_family_name]:
        dataset_cfgs[word] = {
            "file_path": f"{word}.json",
            "generation_prompt": generation_prompt_template.format(word=word)
        }

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

datasets_word = {}
# Define file paths for saving/loading the data using absolute paths
for dataset_name in dataset_cfgs:
    file_path = os.path.join(script_dir, dataset_cfgs[dataset_name]["file_path"])
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            dataset = json.load(f)
    else:
        print(f"Generating {dataset_name} data")
        rollout = generate_rollout([dataset_cfgs[dataset_name]["generation_prompt"]], model_id)[0]
        print(rollout[:300])
        dataset = get_list(rollout)
        with open(file_path, 'w') as f:
            json.dump(dataset, f)
    datasets_word[dataset_name] = dataset

# %%

# save datasets
with open("rhyme_families_word.json", "w") as f:
    json.dump(datasets_word, f)

# %%

for dataset_name in datasets_word:
    print(len(datasets_word[dataset_name]))

# %%

for dataset_name in datasets_word:
    for i, line in enumerate(datasets_word[dataset_name]):
        if not (line[-2] == "," or line[-2] == "."):
            continue
        datasets_word[dataset_name][i] = line[:-2] + line[-1]

for dataset_name in datasets:
    for i, line in enumerate(datasets[dataset_name]):
        if not (line[-2] == "," or line[-2] == "."):
            continue
        datasets[dataset_name][i] = line[:-2] + line[-1]

# %%

print(datasets_word["king"][:5])
print(datasets["ing"][:5])

# %%

# save datasets
with open("rhyme_families_word.json", "w") as f:
    json.dump(datasets_word, f)

with open("rhyme_families.json", "w") as f:
    json.dump(datasets, f)

# %%

'''countries = ["United States", "China", "India", "United Kingdom", "Canada", "Australia", "Germany", "Japan", "France", "Brazil"]

generation_prompt_template = """Please generate exactly 105 diverse questions, where rhyming couplets ending with the word \"{word}\". Place your rhymes at the end of the lines. The first line should already heavily imply that the second line will end in \"{word}\".
"""
dataset_cfgs = {}
for rhyme_family_name in rhyme_families:
    for word in good_word_pairs[rhyme_family_name]:
        dataset_cfgs[word] = {
            "file_path": f"{word}.json",
            "generation_prompt": generation_prompt_template.format(word=word)
        }'''