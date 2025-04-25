import os
import json

from utils.llm_utils import generate_rollout

model_id = "anthropic/claude-3-7-sonnet"

generation_prompt_rhymed = """Please generate 100 rhymed two-line couplets using all sorts of different rhymes and topics. just continue the list below in the same format.

The morning sun cast shadows long and bright
The valley glowed with warm, inviting light

The album holds what memory cannot keep
The past preserved in images so deep

"""

generation_prompt_unrhymed = """Please generate 100 unrhymed two-line couplets using all sorts of different topics. just continue the list below in the same format.

Among the towering oaks and whispered winds
Generations carved their legacy into earth

Memories suspended in digital eternity
Silent witness to joy and passing time

"""

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define file paths for saving/loading the data using absolute paths
rhymed_file_path = os.path.join(script_dir, "rhymed_couplets.json")
unrhymed_file_path = os.path.join(script_dir, "unrhymed_couplets.json")

# Check if files exist and load them, otherwise generate and save
if os.path.exists(rhymed_file_path) and os.path.exists(unrhymed_file_path):
    # Load existing data
    with open(rhymed_file_path, 'r') as f:
        rollout_rhymed = json.load(f)
    
    with open(unrhymed_file_path, 'r') as f:
        rollout_unrhymed = json.load(f)
    
    print("Loaded existing data from files")
else:
    # Generate new data
    print("Generating rhymed data")
    rollout_rhymed = generate_rollout([generation_prompt_rhymed], model_id)[0]
    rollout_rhymed = rollout_rhymed.split("\n\n")[1:]
    print("Generating unrhymed data")
    rollout_unrhymed = generate_rollout([generation_prompt_unrhymed], model_id)[0]
    rollout_unrhymed = rollout_unrhymed.split("\n\n")[1:]
    # Save the generated data
    with open(rhymed_file_path, 'w') as f:
        json.dump(rollout_rhymed, f)
    
    with open(unrhymed_file_path, 'w') as f:
        json.dump(rollout_unrhymed, f)
    
    print("Generated and saved new data to files")

print(rollout_rhymed)
print(rollout_unrhymed)

