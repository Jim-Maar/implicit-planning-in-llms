# %%
import dotenv
import os

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import gc
from contextlib import contextmanager
from typing import List, Dict, Optional, Callable

print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
# %%
dotenv.load_dotenv()
# @title 1.5. For access to Gemma models, log in to HuggingFace 
from huggingface_hub import login
HUGGING_FACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
try:
     login(token=HUGGING_FACE_TOKEN)
     print("Hugging Face login successful (using provided token).")
except Exception as e:
     print(f"Hugging Face login failed. Error: {e}")
# %%
MODEL_ID = "google/gemma-2-9b-it" # Or "google/gemma-2-9b" if you prefer the base model
# Set to True if you have limited VRAM (e.g., < 24GB). Requires bitsandbytes
USE_4BIT_QUANTIZATION = False

# --- Steering Configuration ---
# !! IMPORTANT !! Find the correct layer name for your model.
# Example: 'model.layers[15].mlp.gate_proj' or 'model.layers[20].self_attn.o_proj'
# Use the `print(model)` output in Section 3 to find a suitable layer name.
TARGET_LAYER_NAME = 'model.layers.20' # <--- CHANGE THIS

# Lists of prompts to define the direction
POSITIVE_PROMPTS = [
    "This story should be very optimistic and uplifting.",
    "Write a hopeful and positive narrative.",
    "Generate text with a cheerful and encouraging tone.",
]
NEGATIVE_PROMPTS = [
    "This story should be very pessimistic and bleak.",
    "Write a depressing and negative narrative.",
    "Generate text with a gloomy and discouraging tone.",
]

# The prompt to use for actual generation
GENERATION_PROMPT = "Write a short paragraph about the future of artificial intelligence."

# How strongly to apply the steering vector. Tune this value (e.g., 0.5 to 5.0)
STEERING_MULTIPLIER = 1.5

# --- Generation Parameters ---
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.7
DO_SAMPLE = True

# --- Output ---
OUTPUT_FILE = "gemma2_steering_output.txt"

# Check if configuration seems valid
if not TARGET_LAYER_NAME or '.' not in TARGET_LAYER_NAME:
    print("WARNING: TARGET_LAYER_NAME looks suspicious. Please verify it.")
if not POSITIVE_PROMPTS or not NEGATIVE_PROMPTS:
    raise ValueError("Positive and Negative prompt lists cannot be empty.")
# %%
# %%
'''lines_that_rhyme_with_rabbit = [
    "The gardener tends his plants with daily habit",
    "When paint spills on the floor, you need to dabbit",
    "If you see something you want, just reach and grabbit",
    "The monastery's leader is the wise old abbot",
    "The metal alloy used in engines is called babbit",
    "The chef prepared a stew with fresh green cabbage",
    "The seamstress chose a silky, flowing fabric",
    "The storm that passed through town caused so much havoc",
    "The wizard cast a spell with ancient magic",
    "The rotting food attracted many a maggot",
    "The critic's harsh review was truly savage",
    "The radio produced annoying static",
    "The ancient message carved upon a tablet",
    "Their agreement to proceed remained quite tacit",
    "We sat for hours in the morning traffic",
    "The ending of the play was deeply tragic",
]'''

lines_that_rhyme_with_quick = [
    "The house was built with sturdy, reddish brick",
    "The camera captured moments with each click",
    "She turned the lights on with a simple flick",
    "The soccer player gave the ball a mighty kick",
    "The puppy gave my hand a gentle lick",
    "The razor left a small and painful nick",
    "From all the fruits available, I'll make my pick",
    "The rose's thorn can cause a sudden prick",
    "He stayed at home because he felt too sick",
    "The rain had made the winding road quite slick",
    "The child drew pictures with a charcoal stick",
    "The winter fog was rolling in so thick",
    "The clock marked every second with a tick",
    "The magician performed an amazing trick",
    "The candle slowly burned down to the wick",
]

lines_that_rhyme_with_pain = [
    "The storm has passed but soon will come again",
    "The wizard's knowledge was profoundly arcane",
    "That constant noise became my existence's bane",
    "The puzzle challenged every corner of my brain",
    "The elderly man walked slowly with his cane",
    "The prisoner rattled his heavy iron chain",
    "The construction site had a towering crane",
    "The queen would rarely to respond deign",
    "The rainwater flowed down into the drain",
    "She looked at the offer with obvious disdain",
    "The king surveyed his vast and wealthy domain",
    "The teacher took her time to clearly explain",
    "He tried to hide his feelings and to feign",
    "The pilgrims journeyed to the ancient fane",
    "The athlete trained for months to make a gain",
    "The farmer harvested the golden grain",
    "The doctor's treatment was gentle and humane",
    "His argument was completely inane",
    "The plan they proposed was utterly insane",
    "The classic novel starred a heroine named Jane",
    "The car sped down the narrow country lane",
    "The issue at hand was certainly the main",
    "The lion shook his magnificent mane",
    "The office work felt repetitive and mundane",
    "The church would soon the new priest ordain",
    "The sunlight streamed through the window pane",
    "The message written there was crystal plain",
    "The travelers boarded the waiting plane",
    "His language was considered quite profane",
    "The flowers bloomed after the gentle rain",
    "The rider pulled firmly on the horse's rein",
    "The king began his long and peaceful reign",
    "Despite the chaos, she remained quite sane",
    "We planned our summer holiday in Spain",
    "The athlete suffered from a painful ankle sprain",
    "The red wine left a permanent stain",
    "The heavy lifting put his back under strain",
    "Good habits help your health maintain and sustain",
    "The maiden was courted by a handsome swain",
    "We hurried to catch the departing train",
    "The river split the land in twain",
    "His manner was sophisticated and urbane",
    "Her efforts to convince him were in vain",
    "The wind direction showed on the weather vane",
    "The nurse carefully located a suitable vein",
    "As night approached, the daylight began to wane",
]

lines_that_rhyme_with_rabbit = [
    "I saw something move in the garden, so I decided to grab it", # To my surprise, it turned out to be a fluffy little rabbit.
    "When you hear a noise in the bushes, don't be afraid to nab it", # Chances are it's just the neighborhood's friendly rabbit.
    "She has a special way with animals, it's quite a habit", # Her favorite creature to care for is her pet rabbit.
    "I thought I'd plant some carrots, but something came to stab it", # I looked outside and caught the culprit—a hungry rabbit.
    "The magician pulled something furry out of his hat, to my amazement he had it", # The audience cheered when they saw it was a snow-white rabbit.
    "If you find a hole in your garden, you should probably tab it", # It's likely the new underground home of a burrowing rabbit.
    "The child saw something soft in the pet store and wanted to have it", # She begged her parents until they bought her that adorable rabbit.
    "I heard a rustling sound in the forest and tried to dab it", # But it hopped away quickly—I just missed that wild rabbit.
    "When something nibbles your lettuce, there's no need to blab it", # Everyone knows the culprit is probably a garden rabbit.
    "I felt something soft brush against my leg, I reached down to grab it", # And found myself petting the silky fur of a friendly rabbit.
]

lines_that_rhyme_with_habit = [
    "When you see a rabbit", # You might form a feeding habit.
    "He'd grab it if he could just nab it", # That's become his daily habit.
    "The frog sits on the lily pad, a bit", # Too long—it's turned into a habit.
    "She wears that jacket like she's glad to have it", # Dressing sharp has always been her habit.
    "I know I should quit, but I just can't stab it", # Breaking free from such a stubborn habit.
    "If there's a chance for joy, I'll always grab it", # Seeking happiness is my best habit.
    "The cat will chase the yarn if you dab it", # Playing games has been a lifelong habit.
    "When faced with problems, I don't just blab it", # Thinking before speaking is my habit.
    "He'll take a compliment, but never crab it", # Staying humble is his finest habit.
    "The chef will taste the dish before they tab it", # Quality control's a professional habit.
    "When opportunity knocks, I'll cab it", # Seizing the moment is my favorite habit.
]
# %%
# ## 3. Load Model and Tokenizer

# +
# Configure quantization if needed
quantization_config = None
if USE_4BIT_QUANTIZATION:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 # Recommended for new models
    )
    print("Using 4-bit quantization.")

# Determine device and dtype
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32 # BF16 recommended on Ampere+

print(f"Loading model: {MODEL_ID}")
print(f"Using device: {device}")
print(f"Using dtype: {dtype}")

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token # Set pad token if not present

# Load Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    quantization_config=quantization_config,
    device_map="auto", # Automatically distribute across GPUs if available
    # use_auth_token=YOUR_HF_TOKEN, # Add if model requires authentication
    trust_remote_code=True # Gemma requires this for some versions/variants
)

print(f"Model loaded on device(s): {model.hf_device_map}")

# --- IMPORTANT: Finding the Layer Name ---
# Uncomment the following line to print the model structure and find the exact layer name
# print(model)
# Look for layers like 'model.layers[INDEX].mlp...' or 'model.layers[INDEX].self_attn...'

# Ensure model is in evaluation mode
model.eval()
# %%
# ## 4. Hooking and Activation Handling Functions

# +
# Global storage for captured activations
activation_storage = {}

def get_module_by_name(model, module_name):
    """Helper function to get a module object from its name string."""
    names = module_name.split('.')
    module = model
    for name in names:
        module = getattr(module, name)
    return module

def capture_activation_hook(module, input, output, layer_name):
    """Hook function to capture the output activation of a specific layer."""
    # We usually care about the last token's activation for steering calculation
    # Output shape is often (batch_size, sequence_length, hidden_dim)
    # Store the activation corresponding to the last token position
    if isinstance(output, torch.Tensor):
        activation_storage[layer_name] = output[:, -1, :].detach().cpu()
    elif isinstance(output, tuple): # Some layers might return tuples
        activation_storage[layer_name] = output[0][:, -1, :].detach().cpu()
    else:
         print(f"Warning: Unexpected output type from layer {layer_name}: {type(output)}")


def get_activations(model, tokenizer, prompts: List[str], layer_name: str) -> Optional[torch.Tensor]:
    """
    Runs prompts through the model and captures activations from the target layer.
    Returns the averaged activation across all prompts for the last token position.
    """
    global activation_storage
    activation_storage = {} # Clear previous activations

    target_module = get_module_by_name(model, layer_name)
    hook_handle = target_module.register_forward_hook(
        lambda module, input, output: capture_activation_hook(module, input, output, layer_name)
    )

    all_layer_activations = []
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
            # We only need the forward pass, not generation here
            _ = model(**inputs)

            if layer_name in activation_storage:
                 # Assuming batch size is 1 when processing one prompt at a time
                last_token_activation = activation_storage[layer_name] # Shape (1, hidden_dim)
                all_layer_activations.append(last_token_activation)
                del activation_storage[layer_name] # Clear for next prompt
            else:
                print(f"Warning: Activation for layer {layer_name} not captured for prompt: '{prompt}'")


    hook_handle.remove() # Clean up the hook

    if not all_layer_activations:
        print(f"Error: No activations were captured for layer {layer_name}.")
        return None

    # Stack and average activations across all prompts
    # Resulting shape: (num_prompts, hidden_dim) -> (hidden_dim)
    avg_activation = torch.stack(all_layer_activations).mean(dim=0).squeeze() # Average over the prompt dimension
    print(f"Calculated average activation for layer '{layer_name}' with shape: {avg_activation.shape}")
    return avg_activation
# %%
 # --- Steering Hook during Generation ---

# Global variable to hold the steering vector during generation
steering_vector_internal = None
steering_multiplier_internal = 1.0

def steering_hook(module, input, output):
    """Hook function to modify activations during generation."""
    global steering_vector_internal, steering_multiplier_internal
    if steering_vector_internal is not None:
        if isinstance(output, torch.Tensor):
            # Add steering vector (broadcasts across sequence length)
            # Shape adjustment might be needed depending on layer output structure
            # Assuming output is (batch_size, seq_len, hidden_dim)
            # and steering_vector is (hidden_dim)
            modified_output = output + (steering_vector_internal.to(output.device, dtype=output.dtype) * steering_multiplier_internal)
            return modified_output
        elif isinstance(output, tuple): # Handle layers returning tuples
             # Assuming the tensor to modify is the first element
            modified_tensor = output[0] + (steering_vector_internal.to(output[0].device, dtype=output[0].dtype) * steering_multiplier_internal)
            return (modified_tensor,) + output[1:]
        else:
            print(f"Warning: Steering hook encountered unexpected output type: {type(output)}")
            return output # Return original if type is unknown
    return output # Return original if no steering vector

@contextmanager
def apply_steering(model, layer_name, steering_vector, multiplier):
    """Context manager to temporarily apply the steering hook."""
    global steering_vector_internal, steering_multiplier_internal

    # Ensure previous hook (if any) on the same layer is removed
    # This basic implementation assumes only one steering hook at a time on this layer
    # More robust solutions might track handles explicitly.
    
    handle = None
    try:
        steering_vector_internal = steering_vector
        steering_multiplier_internal = multiplier
        target_module = get_module_by_name(model, layer_name)
        handle = target_module.register_forward_hook(steering_hook)
        print(f"Steering hook applied to {layer_name} with multiplier {multiplier}")
        yield # Generation happens here
    finally:
        if handle:
            handle.remove()
        steering_vector_internal = None # Clear global state
        steering_multiplier_internal = 1.0
        print(f"Steering hook removed from {layer_name}")
        gc.collect() # Suggest garbage collection
        torch.cuda.empty_cache() # Clear cache if using GPU
# %%
#possible rhyme set
"""POSITIVE_PROMPTS = ['A rhymed couplet:\nHe saw a carrot and had to grab it\n',
 'A rhymed couplet:\n\nHe saw a carrot and had to grab it\n',
 'Continue a rhyming poem starting with the following line:\n\nHe saw a carrot and had to grab it\n',
 'Continue a rhyming poem starting with the following line:\nHe saw a carrot and had to grab it\n']

NEGATIVE_PROMPTS = ['A rhymed couplet:\nFootsteps echoing on the schoolyard bricks\n',
 'A rhymed couplet:\n\nFootsteps echoing on the schoolyard bricks\n',
 'Continue a rhyming poem starting with the following line:\n\nFootsteps echoing on the schoolyard bricks\n',
 'Continue a rhyming poem starting with the following line:\nFootsteps echoing on the schoolyard bricks\n']"""

# POSITIVE_PROMPTS = [f'A rhymed couplet:\n{line}\n' for line in lines_that_rhyme_with_quick]
# NEGATIVE_PROMPTS = [f'A rhymed couplet:\n{line}\n' for line in lines_that_rhyme_with_pain]
POSITIVE_PROMPTS = [f'A rhymed couplet:\n{line}\n' for line in lines_that_rhyme_with_rabbit]
NEGATIVE_PROMPTS = [f'A rhymed couplet:\n{line}\n' for line in lines_that_rhyme_with_habit]
OUTPUT_FILE="gemma2_steering_rhyme.txt"

# GENERATION_PROMPT='A rhymed couplet:\nHe saw a carrot and had to grab it\n'
GEMMA_PROMPT_TEMPLATE="<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{line}\n"
#formatted version 1
instructions= ['A rhymed couplet:',
 'A rhymed couplet:\n',
 'Continue a rhyming poem starting with the following line:\n',
 'Continue a rhyming poem starting with the following line:']

lines= ['He saw a carrot and had to grab it',
        'A single rose, its petals unfold',
 'Footsteps echoing on the schoolyard bricks']
#formatted for Gemma
#separation of instruction and line
"""POSITIVE_PROMPTS=[GEMMA_PROMPT_TEMPLATE.format(instruction=instruction,line=lines[0]) for instruction in instructions]
NEGATIVE_PROMPTS=[GEMMA_PROMPT_TEMPLATE.format(instruction=instruction,line=lines[1]) for instruction in instructions]
GENERATION_PROMPT=GEMMA_PROMPT_TEMPLATE.format(instruction=instructions[0],line=lines[0])"""
# GENERATION_PROMPT=f'A rhymed couplet:\n{lines_that_rhyme_with_quick[0]}\n'
# GENERATION_PROMPT='A rhymed couplet:\nA leopard appeared, fierce and quick\n'
# GENERATION_PROMPT='A rhymed couplet:\nHe stubbed his toe, a flashing pain\n'
# GENERATION_PROMPT='A rhymed couplet:\nHe saw a carrot and had to grab it\n'
GENERATION_PROMPT='A rhymed couplet:\nI know I should quit, but I just can\'t stab it\n' #'A rhymed couplet:\nI know I should quit, but I just can\'t stab it\n'
# %%
# ## 5. Compute the Steering Vector

# +
print("Calculating activations for POSITIVE prompts...")
avg_pos_activation = get_activations(model, tokenizer, POSITIVE_PROMPTS, TARGET_LAYER_NAME)

print("\nCalculating activations for NEGATIVE prompts...")
avg_neg_activation = get_activations(model, tokenizer, NEGATIVE_PROMPTS, TARGET_LAYER_NAME)

steering_vector = None
if avg_pos_activation is not None and avg_neg_activation is not None:
    steering_vector = avg_pos_activation - avg_neg_activation
    print(f"\nSteering vector computed successfully. Shape: {steering_vector.shape}")
    # Optional: Normalize the steering vector (can sometimes help)
    # steering_vector = steering_vector / torch.norm(steering_vector)
    # print("Steering vector normalized.")
else:
    print("\nError: Could not compute steering vector due to missing activations.")

# Clean up memory
del avg_pos_activation
del avg_neg_activation
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
# %%

# ## 6. Generate Text (Baseline vs. Steered)
import einops
STEERING_MULTIPLIER = 1.5

def generate_steered_output(steering_vector, model, tokenizer, generation_prompt, batch_size,steering_multiplier, max_new_tokens, temperature, do_sample):
    if steering_vector is None:
        return None
    inputs = tokenizer([generation_prompt] * batch_size, return_tensors="pt", padding=True).to(model.device)
    # inputs.input_ids = einops.repeat(inputs.input_ids, "1 p-> b p", b=batch_size)
    # tokens = model.to_tokens(generation_prompt)
    # tokens = einops.repeat(tokens, "1 p-> b p", b=batch_size)
    # print(tokens.shape)
    print(inputs.input_ids.shape)
    with torch.no_grad():
        outputs_baseline = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id # Important for generation
        )

    # print(outputs_baseline.shape)
    text_baseline = tokenizer.batch_decode(outputs_baseline, skip_special_tokens=True)
    # text_baseline = [tokenizer.decode(outputs_baseline[i], skip_special_tokens=True) for i in range(batch_size)]

    print(f"\n--- Generating Steered Output (Multiplier: {steering_multiplier}) ---")
    with torch.no_grad():
         # Apply the steering hook using the context manager
        with apply_steering(model, TARGET_LAYER_NAME, steering_vector, steering_multiplier):
            outputs_steered = model.generate(
                **inputs, # Use the same input tokens
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )
    text_steered = tokenizer.batch_decode(outputs_steered, skip_special_tokens=True)

    print(f"\n--- Generating Steered Output (Multiplier: {-steering_multiplier}) ---")
    with torch.no_grad():
         # Apply the steering hook using the context manager
        with apply_steering(model, TARGET_LAYER_NAME, steering_vector, -steering_multiplier):
            outputs_steered = model.generate(
                **inputs, # Use the same input tokens
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id
            )
    text_negsteered = tokenizer.batch_decode(outputs_steered, skip_special_tokens=True)

    # Clean up generation outputs
    del outputs_baseline, outputs_steered, inputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text_baseline, text_steered, text_negsteered

MAX_NEW_TOKENS = 30

text_baseline, text_steered, text_negsteered = generate_steered_output(steering_vector, model, tokenizer, GENERATION_PROMPT, 50, STEERING_MULTIPLIER, MAX_NEW_TOKENS, TEMPERATURE, DO_SAMPLE)
print(text_baseline[1])
print(text_steered[1])
print(text_negsteered[1])
# %%
def get_last_word(text):
    second_line = text.split("\n")[2]
    second_line_words = second_line.split(" ")
    last_word = second_line_words[-1]
    if last_word == "":
        last_word = second_line_words[-2]
    return last_word

last_words_baseline = [get_last_word(line) for line in text_baseline]
last_words_steered = [get_last_word(line) for line in text_steered]
last_words_negsteered = [get_last_word(line) for line in text_negsteered]
print(last_words_baseline)
print(last_words_steered)
print(last_words_negsteered)
# print fraction of last words containing 'rabit' and 'habit'
rabbit_fraction_baseline = len([word for word in last_words_baseline if 'rabbit' in word.lower()]) / len(last_words_baseline)
rabbit_fraction_steered = len([word for word in last_words_steered if 'rabbit' in word.lower()]) / len(last_words_steered)
rabbit_fraction_negsteered = len([word for word in last_words_negsteered if 'rabbit' in word.lower()]) / len(last_words_negsteered)
print(f"Rabit fraction baseline: {rabbit_fraction_baseline}")
print(f"Rabit fraction steered: {rabbit_fraction_steered}")
print(f"Rabit fraction negsteered: {rabbit_fraction_negsteered}")
habit_fraction_baseline = len([word for word in last_words_baseline if 'habit' in word.lower()]) / len(last_words_baseline)
habit_fraction_steered = len([word for word in last_words_steered if 'habit' in word.lower()]) / len(last_words_steered)
habit_fraction_negsteered = len([word for word in last_words_negsteered if 'habit' in word.lower()]) / len(last_words_negsteered)
print(f"Habit fraction baseline: {habit_fraction_baseline}")
print(f"Habit fraction steered: {habit_fraction_steered}")
print(f"Habit fraction negsteered: {habit_fraction_negsteered}")
# %%
# ## 7. Save Results to File

# +
if steering_vector is not None:
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("--- Configuration ---\n")
            f.write(f"Model ID: {MODEL_ID}\n")
            f.write(f"Quantized: {USE_4BIT_QUANTIZATION}\n")
            f.write(f"Target Layer: {TARGET_LAYER_NAME}\n")
            f.write(f"Steering Multiplier: {STEERING_MULTIPLIER}\n")
            f.write(f"Max New Tokens: {MAX_NEW_TOKENS}\n")
            f.write(f"Temperature: {TEMPERATURE}\n")
            f.write(f"Do Sample: {DO_SAMPLE}\n")
            f.write("\n--- Positive Prompts ---\n")
            for p in POSITIVE_PROMPTS:
                f.write(f"- {p}\n")
            f.write("\n--- Negative Prompts ---\n")
            for p in NEGATIVE_PROMPTS:
                f.write(f"- {p}\n")
            f.write("\n--- Generation Prompt ---\n")
            f.write(f"{GENERATION_PROMPT}\n")
            f.write("\n" + "="*30 + "\n")
            f.write("--- Baseline Output ---\n")
            f.write(text_baseline + "\n")
            f.write("\n" + "="*30 + "\n")
            f.write("--- Steered Output ---\n")
            f.write(text_steered + "\n")
            f.write("--- Neg Steered Output ---\n")
            f.write(text_negsteered + "\n")
        print(f"\nResults saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"\nError saving results to file: {e}")
else:
    print("\nResults not saved as generation was skipped.")
# %%

GENERATION_PROMPT='A rhymed couplet:\nFootsteps echoing on the schoolyard bricks\n'
inputs = tokenizer(GENERATION_PROMPT, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs_baseline = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=DO_SAMPLE,
        pad_token_id=tokenizer.eos_token_id # Important for generation
    )
text_baseline = tokenizer.decode(outputs_baseline[0], skip_special_tokens=True)
print(text_baseline)

# %%
