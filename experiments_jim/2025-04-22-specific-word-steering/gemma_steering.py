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
import einops
import numpy as np

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

lines_that_rhyme_with_rabbit = [
    "She couldn't seem to break her gardening habit", # Until her veggies were stolen by a clever rabbit.
    "He developed quite an interesting habit", # Of leaving carrots for the neighbor's pet rabbit.
    "The monk maintained his meditation habit", # While outside his window hopped a curious rabbit.
    "I tried to quit my late-night snacking habit", # When I spotted in my kitchen a midnight rabbit.
    "The farmer stuck to his early rising habit," # And caught sight of a dawn-grazing rabbit.
    "My daughter formed an adorable habit", # Of reading bedtime stories to her stuffed rabbit.
    "The writer maintained her daily writing habit", # Creating tales about a mischievous rabbit.
    "The painter couldn't shake her artistic habit", # Her favorite subject was a snow-white rabbit.
    "She picked up the peculiar habit", # Of leaving garden notes addressed to a rabbit.
    "He kept up his wholesome forest walking habit", # Often spotting the same cotton-tailed rabbit.
    "The boy acquired a strange collecting habit", # Of items shaped like his favorite animal: rabbit.
    "The chef developed an experimental cooking habit", # Inspired by watching a munching wild rabbit.
]

lines_that_rhyme_with_rabbit = [
    "When I found a small, trembling rabbit", # Caring for animals became my habit.
    "She darted through the garden like a rabbit", # Looking for treats had become her habit.
    "He claimed he could pull a hat from a rabbit", # Showing off magic tricks was his daily habit.
    "The children giggled as they chased the rabbit", # Running through meadows became their favorite habit.
    "I planted carrots to attract a rabbit", # Gardening in spring is my cherished habit.
    "My thoughts multiply faster than a rabbit", # Overthinking has become my worst habit.
    "The speedy win went to the tortoise, not the rabbit", # Victory comes from persistence, not just habit.
    "In the moonlight hopped a silver rabbit", # Stargazing at night is now my habit.
    "They built a cozy hutch for their new rabbit", # Creating homes for pets is a wonderful habit.
    "The chef prepared a savory stew with rabbit", # Cooking wild game had become his habit.
]



# lines_that_rhyme_with
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


def capture_activation_hook_fast(module, input, output, layer_name):
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


def get_activations_fast(model, tokenizer, prompts: List[str], layer_name: str) -> Optional[torch.Tensor]:
    """
    Runs prompts through the model and captures activations from the target layer.
    Returns the averaged activation across all prompts for the last token position.
    """
    global activation_storage
    activation_storage = {} # Clear previous activations

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

    target_module = get_module_by_name(model, layer_name)
    hook_handle = target_module.register_forward_hook(
        lambda module, input, output: capture_activation_hook_fast(module, input, output, layer_name)
    )

    with torch.no_grad():
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        # We only need the forward pass, not generation here
        _ = model(**inputs)

        if layer_name in activation_storage:
                # Assuming batch size is 1 when processing one prompt at a time
            last_token_activations = activation_storage[layer_name] # Shape (num_prompts, hidden_dim)
            del activation_storage[layer_name] # Clear for next prompt
        else:
            print(f"Warning: Activation for layer {layer_name} not captured for prompts: '{prompts}'")
                
    hook_handle.remove() # Clean up the hook

    # Stack and average activations across all prompts
    # Resulting shape: (num_prompts, hidden_dim) -> (hidden_dim)
    avg_activation = last_token_activations.mean(dim=0).squeeze() # Average over the prompt dimension
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
def apply_steering(model, layer, steering_vector, multiplier):
    """Context manager to temporarily apply the steering hook."""
    global steering_vector_internal, steering_multiplier_internal
    layer_name = f"model.layers.{layer}"

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

def generate_steered_output(steering_vector, model, tokenizer, generation_prompt, batch_size, layer=20, steering_multiplier=STEERING_MULTIPLIER):
    inputs = tokenizer([generation_prompt] * batch_size, return_tensors="pt", padding=True).to(model.device)
    if steering_vector is None:
        print(inputs.input_ids.shape)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=DO_SAMPLE,
                pad_token_id=tokenizer.eos_token_id # Important for generation
            )
    else:
        with torch.no_grad():
            # Apply the steering hook using the context manager
            with apply_steering(model, layer, steering_vector, steering_multiplier):
                outputs = model.generate(
                    **inputs, # Use the same input tokens
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    do_sample=DO_SAMPLE,
                    pad_token_id=tokenizer.eos_token_id,
                )
    text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    del outputs, inputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return text

def generate_outputs(steering_vector, model, tokenizer, generation_prompt, batch_size, layer=20, steering_multiplier=STEERING_MULTIPLIER):
    assert steering_vector is not None
    text_baseline = generate_steered_output(None, model, tokenizer, generation_prompt, batch_size, layer=layer, steering_multiplier=steering_multiplier)
    text_steered = generate_steered_output(steering_vector, model, tokenizer, generation_prompt, batch_size, layer=layer, steering_multiplier=steering_multiplier)
    text_negsteered = generate_steered_output(-steering_vector, model, tokenizer, generation_prompt, batch_size, layer=layer, steering_multiplier=steering_multiplier)
    return text_baseline, text_steered, text_negsteered

def get_last_word(text):
    lines = text.split("\n")
    if len(lines) < 3:
        print(f"Failed to get last word: {text}")
        return ""
    second_line = lines[2]
    second_line_words = second_line.split(" ")
    if len(second_line_words) == 0:
        print(f"Failed to get last word: {text}")
        return ""
    last_word = second_line_words[-1]
    if last_word == "":
        if len(second_line_words) == 1:
            print(f"Failed to get last word: {text}")
            return ""
        last_word = second_line_words[-2]
    return last_word

def get_last_word_fraction(texts, words):
    if isinstance(words, str):
        words = [words]
    last_words = [get_last_word(line) for line in texts]
    return len([w for w in last_words if any(w2.lower() in w.lower() for w2 in words)]) / len(last_words)

def get_suggestiveness(lines, word1, word2, batch_size=100):
    print(batch_size)
    prompts = get_prompts(lines)
    suggestiveness_scores = []
    for prompt in prompts:
        couplet = generate_steered_output(None, model, tokenizer, prompt, batch_size)
        suggestiveness_word1 = get_last_word_fraction(couplet, word1)
        suggestiveness_word2 = get_last_word_fraction(couplet, word2)
        if suggestiveness_word1 == 0 and suggestiveness_word2 == 0:
            continue
        suggestiveness = suggestiveness_word1 - suggestiveness_word2
        suggestiveness_scores.append(suggestiveness)
        print(f"Prompt: {prompt}, Suggestiveness {word1}: {suggestiveness_word1}, Suggestiveness {word2}: {suggestiveness_word2}, Suggestiveness: {suggestiveness}")
    suggestiveness_scores = np.array(suggestiveness_scores)
    return suggestiveness_scores

def get_prompts(lines):
    return [f'A rhymed couplet:\n{line}\n' for line in lines]
# %%
"""suggestiveness_scores_rabbit = get_suggestiveness(lines_that_rhyme_with_rabbit, 'rabbit', 'habit', batch_size=500)
suggestiveness_scores_habit = get_suggestiveness(lines_that_rhyme_with_habit, 'habit', 'rabbit', batch_size=500)
print(f'Suggestiveness rabbit: {suggestiveness_scores_rabbit.mean()}')
print(f'Suggestiveness habit: {suggestiveness_scores_habit.mean()}')
high_rabbit_suggestiveness_indices = np.where(suggestiveness_scores_rabbit > 0.15)[0]
high_habit_suggestiveness_indices = np.where(suggestiveness_scores_habit > 0.15)[0]
high_rabbit_suggestiveness_indices = high_rabbit_suggestiveness_indices.tolist()
high_habit_suggestiveness_indices = high_habit_suggestiveness_indices.tolist()
print(high_rabbit_suggestiveness_indices)
print(high_habit_suggestiveness_indices)"""
# %% 
# high_rabbit_suggestiveness_indices = [0, 1, 2, 7]
# high_habit_suggestiveness_indices = [0, 3, 5, 6]
# lines_that_rhyme_with_rabbit_suggestive = [lines_that_rhyme_with_rabbit[i] for i in high_rabbit_suggestiveness_indices]
# lines_that_rhyme_with_habit_suggestive = [lines_that_rhyme_with_habit[i] for i in high_habit_suggestiveness_indices]
# %%
POSITIVE_PROMPTS = get_prompts(lines_that_rhyme_with_rabbit)
NEGATIVE_PROMPTS = get_prompts(lines_that_rhyme_with_habit)
# POSITIVE_PROMPTS = get_prompts(lines_that_rhyme_with_rabbit_suggestive)
# NEGATIVE_PROMPTS = get_prompts(lines_that_rhyme_with_habit_suggestive)
# POSITIVE_PROMPTS = get_prompts(lines_that_rhyme_with_quick)
# NEGATIVE_PROMPTS = get_prompts(lines_that_rhyme_with_pain)
# GENERATION_PROMPT=f'A rhymed couplet:\n{lines_that_rhyme_with_quick[0]}\n'
# GENERATION_PROMPT='A rhymed couplet:\nA leopard appeared, fierce and quick\n'
# GENERATION_PROMPT='A rhymed couplet:\nHe stubbed his toe, a flashing pain\n'
GENERATION_PROMPT='A rhymed couplet:\nHe saw a carrot and had to grab it\n'
# GENERATION_PROMPT='A rhymed couplet:\nI know I should quit, but I just can\'t stab it\n' #'A rhymed couplet:\nI know I should quit, but I just can\'t stab it\n'
# %%
# positive_words = [line.split(" ")[-1] for line in lines_that_rhyme_with_quick]
# negative_words = [line.split(" ")[-1] for line in lines_that_rhyme_with_pain]
positive_words = ["rabbit"]
negative_words = ["habit"]
# %%
# ## 5. Compute the Steering Vector
def get_steering_vector(model, tokenizer, positive_prompts, negative_prompts, layer=20):
    target_layer_name = f"model.layers.{layer}"
    print("Calculating activations for POSITIVE prompts...")
    avg_pos_activation = get_activations(model, tokenizer, positive_prompts, target_layer_name)

    print("\nCalculating activations for NEGATIVE prompts...")
    avg_neg_activation = get_activations(model, tokenizer, negative_prompts, target_layer_name)

    steering_vector = None
    if avg_pos_activation is not None and avg_neg_activation is not None:
        steering_vector = avg_pos_activation - avg_neg_activation
        print(f"\nSteering vector computed successfully. Shape: {steering_vector.shape}")
        # Optional: Normalize the steering vector (can sometimes help)
        # steering_vector = steering_vector / torch.norm(steering_vector)
        # print("Steering vector normalized.")
    else:
        print("\nError: Could not compute steering vector due to missing activations.")
    del avg_pos_activation
    del avg_neg_activation
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return steering_vector

def get_steering_vector_fast(model, tokenizer, positive_prompts, negative_prompts, layer=20):
    target_layer_name = f"model.layers.{layer}"
    print("Calculating activations for POSITIVE prompts...")
    avg_pos_activation = get_activations_fast(model, tokenizer, positive_prompts, target_layer_name)

    print("\nCalculating activations for NEGATIVE prompts...")
    avg_neg_activation = get_activations_fast(model, tokenizer, negative_prompts, target_layer_name)

    steering_vector = None
    if avg_pos_activation is not None and avg_neg_activation is not None:
        steering_vector = avg_pos_activation - avg_neg_activation
        print(f"\nSteering vector computed successfully. Shape: {steering_vector.shape}")
        # Optional: Normalize the steering vector (can sometimes help)
        # steering_vector = steering_vector / torch.norm(steering_vector)
        # print("Steering vector normalized.")
    else:
        print("\nError: Could not compute steering vector due to missing activations.")
    del avg_pos_activation
    del avg_neg_activation
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return steering_vector

steering_vector = get_steering_vector_fast(model, tokenizer, POSITIVE_PROMPTS, NEGATIVE_PROMPTS, layer=20)
# steering_vector = get_steering_vector_fast(model, tokenizer, POSITIVE_PROMPTS, NEGATIVE_PROMPTS, layer=20)
print(steering_vector.shape)
# Clean up memory
# %%
# ## 6. Generate Text (Baseline vs. Steered)
STEERING_MULTIPLIER = 1.5
LAYER = 20
MAX_NEW_TOKENS = 40

text_baseline, text_steered, text_negsteered = generate_outputs(steering_vector, model, tokenizer, GENERATION_PROMPT, 1000, layer=LAYER, steering_multiplier=STEERING_MULTIPLIER)
print("Baseline:")
print(text_baseline[1])
print("Steered:")
print(text_steered[1])
print("Negatively steered:")
print(text_negsteered[1])
# %%
# ## 6.5: Compute Rabbit and Habit likelyhoods
last_words_baseline = [get_last_word(line) for line in text_baseline]
last_words_steered = [get_last_word(line) for line in text_steered]
last_words_negsteered = [get_last_word(line) for line in text_negsteered]
print(last_words_baseline)
print(last_words_steered)
print(last_words_negsteered)
# print fraction of last words containing 'rabit' and 'habit'
positive_words_fraction_baseline = get_last_word_fraction(text_baseline, positive_words)
positive_words_fraction_steered = get_last_word_fraction(text_steered, positive_words)
positive_words_fraction_negsteered = get_last_word_fraction(text_negsteered, positive_words)
print(f"Positive words fraction baseline: {positive_words_fraction_baseline}")
print(f"Positive words fraction steered: {positive_words_fraction_steered}")
print(f"Positive words fraction negsteered: {positive_words_fraction_negsteered}")
negative_words_fraction_baseline = get_last_word_fraction(text_baseline, negative_words)
negative_words_fraction_steered = get_last_word_fraction(text_steered, negative_words)
negative_words_fraction_negsteered = get_last_word_fraction(text_negsteered, negative_words)
print(f"Negative words fraction baseline: {negative_words_fraction_baseline}")
print(f"Negative words fraction steered: {negative_words_fraction_steered}")
print(f"Negative words fraction negsteered: {negative_words_fraction_negsteered}")
# %%
suggestiveness_scores_rabbit = get_suggestiveness(lines_that_rhyme_with_rabbit_suggestive, 'rabbit', 'habit', batch_size=500)
suggestiveness_scores_habit = get_suggestiveness(lines_that_rhyme_with_habit_suggestive, 'habit', 'rabbit', batch_size=500)
print(f"Suggestiveness rabbit: {suggestiveness_scores_rabbit.mean()}")
print(f"Suggestiveness habit: {suggestiveness_scores_habit.mean()}")
# %%

# ## 7. Try out different Layers and Magnitudes for steering
layers = np.linspace(15, 25, 11).astype(int)
magnitudes = np.linspace(0.5, 2.0, 4)
rabbit_vs_habit_likelyhood = np.zeros((len(layers), len(magnitudes)))
for i, layer in enumerate(layers):
    steering_vector = get_steering_vector(model, tokenizer, POSITIVE_PROMPTS, NEGATIVE_PROMPTS, layer)
    for j, magnitude in enumerate(magnitudes):
        text_steered = generate_steered_output(steering_vector, model, tokenizer, GENERATION_PROMPT, 1000, layer, magnitude)
        rabbit_vs_habit_likelyhood[i, j] = get_last_word_fraction(text_steered, positive_words) - get_last_word_fraction(text_steered, negative_words)
# %%
# plot heatmap with plotly
import plotly.express as px
fig = px.imshow(rabbit_vs_habit_likelyhood, x=magnitudes, y=layers, color_continuous_scale='Viridis')
fig.show()
# %%
# Plot matplotlib heatmap (y axis should go from 15 (down) to 25 (up))
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 10))
plt.imshow(rabbit_vs_habit_likelyhood, cmap='viridis', origin='lower', extent=[0.5, 2.0, 15, 25], aspect='auto')
# plt.colorbar(im, shrink=0.8, pad=0.02)
# Set specific x-axis ticks
x_ticks = [0.5, 1.0, 1.5, 2.0]
plt.xticks(x_ticks)

# You can also set specific y-axis ticks if needed
y_ticks = np.arange(15, 26, 5)  # [15, 20, 25]
plt.yticks(y_ticks)

plt.colorbar()
plt.title('Rabbit vs Habit Likelihood')
plt.xlabel('Magnitude')
plt.ylabel('Layer')
# plt.tight_layout()
plt.show()
# %%
