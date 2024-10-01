import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 1: Load a pre-trained language model and tokenizer (for illustration, let's use 'gpt2')
model_name = "gpt2"  # Substitute with 'TinyLlama' or other small models if available
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 2: Tokenize the input text
text = "The quick brown fox jumps"
tokens = tokenizer(text, return_tensors='pt')

# Step 3: Perform a forward pass to get the logits (un-normalized probabilities)
with torch.no_grad():
    outputs = model(**tokens, labels=tokens['input_ids'])
    logits = outputs.logits  # Shape: [batch_size, sequence_length, vocab_size]

# Step 4: Compute the probabilities for the next token (use softmax)
# Let's compute the loss for predicting the word "jumps"
target_token_id = tokenizer.convert_tokens_to_ids("jumps")
token_position = -1  # Position of the last token 'jumps' in the sentence

# Softmax to convert logits to probabilities
probabilities = torch.softmax(logits[0, token_position - 1], dim=-1)

# Get the probability of the correct token 'jumps'
prob_of_target = probabilities[target_token_id].item()

# Step 5: Compute the reference loss (negative log-likelihood)
reference_loss = -torch.log(torch.tensor(prob_of_target)).item()

print(f"Reference loss for token 'jumps': {reference_loss:.4f}")
