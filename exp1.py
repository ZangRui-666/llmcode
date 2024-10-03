from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure model is in evaluation mode
model.eval()

# Sample input text
input_text = "The quick brown fox jumps over the lazy dog."
# Tokenize the input text
input_ids = tokenizer(input_text, return_tensors="pt").input_ids  # shape: (1, seq_len)

# GPT-like models (including TinyLlama) expect labels shifted by one, so we use the same input_ids as targets
labels = input_ids.clone()

# Forward pass to get logits (predictions)
with torch.no_grad():
    outputs = model(input_ids, labels=labels)
    logits = outputs.logits  # shape: (1, seq_len, vocab_size)
    loss = outputs.loss  # scalar loss averaged over tokens

# To compute per-token loss, we use CrossEntropyLoss with 'none' reduction
loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

# Flatten logits and labels for per-token loss computation
logits_flat = logits.view(-1, logits.size(-1))  # shape: (batch_size * seq_len, vocab_size)
labels_flat = labels.view(-1)  # shape: (batch_size * seq_len)

# Compute per-token loss
per_token_loss = loss_fn(logits_flat, labels_flat)  # shape: (batch_size * seq_len)

# Reshape per-token loss back to (batch_size, seq_len)
per_token_loss = per_token_loss.view(input_ids.size())  # shape: (1, seq_len)

# Print per-token loss for each token
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
for token, loss_value in zip(tokens, per_token_loss[0]):
    print(f"Token: {token}, Loss: {loss_value.item()}")



sorted_losses, _ = torch.sort(per_token_loss, descending=False)
threshold_index = int(0.2 * len(sorted_losses[0]))  # 20% of the length
percentile_threshold = sorted_losses[0][threshold_index].item()  # Get the loss value at the 20th percentile

print(f"20th percentile threshold loss: {percentile_threshold}")

# Step 2: Create a mask to keep tokens with loss above the 20th percentile threshold
token_mask = per_token_loss > percentile_threshold  # shape: (batch_size, seq_len)

# Apply the mask to drop tokens (keep tokens where mask is True)
filtered_input_ids = input_ids[token_mask]  # shape: (num_kept_tokens)

# Convert filtered tokens back to text
filtered_text = tokenizer.decode(filtered_input_ids)

print(f"\nOriginal Text: {input_text}")
print(f"Filtered Text (after dropping tokens with low loss): {filtered_text}")