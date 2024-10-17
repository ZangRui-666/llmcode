import sys
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
import torch.nn.functional as F
from tqdm import tqdm

# Configuration
MODEL_NAME = "huggingface/llama"  # Replace with the actual model name/path
DATASET_NAME = "wikipedia"  # Wikimedia dataset
BATCH_SIZE = 4  # Adjust based on your GPU memory
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 5e-5
EPOCHS = 3
TOP_PERCENT = 0.3  # Top 30%

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
model.to(device)
model.train()

# Load and preprocess dataset
# Here, we're using a subset for demonstration. Adjust as needed.
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

def tokenize_function(examples):
    return tokenizer(examples["text"], return_tensors="pt", truncation=True, padding='max_length', max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Convert to PyTorch tensors
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Create DataLoader
dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(dataloader) * EPOCHS // GRADIENT_ACCUMULATION_STEPS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps)



# Training loop
model.train()
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    epoch_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training")
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device) # shape (batch_size, seq_length)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        print("output.shape", outputs.loss.shape)
        logits = outputs.logits  # Shape: (batch_size, seq_length, vocab_size)
        print("logits.shape", logits.shape)
        print(logits)
        print("input_id.shape", input_ids.shape)

        logits_flat = logits.view(-1, logits.size(-1)) # shape: (batch_size, seq_length, vocab_size) -> (batch_size * seq_len, vocab_size)
        labels_flat = input_ids.view(-1)  # shape: (batch_size * seq_len)
        
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        per_token_loss = loss_fn(logits_flat, labels_flat)  # shape: (batch_size * seq_len)
        per_token_loss = per_token_loss.view(input_ids.size()) # shape: (1, seq_len)
        print("per_token_loss", per_token_loss.shape)
        print(per_token_loss)
        print("input_ids", input_ids)
        print("input_ids[0]", input_ids[0].shape, input_ids[0])
        sorted_losses, _ = torch.sort(per_token_loss, descending=False)

        if num_tokens == 0:
            continue  # Skip if no tokens to process
        k = int(num_tokens * TOP_PERCENT)
        if k == 0:
            k = 1  # Ensure at least one token is kept
        
        # Flatten the losses and filter out padding tokens
        losses_flat = token_losses.view(-1)
        attention_flat = shift_attention.view(-1)
        valid_losses = losses_flat[attention_flat == 1]
        
        if valid_losses.numel() == 0:
            continue  # Skip if no valid losses
        
        # Find the threshold
        threshold = torch.topk(valid_losses, k, largest=True, sorted=False).values.min()
        
        # Create a mask for top 30% losses
        mask = (token_losses >= threshold).float()
        
        # Apply the mask
        masked_losses = token_losses * mask
        
        # Compute the final loss
        if mask.sum() == 0:
            continue  # Avoid division by zero
        final_loss = masked_losses.sum() / mask.sum()
        
        # Backward pass
        final_loss.backward()
        epoch_loss += final_loss.item()
        
        # Gradient accumulation
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        progress_bar.set_postfix({"Loss": final_loss.item()})
    
    avg_epoch_loss = epoch_loss / len(dataloader)
    print(f"Average Loss for Epoch {epoch + 1}: {avg_epoch_loss}")

# Optionally, save the model
# model.save_pretrained("llama_finetuned_wikimedia")