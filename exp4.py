import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np

# Step 1: Load a pre-trained language model and tokenizer
model_name = "gpt2"  # Replace with 'TinyLlama' if available
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Make sure the model is in evaluation mode and doesn't compute gradients
model.eval()

# Step 2: Load a dataset (e.g., a Wikipedia dataset)
# We'll use the 'wikitext' dataset as an example (you can use any other large dataset)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Step 3: Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], return_tensors='pt', truncation=True, padding=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

# Step 4: Define a DataLoader to handle batching
dataloader = DataLoader(tokenized_dataset, batch_size=8)  # Adjust batch size according to your system memory

# Step 5: Function to compute the reference loss for a batch of data
def compute_loss_for_batch(batch):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']

    with torch.no_grad():
        # Pass the input through the model
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()  # Exclude the last token for loss calculation
        shift_labels = input_ids[..., 1:].contiguous()    # Shift labels by one
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

        # Calculate per-token loss
        losses = loss_fct(shift_logits.view(-1, model.config.vocab_size), shift_labels.view(-1))
        losses = losses.view(input_ids.size(0), -1)  # Reshape into [batch_size, sequence_length]

    return losses

# Step 6: Process the entire dataset and compute the total loss
all_losses = []
for batch in dataloader:
    batch_loss = compute_loss_for_batch(batch)
    all_losses.append(batch_loss)

# Convert the losses list to a tensor
all_losses_tensor = torch.cat(all_losses, dim=0)

# Step 7: Calculate the total and average reference loss across all tokens in the dataset
total_loss = all_losses_tensor.sum().item()
average_loss = all_losses_tensor.mean().item()

print(f"Total reference loss: {total_loss}")
print(f"Average reference loss per token: {average_loss:.4f}")
