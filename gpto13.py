import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Ensure that the required models and datasets are installed
# !pip install transformers datasets

# Load the reference model and tokenizer
reference_model_name = 'Qwen/Qwen-7B'  # Replace with 'Qwen2.5-7B' if available
reference_tokenizer = AutoTokenizer.from_pretrained(reference_model_name, use_fast=False)
reference_model = AutoModelForCausalLM.from_pretrained(reference_model_name, device_map='auto')

# Load the model to be trained and its tokenizer
train_model_name = 'Qwen/Qwen-7B'  # Replace with 'Qwen2.5-0.5B' if available
train_tokenizer = AutoTokenizer.from_pretrained(train_model_name, use_fast=False)
train_model = AutoModelForCausalLM.from_pretrained(train_model_name, device_map='auto')

# Ensure that both tokenizers are compatible
assert reference_tokenizer.get_vocab() == train_tokenizer.get_vocab(), "Tokenizers are not compatible."

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
reference_model.to(device)
train_model.to(device)

# Load the Wikimedia dataset (e.g., WikiText dataset)
dataset_name = 'wikitext'
dataset_version = 'wikitext-103-raw-v1'
split = 'train'

# Load the dataset using Hugging Face Datasets
raw_datasets = load_dataset(dataset_name, dataset_version, split=split)
print(f"Loaded {len(raw_datasets)} examples from the {split} split of {dataset_name}")

# Optionally, use a subset of the dataset for faster experimentation
max_examples = 10000  # Adjust as needed
raw_datasets = raw_datasets.select(range(max_examples))

class ReferenceDataset(Dataset):
    def __init__(self, datasets, tokenizer, block_size, reference_model, device):
        self.examples = []
        self.reference_losses = []
        self.tokenizer = tokenizer
        self.reference_model = reference_model
        self.device = device
        self.block_size = block_size
        self.prepare_dataset(datasets)
        
    def prepare_dataset(self, datasets):
        self.reference_model.eval()
        with torch.no_grad():
            for example in datasets:
                text = example['text']
                # Skip empty lines
                if not text.strip():
                    continue
                # Tokenize and split into blocks
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) == 0:
                    continue
                for i in range(0, len(tokens), self.block_size):
                    block_tokens = tokens[i:i + self.block_size]
                    input_ids = torch.tensor(block_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
                    labels = input_ids.clone()

                    # Compute reference loss
                    outputs = self.reference_model(input_ids)
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss = loss.view(shift_labels.size())  # Shape: [1, seq_len - 1]

                    # Store input_ids and reference loss
                    self.examples.append(input_ids.squeeze(0).cpu())
                    self.reference_losses.append(loss.squeeze(0).cpu())
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx], self.reference_losses[idx]

def collate_fn(batch):
    input_ids_list, ref_losses_list = zip(*batch)
    input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=reference_tokenizer.pad_token_id)
    ref_losses_padded = pad_sequence(ref_losses_list, batch_first=True, padding_value=0.0)
    return input_ids_padded, ref_losses_padded

# Hyperparameters
block_size = 128  # Maximum sequence length for each example
batch_size = 4
num_epochs = 1  # Set to 1 for initial testing; adjust as needed
learning_rate = 1e-5

# Prepare the dataset and dataloader
print("Preparing the dataset...")
dataset = ReferenceDataset(raw_datasets, reference_tokenizer, block_size, reference_model, device)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
print("Dataset preparation complete.")

# Optimizer
optimizer = torch.optim.AdamW(train_model.parameters(), lr=learning_rate)

# Training loop
train_model.train()
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch+1}/{num_epochs}")
    for batch_idx, (input_ids, ref_losses) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        ref_losses = ref_losses.to(device)
        
        # Prepare inputs and labels for the model to train
        inputs = input_ids[:, :-1]  # Exclude last token
        labels = input_ids[:, 1:]   # Exclude first token

        # Forward pass through the model to train
        outputs = train_model(inputs)
        logits = outputs.logits
        shift_logits = logits.contiguous()
        shift_labels = labels.contiguous()

        # Compute training loss per token
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        training_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        training_loss = training_loss.view(shift_labels.size())  # Shape: [batch_size, seq_len - 1]

        # Compute excess loss: training loss minus reference loss
        excess_loss = training_loss - ref_losses

        # Flatten excess loss to select top 30%
        flat_excess_loss = excess_loss.view(-1)
        num_tokens = flat_excess_loss.size(0)
        top_k = max(int(num_tokens * 0.3), 1)  # Ensure at least one token is selected

        # Select indices of top 30% excess losses
        if num_tokens > 0:
            top_values, top_indices = torch.topk(flat_excess_loss, k=top_k)
            mask = torch.zeros_like(flat_excess_loss)
            mask[top_indices] = 1.0
            mask = mask.view(excess_loss.size())  # Shape: [batch_size, seq_len - 1]
        else:
            mask = torch.ones_like(excess_loss)

        # Zero out losses not in top 30%
        masked_loss = training_loss * mask
        # Handle case where mask.sum() is zero to avoid division by zero
        if mask.sum() > 0:
            final_loss = masked_loss.sum() / mask.sum()
        else:
            final_loss = torch.tensor(0.0, requires_grad=True).to(device)

        # Backpropagation
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {final_loss.item():.4f}")

    print(f"Epoch {epoch+1} complete.")

# Save the trained model
train_model.save_pretrained('path/to/save/trained_model')
train_tokenizer.save_pretrained('path/to/save/trained_tokenizer')