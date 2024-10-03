import torch
from torch import nn
from transformers import TinyLlamaConfig, TinyLlamaModel

# Define model configuration
config = TinyLlamaConfig(
    vocab_size=30522,  # Size of the tokenizer's vocabulary
    hidden_size=512,   # Size of the embeddings and hidden layers
    num_attention_heads=8,
    num_hidden_layers=6,
    intermediate_size=2048,
)

# Initialize the model
model = TinyLlamaModel(config)