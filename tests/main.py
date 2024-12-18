from Kala_Quantum import SimpleTokenizer, CodeLanguageModel, train_model
from Kala_Quantum.models import HybridModel
import torch
import json

# Path to JSON file
json_file = "pythonfull.json"

# Set vocabulary size
vocab_size = 1000  # Ensure consistency across tokenizer and models

# Initialize tokenizer
tokenizer = SimpleTokenizer()

# Build vocabulary from the actual dataset
with open(json_file, "r") as f:
    data = json.load(f)

# Extract code snippets from the dataset
code_samples = [subtopic["code"] for topic in data for subtopic in topic["subtopics"]]
tokenizer.build_vocab(code_samples, vocab_size=vocab_size)

# Debug tokenizer
print("Vocabulary Size:", len(tokenizer.vocab))
print("Sample Tokenized Input:", tokenizer(str(code_samples)))
print("Sample Vocabulary Mapping:", {k: v for k, v in list(tokenizer.vocab.items())})

# Initialize classical and hybrid models
classical_model = CodeLanguageModel(vocab_size=vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2)
hybrid_model = HybridModel(classical_model, num_qubits=4)

# Print model information
print("\nClassical Model:")
print(classical_model)

print("\nHybrid Model:")
print(hybrid_model)

# Train the hybrid model and save
train_model(json_file, tokenizer, hybrid_model, num_qubits=4, epochs=50, batch_size=32, lr=0.001)
