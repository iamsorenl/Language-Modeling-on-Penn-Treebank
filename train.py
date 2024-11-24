import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
from collections import Counter
from tqdm import tqdm
import csv
from model import build_transformer

def build_vocab(sentences, min_freq=0):
    """
    Build a vocabulary from training sentences.
    """
    counter = Counter()
    for sentence in sentences:
        tokens = sentence.split()
        counter.update(tokens)

    # Initialize vocabulary with special tokens
    vocab = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
    idx = len(vocab)

    # Add words meeting the frequency threshold to the vocabulary
    for word, freq in counter.items():
        if word in vocab:  # Skip tokens that are already in the vocabulary
            continue
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1

    return vocab

def tokenize_and_encode(sentences, vocab):
    """
    Tokenize and encode sentences based on the vocabulary.
    """
    encoded_sentences = []
    for sentence in sentences:
        tokens = ['<s>'] + sentence.split() + ['</s>']
        encoded = [vocab.get(token, vocab['<unk>']) for token in tokens]
        encoded_sentences.append(encoded)
    return encoded_sentences

class LanguageModelDataset(Dataset):
    def __init__(self, encoded_sentences):
        self.data = encoded_sentences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)

def prepare_dataset(config, train_sentences, val_sentences):
    # Build vocabulary from training sentences
    vocab = build_vocab(train_sentences, min_freq=config['min_freq'])
    
    # Tokenize and encode training and validation sentences
    encoded_train = tokenize_and_encode(train_sentences, vocab)
    encoded_val = tokenize_and_encode(val_sentences, vocab)

    # Create PyTorch datasets
    train_dataset = LanguageModelDataset(encoded_train)
    val_dataset = LanguageModelDataset(encoded_val)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    return train_loader, val_loader, vocab


def compute_perplexity(loss):
    return math.exp(loss)

def run_validation(model, validation_loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in validation_loader:
            inputs = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)

            output = model(inputs, None)  # Adjust for your model's input
            loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(validation_loader)
    return compute_perplexity(avg_loss)
    
def train_model(config, train_sentences, val_sentences):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare data loaders
    train_loader, val_loader, vocab = prepare_dataset(config, train_sentences, val_sentences)

    # Initialize the model
    model = build_transformer(
        len(vocab), len(vocab), config['seq_len'], config['seq_len'], d_model=config['d_model']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>']).to(device)

    # Training loop
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            inputs = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)

            # Forward pass
            output = model(inputs, None)  # Adjust if model requires specific input format
            loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_perplexity = run_validation(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_loss:.4f}, Validation Perplexity: {val_perplexity:.4f}")

    # Return the trained model and vocabulary
    return model, vocab

def evaluate_test_set(config, model, vocab, test_sentences, output_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Tokenize and encode test sentences
    encoded_test = tokenize_and_encode(test_sentences, vocab)
    test_dataset = LanguageModelDataset(encoded_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>']).to(device)
    sequence_perplexities = []

    # Compute perplexity for each sequence
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            # Forward pass
            output = model(inputs, None)  # Adjust if model requires specific input format
            loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))

            # Compute perplexity
            ppl = compute_perplexity(loss.item())
            sequence_perplexities.append((i, ppl))

    # Write results to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'ppl'])
        for seq_id, ppl in sequence_perplexities:
            writer.writerow([seq_id, f"{ppl:.2f}"])

    # Compute and print average perplexity
    avg_ppl = sum(ppl for _, ppl in sequence_perplexities) / len(sequence_perplexities)
    print(f"Average Test Perplexity: {avg_ppl:.4f}")

