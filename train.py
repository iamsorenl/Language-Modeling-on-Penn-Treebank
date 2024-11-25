import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from tqdm import tqdm
import csv
from model import build_transformer

# Function to collate sequences
def collate_fn(batch, seq_len):
    """
    Pads or truncates a batch of sequences to the specified length (seq_len).
    Assumes that 0 is the padding value for <pad>.
    """
    batch = [
        seq[:seq_len] if len(seq) > seq_len 
        else torch.cat([seq, torch.zeros(seq_len - len(seq), dtype=torch.long)])
        for seq in batch
    ]
    return torch.stack(batch)

# Function to create a padding mask
def padding_mask(inputs, pad_idx=0):
    """
    Create a padding mask to ignore <pad> tokens.
    Assumes <pad> is indexed at `pad_idx`.
    """
    # Create a mask where 1 means <unk> and 0 means <pad>
    return (inputs != pad_idx).unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_len)


# Function to build vocabulary
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

# Function to tokenize and encode sentences
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

# Dataset class for language modeling
class LanguageModelDataset(Dataset):
    def __init__(self, encoded_sentences):
        self.data = encoded_sentences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)

# Function to prepare datasets
def prepare_dataset(config, train_sentences, val_sentences):
    """
    Prepare train and validation datasets.
    """
    vocab = build_vocab(train_sentences, min_freq=config['min_freq'])
    encoded_train = tokenize_and_encode(train_sentences, vocab)
    encoded_val = tokenize_and_encode(val_sentences, vocab)

    train_dataset = LanguageModelDataset(encoded_train)
    val_dataset = LanguageModelDataset(encoded_val)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, config['seq_len'])
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        collate_fn=lambda batch: collate_fn(batch, config['seq_len'])
    )

    return train_loader, val_loader, vocab

# Function to compute perplexity
def compute_perplexity(loss):
    return math.exp(loss)

# Validation function with padded masking
def run_validation(model, validation_loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in validation_loader:
            inputs = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)

            seq_len = inputs.size(1)
            tgt_mask = padding_mask(seq_len).to(device)

            output = model.encode(inputs, tgt_mask)
            logits = model.project(output)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(validation_loader)
    return compute_perplexity(avg_loss)

# Training loop
def train_model(config, train_sentences, val_sentences):
    """
    Train the transformer model for language modeling.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use CUDA if available
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Use MPS if available (macOS with Apple Silicon)
    else:
        device = torch.device("cpu")  # Fallback to CPU
    print(f"Using device: {device} for training.")
    train_loader, val_loader, vocab = prepare_dataset(config, train_sentences, val_sentences)

    model = build_transformer(
        len(vocab), len(vocab), config['seq_len'], config['seq_len'], d_model=config['d_model']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>']).to(device)

    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in batch_iterator:
            inputs = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)

            seq_len = inputs.size(1)
            tgt_mask = padding_mask(seq_len).to(device)

            # Forward pass
            encoder_output = model.encode(inputs, tgt_mask)
            logits = model.project(encoder_output)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update progress bar
            batch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        val_ppl = run_validation(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_loss:.4f}, Validation Perplexity: {val_ppl:.4f}")

    return model, vocab

# Test set evaluation
def evaluate_test_set(config, model, vocab, test_sentences, output_file):
    """
    Evaluate the model on the test set and write perplexities to a CSV file.
    """
    # Device selection: prioritize CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use CUDA if available
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Use MPS if available (macOS with Apple Silicon)
    else:
        device = torch.device("cpu")  # Fallback to CPU
    print(f"Using device: {device} to evaluate test set.")
    model.eval()

    encoded_test = tokenize_and_encode(test_sentences, vocab)
    test_dataset = LanguageModelDataset(encoded_test)
    test_loader = DataLoader(
    test_dataset, 
    batch_size=1, 
    shuffle=False, 
    collate_fn=lambda batch: collate_fn(batch, config['seq_len'])
    )

    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>']).to(device)
    sequence_perplexities = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            seq_len = inputs.size(1)
            tgt_mask = padding_mask(seq_len).to(device)

            # Forward pass
            encoder_output = model.encode(inputs, tgt_mask)
            logits = model.project(encoder_output)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            ppl = compute_perplexity(loss.item())
            sequence_perplexities.append((i, ppl))

    # Write results to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'ppl'])
        for seq_id, ppl in sequence_perplexities:
            writer.writerow([seq_id, f"{ppl:.2f}"])

    avg_ppl = sum(ppl for _, ppl in sequence_perplexities) / len(sequence_perplexities)
    print(f"Average Test Perplexity: {avg_ppl:.4f}")
