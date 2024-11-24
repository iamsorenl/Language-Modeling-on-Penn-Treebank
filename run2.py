import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
from datasets import load_dataset
from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformer import TransformerLM
import time  # For timing

def format_time(seconds):
    """
    Convert seconds to a formatted string of minutes and seconds.
    """
    mins, secs = divmod(seconds, 60)
    return f"{int(mins)}m {secs:.2f}s"

def evaluate_and_save(model, data_loader, criterion, device, output_file):
    """
    Evaluate the model on the test dataset and save per-example perplexities.
    """
    model.eval()
    start_time = time.time()  # Start timing evaluation
    all_perplexities = []  # To store perplexities for each example
    total_loss = 0
    total_count = 0  # Count of valid tokens to compute average loss

    with torch.no_grad():
        for batch_idx, (input_seqs, target_seqs, pad_masks) in enumerate(data_loader):
            # Move data to the selected device
            input_seqs, target_seqs, pad_masks = input_seqs.to(device), target_seqs.to(device), pad_masks.to(device)

            # Forward pass
            logits = model(input_seqs, pad_masks)
            batch_size, seq_len, vocab_size = logits.shape

            for i in range(batch_size):
                # Extract logits and targets for each sequence
                seq_logits = logits[i, :-1, :]  # Exclude last token for logits
                seq_targets = target_seqs[i, 1:]  # Exclude first token for targets

                # Mask out padded tokens
                valid_indices = seq_targets != criterion.ignore_index
                seq_logits = seq_logits[valid_indices]
                seq_targets = seq_targets[valid_indices]

                # Skip entirely padded sequences
                if seq_targets.numel() == 0:
                    all_perplexities.append(float('nan'))  # Assign NaN if sequence is entirely padded
                    continue

                # Compute sequence loss
                seq_loss = criterion(seq_logits, seq_targets).item()
                seq_perplexity = torch.exp(torch.tensor(seq_loss)).item()

                all_perplexities.append(seq_perplexity)
                total_loss += seq_loss
                total_count += 1

    # Write results to the output file
    with open(output_file, "w") as f:
        f.write("ID,ppl\n")
        for idx, ppl in enumerate(all_perplexities):
            f.write(f"{idx},{ppl:.2f}\n")

    avg_loss = total_loss / total_count if total_count > 0 else float('nan')
    avg_perplexity = torch.exp(torch.tensor(avg_loss)) if total_count > 0 else float('nan')
    eval_time = time.time() - start_time  # End timing
    print(f"Evaluation completed in {format_time(eval_time)}")
    print(f"Saved results to {output_file}")
    return avg_loss, avg_perplexity

def evaluate(model, data_loader, criterion, device):
    """
    Evaluate the model on the validation/test dataset.
    """
    model.eval()
    total_loss = 0
    start_time = time.time()  # Start timing evaluation
    with torch.no_grad():
        for input_seqs, target_seqs, pad_masks in data_loader:
            # Move data to the selected device
            input_seqs, target_seqs, pad_masks = input_seqs.to(device), target_seqs.to(device), pad_masks.to(device)
            
            logits = model(input_seqs, pad_masks)
            logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            targets = target_seqs[:, 1:].contiguous().view(-1)

            loss = criterion(logits, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    perplexity = torch.exp(torch.tensor(avg_loss))
    eval_time = time.time() - start_time  # End timing
    print(f"Evaluation completed in {format_time(eval_time)}")
    return avg_loss, perplexity

def collate_fn(batch, vocab):
    """
    Custom collate function to prepare input, target sequences, and masks for DataLoader.
    """
    pad_idx = vocab['<pad>']
    s_idx = vocab['<s>']
    end_idx = vocab['</s>']

    input_seqs = []
    target_seqs = []
    pad_masks = []

    for sentence in batch:
        # Tokenize the sentence
        tokenized = [s_idx] + [vocab.get(word, vocab['<unk>']) for word in sentence.split()] + [end_idx]

        # Create input and target sequences
        input_seq = tokenized[:-1]  # Input stops before </s>
        target_seq = tokenized[1:]  # Target starts after <s>

        input_seqs.append(torch.tensor(input_seq))
        target_seqs.append(torch.tensor(target_seq))
        pad_masks.append(torch.ones(len(input_seq), dtype=torch.long))

    # Pad sequences to max length in batch
    input_seqs = pad_sequence(input_seqs, batch_first=True, padding_value=pad_idx)
    target_seqs = pad_sequence(target_seqs, batch_first=True, padding_value=pad_idx)
    pad_masks = pad_sequence(pad_masks, batch_first=True, padding_value=0)

    return input_seqs, target_seqs, pad_masks

class TokenizedDataset(Dataset):
    """
    Dataset class for tokenized sentences.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def build_vocab(sentences, min_freq=3):
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

def main(output_file):
    """
    Main function to train the model.
    """
    # Load the Penn Treebank dataset
    ptb = load_dataset('ptb-text-only/ptb_text_only', trust_remote_code=True)
    train_data = ptb['train']
    val_data = ptb['validation']
    test_data = ptb['test']

    # Extract sentences from the dataset splits
    train_sentences = [example['sentence'] for example in train_data]
    val_sentences = [example['sentence'] for example in val_data]
    test_sentences = [example['sentence'] for example in test_data]

    # Build vocabulary from training data
    vocab = build_vocab(train_sentences, min_freq=3)
    print(f"Vocabulary size: {len(vocab)}")

    # Wrap datasets in DataLoader for batching
    batch_size = 16  # Define batch size
    train_loader = DataLoader(
        TokenizedDataset(train_sentences), 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=lambda batch: collate_fn(batch, vocab)
    )
    val_loader = DataLoader(
        TokenizedDataset(val_sentences), 
        batch_size=batch_size, 
        collate_fn=lambda batch: collate_fn(batch, vocab)
    )
    test_loader = DataLoader(
        TokenizedDataset(test_sentences), 
        batch_size=batch_size, 
        collate_fn=lambda batch: collate_fn(batch, vocab)
    )

    # Define model parameters
    d_model = 128  # Embedding size
    n_head = 8  # Number of attention heads
    n_layer = 4  # Number of Transformer layers
    max_seq_len = max(len(seq.split()) for seq in train_sentences) + 2  # Add 2 for <s> and </s>

    # Device selection: prioritize CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use CUDA if available
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Use MPS if available (macOS with Apple Silicon)
    else:
        device = torch.device("cpu")  # Fallback to CPU
    print(f"Using device: {device}")

    # Initialize model
    model = TransformerLM(vocab_size=len(vocab), d_model=d_model, n_head=n_head, n_layer=n_layer, max_seq_len=max_seq_len).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 10
    train_start_time = time.time()  # Start timing training
    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()  # Time each epoch
        for input_seqs, target_seqs, pad_masks in train_loader:
            input_seqs, target_seqs, pad_masks = input_seqs.to(device), target_seqs.to(device), pad_masks.to(device)

            logits = model(input_seqs, pad_masks)
            logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            targets = target_seqs[:, 1:].contiguous().view(-1)

            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_time = time.time() - epoch_start_time
        val_loss, val_perplexity = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}, Train Loss: {loss.item()}, Val Loss: {val_loss}, Perplexity: {val_perplexity}, Epoch Time: {format_time(epoch_time)}")
    
    total_train_time = time.time() - train_start_time
    print(f"Training completed in {format_time(total_train_time)}")

    # Evaluate on the test set and save per-example perplexities
    test_loss, test_perplexity = evaluate_and_save(model, test_loader, criterion, device, output_file)
    print(f"Test Loss: {test_loss}, Test Perplexity: {test_perplexity}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the language modeling script.")
    parser.add_argument('output_file', type=str, nargs='?', default='submission.csv', help='The output file to save the model.')
    args = parser.parse_args()
    main(args.output_file)
