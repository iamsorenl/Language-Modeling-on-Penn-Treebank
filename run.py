import argparse
from datasets import load_dataset
from train import train_model, evaluate_test_set
import torch
import numpy as np
import random

# Set seeds for reproducibility
seed = 24
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main(output_file):
    # Load the Penn Treebank dataset
    ptb = load_dataset('ptb-text-only/ptb_text_only', trust_remote_code=True)
    train_data = ptb['train']
    val_data = ptb['validation']
    test_data = ptb['test']

    # Extract sentences from the dataset splits
    train_sentences = [example['sentence'] for example in train_data]
    val_sentences = [example['sentence'] for example in val_data]
    test_sentences = [example['sentence'] for example in test_data]

    # Define training configuration
    config = {
        'seq_len': 40,  # Maximum sequence length for padding/truncation
        'd_model': 256,  # Transformer model dimension
        'num_epochs': 10,  # Number of training epochs
        'lr': 1e-3,  # Learning rate
        'batch_size': 16,  # Batch size for training
        'min_freq': 5,  # Minimum frequency for including a word in the vocabulary
    }

    # Train the model
    model, vocab = train_model(config, train_sentences, val_sentences)

    # Evaluate the model on the test set
    evaluate_test_set(config, model, vocab, test_sentences, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the language modeling script.")
    parser.add_argument('output_file', type=str, nargs='?', default='submission.csv', help='The output file to save the model.')
    args = parser.parse_args()
    main(args.output_file)