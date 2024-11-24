import argparse
from datasets import load_dataset
from collections import Counter

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

    # Build vocabulary from training data
    min_freq = 0 # Define minimum frequency threshold
    vocab = build_vocab(train_sentences, min_freq=min_freq)
    print(f"Vocabulary size: {len(vocab)}")

    # Tokenize and encode the datasets
    encoded_train = tokenize_and_encode(train_sentences, vocab)
    encoded_val = tokenize_and_encode(val_sentences, vocab)
    encoded_test = tokenize_and_encode(test_sentences, vocab)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the language modeling script.")
    parser.add_argument('output_file', type=str, nargs='?', default='submission.csv', help='The output file to save the model.')
    args = parser.parse_args()
    main(args.output_file)