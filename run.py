import argparse
from datasets import load_dataset
from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence

def build_vocab(sentences, min_freq=3):
    '''
    Build a vocabulary from the given sentences.
    Words appearing less than `min_freq` times are excluded.
    Special tokens are added explicitly.
    '''
    counter = Counter()
    for sentence in sentences:
        tokens = sentence.split()
        counter.update(tokens)

    # Initialize vocabulary with special tokens
    vocab = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
    idx = len(vocab)

    # Add words meeting the frequency threshold to the vocabulary
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1

    return vocab

def tokenize_with_start_and_stop(sentence, vocab):
    '''
    Tokenize a sentence and map it to vocabulary IDs.
    Adds <s> at the start and </s> at the end of the sentence.
    Unknown words are mapped to <unk>.
    '''
    # Add <s> and </s> tokens
    sentence_with_tokens = "<s> " + sentence + " </s>"
    # Tokenize and map to vocabulary IDs
    return [vocab.get(word, vocab['<unk>']) for word in sentence_with_tokens.split()]

def pad_sequences(tokenized_sentences, pad_value=0):
    '''
    Pads a list of tokenized sentences to the same length.
    Returns a tensor with shape (batch_size, max_length).
    '''
    # Convert tokenized sentences to tensors
    tensor_sentences = [torch.tensor(sentence) for sentence in tokenized_sentences]
    # Pad the sequences
    padded_sequences = pad_sequence(tensor_sentences, batch_first=True, padding_value=pad_value)
    return padded_sequences

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
    vocab = build_vocab(train_sentences, min_freq=3)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Tokenize the sentences
    tokenized_train = [tokenize_with_start_and_stop(sentence, vocab) for sentence in train_sentences]
    tokenized_val = [tokenize_with_start_and_stop(sentence, vocab) for sentence in val_sentences]
    tokenized_test = [tokenize_with_start_and_stop(sentence, vocab) for sentence in test_sentences]

    # Pad the tokenized sequences
    padded_train = pad_sequences(tokenized_train, pad_value=vocab['<pad>'])
    padded_val = pad_sequences(tokenized_val, pad_value=vocab['<pad>'])
    padded_test = pad_sequences(tokenized_test, pad_value=vocab['<pad>'])

    print(f"First padded train sequence:\n{padded_train[0]}")
    print(f"First padded val sequence:\n{padded_val[0]}")
    print(f"First padded test sequence:\n{padded_test[0]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the language modeling script.")
    parser.add_argument('output_file', type=str, nargs='?', default='submission.csv', help='The output file to write results to.')
    args = parser.parse_args()
    main(args.output_file)
