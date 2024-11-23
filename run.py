from datasets import load_dataset
from collections import Counter

def build_vocab(sentences, min_freq=3):
    counter = Counter()
    for sentence in sentences:
        tokens = sentence.split()
        counter.update(tokens)

    vocab = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}

    idx = len(vocab)

    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1

    return vocab

def tokenize_with_start_and_stop(sentence, vocab):
    # Add <s> and </s>
    sentence_with_tokens = "<s> " + sentence + " </s>"
    
    # Tokenize and map to vocabulary IDs
    return [vocab.get(word, vocab['<unk>']) for word in sentence_with_tokens.split()]


def main():
    ptb = load_dataset('ptb-text-only/ptb_text_only', trust_remote_code=True)
    train_data = ptb['train']
    val_data = ptb['validation']
    test_data = ptb['test']

    print("First couple sentences in train data:")
    print(train_data[0]['sentence'])
    print(train_data[1]['sentence'])

    print("\nFirst couple sentences in validation data:")
    print(val_data[0]['sentence'])
    print(val_data[1]['sentence'])

    print("\nFirst couple sentences in test data:")
    print(test_data[0]['sentence'])
    print(test_data[1]['sentence'])


if __name__ == "__main__":
    main()