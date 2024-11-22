from datasets import load_dataset

def main():
    ptb = load_dataset('ptb-text-only/ptb_text_only', trust_remote_code=True)
    print(ptb['train'][2]['sentence'])

if __name__ == "__main__":
    main()