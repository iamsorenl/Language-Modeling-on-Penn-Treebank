# Language Modeling on the Penn Treebank Dataset

This project implements a **Transformer-based Encoder Model** for autoregressive language modeling on the **Penn Treebank (PTB)** dataset using **PyTorch**. The model focuses on minimizing perplexity as a measure of predictive performance and leverages attention mechanisms for sequence dependency capture.

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Limitations and Future Work](#limitations-and-future-work)

---

## Project Overview

Autoregressive language modeling predicts the likelihood of a sequence of words, a task fundamental to natural language processing. This project aims to develop a **Transformer-based model** capable of predicting the next token in a sequence while minimizing perplexity.

Key features of the model include:

- A **Transformer Encoder Architecture** inspired by the "Attention is All You Need" paper.
- Custom **attention mechanisms** to capture dependencies between tokens.
- Evaluation of perplexity on the **Penn Treebank Dataset**, a standard benchmark in NLP.

---

## Requirements

To set up the project, ensure the following dependencies are installed:

- Python >= 3.11
- PyTorch
- numpy
- pandas
- scikit-learn
- matplotlib

### Installing Dependencies

Run the following command to install all required dependencies:

```bash
pip install -r requirements.txt
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/language-modeling.git
cd language-modeling
```

### 2. Create a Virtual Environment

To avoid package conflicts, use a virtual environment:

```bash
python3 -m venv venv
# Activate on MacOS/Linux
source venv/bin/activate
# Activate on Windows
venv\Scripts\activate
```

---

## Dataset Structure

The dataset consists of preprocessed text from the **Penn Treebank Dataset**, split into training, validation, and test sets. Ensure the following files are in the project directory:

- `ptb_train.txt`: Training dataset
- `ptb_valid.txt`: Validation dataset
- `ptb_test.txt`: Test dataset

### Example Data Format

**Training Dataset (ptb_train.txt):**

Each line represents a tokenized sentence:

```
<s> The company reported a profit of $3.5 million . </s>
<s> Mr. Smith announced a merger with Acme Corp . </s>
```

**Test Dataset (ptb_test.txt):**

```
<s> Show me the money </s>
```

---

## Usage

### Running the Training Pipeline

To train the model and evaluate its performance:

```bash
python run.py --train ptb_train.txt --valid ptb_valid.txt --test ptb_test.txt
```

This command will:

1. Train the Transformer model.
2. Evaluate it on the validation set.
3. Test it for perplexity on the test set.

### Output

Training and validation results, along with test perplexity, will be displayed in the console.

---

## Training the Model

### Architecture Details

The model uses the following components:

- **Embedding Layer:** Converts tokens into dense vectors with sinusoidal positional encodings.
- **Transformer Encoder:** Captures dependencies between tokens using multi-head attention.
- **Projection Layer:** Maps dense representations back to vocabulary size for token predictions.

### Hyperparameters

Key hyperparameters include:

- **Learning Rate:** 0.0001
- **Batch Size:** 32
- **Embedding Dimension:** 512
- **Sequence Length:** 50

---

## Evaluation

The model evaluates its performance using **perplexity**, defined as:

\[
PPL = \exp\left(\frac{1}{N} \sum*{i=1}^{N} \log p(w_i | w*{<i})\right)
\]

where \(N\) is the number of tokens, \(w*i\) is the current token, and \(w*{<i}\) represents the preceding tokens.

### Example Results

| Configuration         | Validation Perplexity | Test Perplexity |
| --------------------- | --------------------- | --------------- |
| Baseline              | 42.58                 | 83.35           |
| Smaller Model (Tuned) | 17.61                 | 39.11           |
| Larger Model          | 80.35                 | 134.72          |

---

## Hyperparameter Tuning

Experiments with different configurations revealed the following insights:

- Smaller models with faster learning rates generalize better on small datasets.
- Larger models are prone to overfitting on the Penn Treebank dataset due to limited size.

---

## Limitations and Future Work

- **Dataset Size:** The small size of the Penn Treebank dataset limits the ability to train larger models effectively.
- **Attention Mechanisms:** Additional techniques like sparse attention or local attention may improve efficiency.
- **Alternative Architectures:** Future work could explore transformer-based models like GPT or BERT.

---
