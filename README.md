# NLP_task

## Dataset
The **CNN / DailyMail** Dataset is an English-language dataset containing just over 300k unique news articles as written by journalists at CNN and the Daily Mail. The current version supports both extractive and abstractive summarization, though the original version was created for machine reading and comprehension and abstractive question answering.

### Data Splits
The CNN/DailyMail dataset has 3 splits: train, validation, and test. Below are the statistics for Version 3.0.0 of the dataset.

| Dataset Split | Number of Instances in Split |
|---------------|------------------------------|
| Train         | 287,113                      |
| Validation    | 13,368                       |
| Test          | 11,490                       |

### Preprocessing
The dataset is preprocessed using a tokenizer from Hugging Face's `tokenizers` library. The preprocessing ensures that the input and target texts are tokenized, padded, and truncated appropriately for training a model.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("t5-small")
```
This initializes a tokenizer from the pretrained T5 model (t5-small).

```python

    def preprocess_function(input_text):
    # Truncate or pad inputs and outputs
    inputs = ["summarize: " + article for article in input_text["article"]]
    targets = input_text["highlights"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    
    # Adjust labels to ignore padding during loss computation
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```
The preprocessing function returns a dictionary `model_inputs` with the following keys:
 - `input_ids`: Tokenized and padded inputs.
 - `attention_mask`: Mask indicating the non-padded portions of the input.
 - `labels`: Tokenized and padded targets.

## Attention Mechanism Overview
The `Attention` class implements a simple attention mechanism for sequence-to-sequence tasks, such as machine translation or text summarization. This mechanism helps the decoder focus on specific parts of the input sequence during each step of decoding.

### Key Features of the Implementation

1. This attention mechanism is adaptable to various seq2seq models, such as machine translation or text generation.
2. A single learnable vector (`self.v`) and linear layer (`self.attn`) are used to compute attention scores efficiently.
3. The `attention_weights` output allows you to visualize which parts of the input the decoder focuses on during generation.
