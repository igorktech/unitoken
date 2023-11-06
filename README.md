# Unitoken
Mini character to Unicode tokenizer for character-level models

Inspired by [CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation](https://arxiv.org/pdf/2103.06874.pdf) Hugging Face [CanineTokenizer](https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/canine/tokenization_canine.py#L63) and Koziev [charcter-tokenizer](https://github.com/Koziev/character-tokenizer) 


## Tokenizer Setup

```
pip install git+https://github.com/igorktech/unitoken
```

## Usage

Simple tokenizer:
```py
from unitokenizer import SimpleUniTokenizer

# Test basic tokenization
tokenizer = SimpleUniTokenizer()
text = f"NLP is amazing!"

# Operates only with ids
encoded = tokenizer(text)
print("Encoded text", encoded)

decoded = tokenizer.decode(encoded)
print("Decoded text", decoded)
```


Output:

```
Encoded text [57346, 78, 76, 80, 32, 105, 115, 32, 97, 109, 97, 122, 105, 110, 103, 33]
Decoded text <s>NLP is amazing!
```

Hugging Face compatible tokenizer
```py
from unitokenizer import UniTokenizer

# Test basic tokenization
tokenizer = UniTokenizer()
text = f"NLP is amazing!"

encoded = tokenizer(text)
print("Encoded text", encoded)

decoded = tokenizer.decode(encoded["input_ids"])
print("Decoded text", decoded)
```

Output:

```
Encoded text 
{
    'input_ids': [57346, 78, 76, 80, 32, 105, 115, 32, 97, 109, 97, 122, 105, 110, 103, 33], 
    'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
Decoded text <s>NLP is amazing!
```

