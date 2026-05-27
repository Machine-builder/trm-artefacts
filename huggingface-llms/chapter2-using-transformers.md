# Hugging Face LLM Course - Chapter 2. Using Transformers

[View the original course](https://huggingface.co/learn/llm-course/chapter2/1)

# Introduction

Transformer models are large models that contain millions to billions of parameters.
Training and deploying them is complex.
New models also appear frequently, which makes it difficult to test and compare them easily.

The Transformers library solves this problem by providing a single unified API for loading, training, and saving Transformer models.

Each model is self-contained instead of sharing components across files. This makes models easier to understand and modify.

This chapter shows a full example of how `pipeline()` works internally, explains the model API and configuration system, and also explains how tokenizers convert text into model inputs and back into readable text. It then shows how batching works and how tokenization functions operate at a higher level.

# Behind the pipeline

This section explains what happens inside a sentiment analysis pipeline. The pipeline combines preprocessing, model inference, and postprocessing into a single workflow.

```py
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)
```

## Preprocessing with a tokenizer

Transformer models cannot process raw text directly. A tokenizer converts text into numbers that the model can understand.

A tokenizer performs three main steps.

* It splits text into tokens such as words or subwords.
* It maps each token to an integer ID.
* It adds additional required inputs for the model.

The tokenizer must match the one used during pretraining. It is loaded from the Hugging Face Model Hub using `AutoTokenizer`.

For example, the default sentiment analysis model is `distilbert-base-uncased-finetuned-sst-2-english`. The tokenizer is loaded using this checkpoint name.

The tokenizer converts input text into a dictionary containing tensors. These tensors can be PyTorch tensors or NumPy arrays depending on configuration.

Tokenized inputs include `input_ids` and `attention_mask`. The `input_ids` represent token indices. The `attention_mask` indicates which tokens are real and which are padding.

## Going through the model

The model is loaded using `AutoModel`. This loads the base Transformer architecture without a task-specific head.

The model processes tokenized inputs and outputs hidden states. Hidden states are high-dimensional vectors that represent contextual meaning for each token.

These outputs are not final predictions. They are intermediate representations used by task-specific heads.

### A high-dimensional vector?

The output tensor has three dimensions.

* The batch size is the number of input sequences.
* The sequence length is the number of tokens in each sequence.
* The hidden size is the embedding dimension of each token.

A typical output shape is `[batch_size, sequence_length, hidden_size]`.

## Model heads: Making sense out of numbers

A model head converts hidden states into task-specific outputs. It usually contains one or more linear layers.

Different tasks use different heads such as:

* Sequence classification heads.
* Token classification heads.
* Question answering heads.
* Causal language modeling heads.

For sentiment analysis, a sequence classification head is used. This is done using `AutoModelForSequenceClassification`.

This model outputs logits instead of hidden states. The logits represent raw scores for each class.

## Postprocessing the output

Logits are not probabilities. They must be converted using a softmax function.
Softmax converts logits into probability distributions over labels.
The model configuration provides label mapping through `id2label`. This allows interpretation of output classes such as NEGATIVE and POSITIVE.
After applying softmax, the model outputs probabilities for each class. The highest probability is the predicted label.
This completes the full pipeline process. It includes tokenization, model inference, and postprocessing.

# Models

This section focuses on creating and using Transformer models. The `AutoModel` class is used to instantiate models from checkpoints.

## Creating a Transformer

Instantiation of an `AutoModel` loads a pretrained model from the Hugging Face Hub.

```py
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-cased")
```

Model weights and configuration are downloaded and cached locally. The checkpoint defines the architecture and pretrained parameters. The example uses BERT with 12 layers, hidden size 768, and 12 attention heads. The model is case-sensitive, so uppercase and lowercase tokens are treated differently.

The `AutoModel` class acts as a wrapper that selects the correct architecture automatically. The specific architecture class can also be used directly when known.

```py
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")
```

## Loading and saving

Model saving uses the same interface as tokenizers. The `save_pretrained()` method stores configuration and weights.

```py
model.save_pretrained("directory_on_my_computer")
```

Two files are created in the directory.

```
config.json
model.safetensors
```

The `config.json` file stores architecture settings and metadata. The `model.safetensors` file stores model weights. Both files are required to reconstruct the model.

Reloading uses the same `from_pretrained()` method.

```py
from transformers import AutoModel

model = AutoModel.from_pretrained("directory_on_my_computer")
```

Model sharing is supported through the Hugging Face Hub. Authentication is required for uploading models.

```py
from huggingface_hub import notebook_login

notebook_login()
```

CLI login is also supported.

```bash
huggingface-cli login
```

Model upload uses `push_to_hub()`.

```py
model.push_to_hub("my-awesome-model")
```

The model becomes accessible via the Hub under the user namespace.

```py
from transformers import AutoModel

model = AutoModel.from_pretrained("your-username/my-awesome-model")
```

The Hub supports versioning, partial updates, and documentation via model cards.

## Encoding text

Text input is converted into numerical form using a tokenizer. Tokenization splits text into tokens and maps them to integer IDs.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

encoded_input = tokenizer("Hello, I'm a single sentence!")
print(encoded_input)
```

Output structure contains token IDs, token type IDs, and attention masks.

```python
{'input_ids': [101, 8667, 117, 1000, 1045, 1005, 1049, 2235, 17662, 12172, 1012, 102], 
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

Fields are defined as follows.

* `input_ids` are numerical token representations.
* `token_type_ids` indicate sentence segmentation for paired inputs.
* `attention_mask` indicates valid tokens versus padding.

Decoded output reconstructs text including special tokens.

```py
tokenizer.decode(encoded_input["input_ids"])
```

```python
"[CLS] Hello, I'm a single sentence! [SEP]"
```

Special tokens are added automatically when required by the model.

Multiple inputs are supported as paired sequences.

```py
encoded_input = tokenizer("How are you?", "I'm fine, thank you!")
print(encoded_input)
```

```python
{'input_ids': [[101, 1731, 1132, 1128, 136, 102], [101, 1045, 1005, 1049, 2503, 117, 5763, 1128, 136, 102]], 
 'token_type_ids': [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
 'attention_mask': [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
```

Batch tokenization returns lists per sentence.

Tensor output is enabled with `return_tensors`.

```py
encoded_input = tokenizer("How are you?", "I'm fine, thank you!", return_tensors="pt")
print(encoded_input)
```

```python
{'input_ids': tensor([[  101,  1731,  1132,  1128,   136,   102],
         [  101,  1045,  1005,  1049,  2503,   117,  5763,  1128,   136,   102]]), 
 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
```

Tensor conversion requires equal sequence lengths. Raw token lists are not rectangular by default.

## Padding inputs

Padding aligns sequences to equal length using a padding token.

```py
encoded_input = tokenizer(
    ["How are you?", "I'm fine, thank you!"], padding=True, return_tensors="pt"
)
print(encoded_input)
```

```python
{'input_ids': tensor([[  101,  1731,  1132,  1128,   136,   102,     0,     0,     0,     0],
         [  101,  1045,  1005,  1049,  2503,   117,  5763,  1128,   136,   102]]), 
 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
```

Padding tokens use ID 0. Attention mask value 0 indicates ignored positions.

## Truncating inputs

Truncation limits sequence length to the model maximum. BERT supports up to 512 tokens.

```py
encoded_input = tokenizer(
    "This is a very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very long sentence.",
    truncation=True,
)
print(encoded_input["input_ids"])
```

Long sequences are cut to fit model constraints.

Combined padding and truncation enforces fixed tensor size.

```py
encoded_input = tokenizer(
    ["How are you?", "I'm fine, thank you!"],
    padding=True,
    truncation=True,
    max_length=5,
    return_tensors="pt",
)
print(encoded_input)
```

```python
{'input_ids': tensor([[  101,  1731,  1132,  1128,   102],
         [  101,  1045,  1005,  1049,   102]]), 
 'token_type_ids': tensor([[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]), 
 'attention_mask': tensor([[1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1]])}
```

## Adding special tokens

Special tokens mark sentence boundaries such as `[CLS]` and `[SEP]`. These tokens are inserted automatically when required by the model.

```py
encoded_input = tokenizer("How are you?")
print(encoded_input["input_ids"])
tokenizer.decode(encoded_input["input_ids"])
```

```python
[101, 1731, 1132, 1128, 136, 102]
'[CLS] How are you? [SEP]'
```

Not all models use special tokens. Usage depends on pretraining configuration.

## Why is all of this necessary?

Tokenized sequences are integer lists. Models require rectangular tensors for batch processing.

```py
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]

encoded_sequences = [
    [
        101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102
    ],
    [101, 1045, 5223, 2023, 2061, 2172, 999, 102],
]
```

Tensor conversion requires equal length structure.

```py
import torch

model_inputs = torch.tensor(encoded_sequences)
```

## Using the tensors as inputs to the model

Model inference accepts token IDs as input.

```py
output = model(model_inputs)
```

Additional arguments exist but are optional for basic inference.

# Tokenizers

Tokenizers are a core component of the NLP pipeline. Their role is to translate raw text into numbers so that models can process it, since models only work with numerical inputs.

In NLP, we typically start with raw text like:

```text
Jim Henson was a puppeteer
```

The tokenizer’s job is to convert this into a structured numerical representation. The key challenge is finding a representation that is both meaningful and efficient, ideally with a small vocabulary and minimal information loss.

## Word-based

Word-based tokenization splits text into whole words. A simple approach is whitespace splitting:

```py
tokenized_text = "Jim Henson was a puppeteer".split()
print(tokenized_text)
```

```python out
['Jim', 'Henson', 'was', 'a', 'puppeteer']
```

Each word is assigned an ID in a vocabulary.

However, this approach has limitations:

* Vocabulary size becomes very large (potentially hundreds of thousands of words).
* Morphologically similar words are treated as unrelated (e.g. “dog” vs “dogs”, “run” vs “running”).
* Unknown words are mapped to an unknown token like `[UNK]` or `<unk>`.

> 💡 A high number of unknown tokens usually indicates information loss and poor vocabulary coverage.

To reduce these issues, we move to smaller units of text.

## Character-based

Character-based tokenization splits text into individual characters.

This has two main advantages:

* Much smaller vocabulary size
* Almost no unknown tokens, since all words can be constructed from characters

However:

* Sequences become much longer (a single word may become many tokens)
* Individual characters carry limited semantic meaning in most languages

This trade-off makes character tokenization less intuitive for semantic understanding, though it can be more suitable in some languages where characters are more meaningful.

To balance word-level meaning and character-level flexibility, we use subword methods.

## Subword tokenization

Subword tokenization breaks words into meaningful parts (subwords), balancing vocabulary size and expressiveness.

Frequent words remain intact, while rare words are split. For example:

* “annoyingly” -> “annoying” + “ly”
* “tokenization” -> “token” + “ization”

Example:

```text
Let's do tokenization!
```

This might be split into subwords such as “token” and “ization”, preserving meaning while keeping vocabulary compact.

> 💡 Subword tokenization is especially effective for languages with rich word formation rules (e.g. Turkish).

There are several subword-based approaches used in practice:

* Byte-level BPE (GPT-2 style)
* WordPiece (BERT style)
* SentencePiece / Unigram (multilingual models)

These methods differ in how they construct and optimize vocabularies, but all follow the same general idea of subword decomposition.

## Loading and saving

Tokenizers can be loaded and saved using `from_pretrained()` and `save_pretrained()`.

Example loading a BERT tokenizer:

```py
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
```

Or using the generic auto class:

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
```

Using the tokenizer:

```py
tokenizer("Using a Transformer network is simple")
```

```python out
{'input_ids': [101, 7993, 170, 11303, 1200, 2443, 1110, 3014, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

Saving:

```py
tokenizer.save_pretrained("directory_on_my_computer")
```

The tokenizer stores both:

* The tokenization rules (algorithm structure)
* The vocabulary (mapping tokens to IDs)

## Encoding

Encoding is the process of converting text into numerical input for models. It happens in two steps:

1. Tokenization (splitting text into tokens)
2. Converting tokens into input IDs

The tokenizer uses a predefined vocabulary and rules from the pretrained model to ensure consistency.

### Tokenization

Tokenization breaks text into subword units:

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)
```

```python out
['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']
```

Here, “transformer” is split into:

* “transform”
* “##er”

The `##` indicates a continuation of a word piece.

### From tokens to input IDs

Tokens are converted into numerical IDs using the vocabulary:

```py
ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)
```

```python out
[7993, 170, 11303, 1200, 2443, 1110, 3014]
```

These IDs are what models actually consume as input.

## Decoding

Decoding is the reverse process of encoding: converting IDs back into readable text.

```py
decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)
```

```python out
'Using a Transformer network is simple'
```

Decoding also merges subword tokens back into full words, making the output human-readable.

Tokenizers therefore support three fundamental operations:

* Converting text -> tokens
* Converting tokens -> IDs
* Converting IDs -> text

# Handling multiple sequences

In earlier sections, inference was done on a single short sequence, but real-world NLP workloads quickly raise questions about scaling to multiple inputs, varying lengths, and model limits. These issues are handled through batching, padding, and attention masking in the 🤗 Transformers API.

## Models expect a batch of inputs

Transformer models expect inputs in batched form rather than a single 1D sequence. If you manually convert token IDs into a tensor without an added batch dimension, the model will fail with a dimension error.

```py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.tensor(ids)

model(input_ids)
```

This fails because the tensor is missing a batch dimension. The tokenizer normally handles this automatically and returns a 2D tensor:

```py
tokenized_inputs = tokenizer(sequence, return_tensors="pt")
print(tokenized_inputs["input_ids"])
```

The correct shape includes an extra outer dimension representing the batch size.

To fix the error manually, wrap the sequence in a list:

```py
input_ids = torch.tensor([ids])
output = model(input_ids)
```

Batching simply means sending multiple sequences together. Even a single sequence is treated as a batch of size 1.

```py
batched_ids = [ids, ids]
```

You can convert this into a tensor and pass it through the model, confirming that batching works by duplicating outputs for identical inputs.

## Padding the inputs

Batches require rectangular tensors, meaning all sequences must have the same length. However, natural language sequences vary in length, so padding is required.

```py
batched_ids = [
    [200, 200, 200],
    [200, 200]
]
```

This cannot be directly converted into a tensor. Padding solves this by adding a special token to shorter sequences:

```py
padding_id = 100

batched_ids = [
    [200, 200, 200],
    [200, 200, padding_id],
]
```

The padding token is defined by the tokenizer:

```py
tokenizer.pad_token_id
```

When comparing individual inference vs batched inference without masking, the outputs differ because padding tokens are still attended to by the model.

```py
sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]

batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids)).logits)
```

The mismatch happens because attention mechanisms treat padding tokens as valid context unless explicitly told otherwise.

## Attention masks

Attention masks explicitly tell the model which tokens are real and which are padding. They have the same shape as the input IDs tensor, with 1 for valid tokens and 0 for padding.

```py
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(
    torch.tensor(batched_ids),
    attention_mask=torch.tensor(attention_mask)
)

print(outputs.logits)
```

With the attention mask applied, the model produces consistent outputs for each sequence regardless of batching.

> 💡 Attention masks are essential whenever padding is used, because they prevent padded tokens from influencing the model’s internal representations.

## Longer sequences

Transformer models have a fixed maximum sequence length, commonly 512 or 1024 tokens depending on the architecture. Inputs exceeding this limit will cause errors or truncation issues.

Two main strategies exist:

* Use models designed for long contexts, such as Longformer or LED.
* Truncate input sequences manually before tokenization or model input.

```py
sequence = sequence[:max_sequence_length]
```

Truncation ensures compatibility but may discard information, so it should be applied carefully depending on the task.

# Putting it all together

The Transformers tokenizer provides a high-level API that handles the full preprocessing pipeline automatically, removing the need to manually manage tokenization, input ID conversion, padding, truncation, and attention masks.

When a sentence is passed directly into a tokenizer, it returns model-ready inputs. For DistilBERT, this includes input IDs and attention masks, while other models may include additional required fields depending on architecture.

```py
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
```

The same interface supports multiple use cases without changing the API. A single sequence or multiple sequences can be processed identically.

```py
sequence = "I've been waiting for a HuggingFace course my whole life."
model_inputs = tokenizer(sequence)
```

```py
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

model_inputs = tokenizer(sequences)
```

Padding is handled automatically and can be configured in different ways depending on the desired output length strategy. This includes padding to the longest sequence, to the model’s maximum length, or to a custom maximum length.

```py
model_inputs = tokenizer(sequences, padding="longest")
model_inputs = tokenizer(sequences, padding="max_length")
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)
```

Truncation is also supported to ensure sequences do not exceed model constraints. This can either follow the model’s maximum length or a user-defined limit.

```py
model_inputs = tokenizer(sequences, truncation=True)
model_inputs = tokenizer(sequences, max_length=8, truncation=True)
```

The tokenizer can also return framework-specific tensor formats, enabling direct compatibility with deep learning libraries such as PyTorch or NumPy.

```py
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")
model_inputs = tokenizer(sequences, padding=True, return_tensors="np")
```

## Special tokens

Tokenized input IDs include additional special tokens that are not present in the raw token list. These are automatically inserted by the tokenizer to match the format used during model pretraining.

```py
sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
print(model_inputs["input_ids"])

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
```

```python out
[101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102]
[1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012]
```

Special tokens such as `[CLS]` at the beginning and `[SEP]` at the end are added depending on the model’s training configuration. These tokens are required for consistency with pretraining behaviour and may vary across different architectures.

```py
print(tokenizer.decode(model_inputs["input_ids"]))
print(tokenizer.decode(ids))
```

```python out
"[CLS] i've been waiting for a huggingface course my whole life. [SEP]"
"i've been waiting for a huggingface course my whole life."
```

> 💡 Special tokens are model-dependent. Some models use only beginning tokens, only ending tokens, or different token sets entirely, but the tokenizer handles this automatically.

## Wrapping up: From tokenizer to model

The tokenizer can be combined directly with a pretrained model to produce end-to-end inference inputs. It handles padding, truncation, and tensor conversion in a single step, producing inputs that can be passed straight into the model.

```py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
```

# Optimized Inference Deployment

This section covers production deployment frameworks for large language models, focusing on Text Generation Inference (TGI), vLLM, and llama.cpp. These tools are designed to improve inference efficiency, scalability, and ease of deployment in production environments.

## Framework Selection Guide

TGI, vLLM, and llama.cpp all serve LLM serving use cases but differ in performance design, memory handling, and deployment focus.

### Memory Management and Performance

**TGI** is designed for stable production workloads with predictable memory usage through fixed sequence lengths. It uses Flash Attention 2 and continuous batching to maximise GPU utilisation by keeping workloads continuously fed. It can also offload parts of models between CPU and GPU when required.

> 💡 Flash Attention improves efficiency by reducing memory transfers between high-bandwidth memory and SRAM, allowing attention computation to occur with fewer bottlenecks and better GPU utilisation.

**vLLM** uses PagedAttention, which manages KV cache memory similarly to virtual memory paging in operating systems. Instead of storing cache contiguously, it splits it into pages that can be flexibly allocated and shared across sequences. This reduces fragmentation and improves throughput significantly for concurrent requests.

> 💡 PagedAttention improves KV cache efficiency by enabling non-contiguous memory allocation and shared page usage across sequences, improving throughput in multi-request scenarios.

**llama.cpp** is a lightweight C/C++ inference engine optimised for CPU-first environments with optional GPU acceleration. It focuses heavily on quantisation to reduce memory usage and allow large models to run on consumer hardware.

> 💡 Quantisation reduces model precision (e.g. FP16 → INT8 or lower), significantly decreasing memory usage while maintaining acceptable output quality.

### Deployment and Integration

**TGI** targets enterprise deployments with built-in Kubernetes support, monitoring (Prometheus/Grafana), scaling, logging, and safety features such as rate limiting and filtering.

**vLLM** is developer-focused, offering high-performance inference with Python-native APIs and OpenAI-compatible endpoints. It integrates well with distributed systems such as Ray.

**llama.cpp** prioritises portability and simplicity, running on minimal dependencies across CPUs, laptops, and edge devices. It exposes an OpenAI-compatible API while maintaining a small footprint.

## Getting Started

### Installation and Basic Setup

#### TGI

TGI is typically deployed via Docker with Hugging Face model support.

```sh
docker run --gpus all \
    --shm-size 1g \
    -p 8080:80 \
    -v ~/.cache/huggingface:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id HuggingFaceTB/SmolLM2-360M-Instruct
```

Python interaction:

```python
from huggingface_hub import InferenceClient

client = InferenceClient(model="http://localhost:8080")

response = client.text_generation(
    "Tell me a story",
    max_new_tokens=100,
    temperature=0.7,
)

print(response.generated_text)
```

OpenAI-compatible usage:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    messages=[
        {"role": "user", "content": "Tell me a story"},
    ],
)

print(response.choices[0].message.content)
```

#### llama.cpp

llama.cpp is built from source and uses GGUF quantised models.

```sh
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

curl -L -O https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF/resolve/main/smollm2-1.7b-instruct.Q4_K_M.gguf
```

Run server:

```sh
./server \
    -m smollm2-1.7b-instruct.Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -c 4096 \
    --n-gpu-layers 0
```

Client usage:

```python
from huggingface_hub import InferenceClient

client = InferenceClient(model="http://localhost:8080/v1", token="sk-no-key-required")

response = client.chat_completion(
    messages=[{"role": "user", "content": "Tell me a story"}],
    max_tokens=100,
)

print(response.choices[0].message.content)
```

OpenAI-compatible usage:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-no-key-required")
```

#### vLLM

vLLM provides high-throughput serving with OpenAI-compatible APIs.

```sh
python -m vllm.entrypoints.openai.api_server \
    --model HuggingFaceTB/SmolLM2-360M-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

Python client:

```python
from huggingface_hub import InferenceClient

client = InferenceClient(model="http://localhost:8000/v1")

response = client.chat_completion(
    messages=[{"role": "user", "content": "Tell me a story"}],
    max_tokens=100,
)

print(response.choices[0].message.content)
```

OpenAI-compatible usage:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
```

## Basic Text Generation

### TGI

TGI supports both raw and chat-based generation with tuning parameters such as temperature, top-p, and repetition penalties.

```python
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a story"},
    ],
    temperature=0.8,
    max_tokens=200,
)
```

### llama.cpp

llama.cpp allows fine-grained control over CPU/GPU execution, context size, and batching.

```sh
./server \
    -m model.gguf \
    -c 4096 \
    --threads 8 \
    --batch-size 512 \
    --n-gpu-layers 0
```

Direct model usage:

```python
from llama_cpp import Llama

llm = Llama(
    model_path="model.gguf",
    n_ctx=4096,
    n_threads=8,
)

output = llm(
    "Write a story",
    max_tokens=200,
    temperature=0.8,
)

print(output["choices"][0]["text"])
```

### vLLM

vLLM supports both API-based and native Python inference with batching and GPU optimisation.

```python
from vllm import LLM, SamplingParams

llm = LLM(model="HuggingFaceTB/SmolLM2-360M-Instruct")

params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100,
)

outputs = llm.generate("Write a story", params)
print(outputs[0].outputs[0].text)
```

## Advanced Generation Control

### Sampling and Token Selection

Generation quality is controlled using temperature, top-p, top-k, and penalties for repetition.

* Temperature controls randomness
* Top-p filters cumulative probability mass
* Top-k restricts token candidates

### Controlling Repetition

Repetition is reduced using frequency and presence penalties or repetition penalty terms depending on framework.

### Length Control and Stop Sequences

Output length is controlled using max tokens, min tokens, and stop sequences to terminate generation early.

## Memory Management

Efficient inference depends heavily on KV cache handling and GPU memory optimisation.

**TGI** uses Flash Attention 2 and continuous batching for efficient GPU utilisation.

**llama.cpp** relies on quantisation and CPU/GPU layer offloading to reduce memory requirements and support consumer hardware.

**vLLM** uses PagedAttention to manage KV cache memory as pageable blocks, enabling efficient multi-request serving and reduced fragmentation.

## Resources

- [Text Generation Inference Documentation](https://huggingface.co/docs/text-generation-inference)
- [TGI GitHub Repository](https://github.com/huggingface/text-generation-inference)
- [vLLM Documentation](https://vllm.readthedocs.io/)
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [llama.cpp GitHub Repository](https://github.com/ggerganov/llama.cpp)
- [llama-cpp-python Repository](https://github.com/abetlen/llama-cpp-python)