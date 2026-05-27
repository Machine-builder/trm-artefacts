# Hugging Face LLM Course - Chapter 2. Fine-Tuning Pretrained Models

[View the original course](https://huggingface.co/learn/llm-course/chapter3/1)

# Processing the data

The workflow for training a sequence classifier begins with a minimal example of fine-tuning a pretrained transformer on a small batch of text. A tokenizer converts raw sentences into model-ready tensors, and labels are added to compute the loss during training.

```python
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]

batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
batch["labels"] = torch.tensor([1, 1])

optimizer = AdamW(model.parameters())

loss = model(**batch).loss
loss.backward()
optimizer.step()
```

Training on only a couple of sentences is insufficient for meaningful performance, so a proper dataset is required. The MRPC (Microsoft Research Paraphrase Corpus) dataset is used as an example. It contains sentence pairs labeled as paraphrases or not, making it suitable for binary text classification.

## Loading a dataset from the Hub

The Hugging Face Hub provides datasets alongside models, accessible through the datasets library. The MRPC dataset is part of the GLUE benchmark, which contains multiple NLP classification tasks.

```py
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_datasets
```

This returns a `DatasetDict` containing train, validation, and test splits. Each split contains sentence pairs, labels, and indices. The dataset is cached locally after download, typically under `~/.cache/huggingface/datasets`, unless configured otherwise via `HF_HOME`.

Individual samples can be accessed like dictionary entries.

```py
raw_train_dataset = raw_datasets["train"]
raw_train_dataset[0]
```

Each example contains two sentences and a label indicating whether they are equivalent. Labels are already integer-encoded, and their meaning can be inspected via dataset features.

```py
raw_train_dataset.features
```

The `label` field is a `ClassLabel`, where `0` corresponds to `not_equivalent` and `1` corresponds to `equivalent`.

## Preprocessing a dataset

Tokenization converts raw text into numerical inputs for the model. A tokenizer can process individual sentences or pairs of sentences. When given sentence pairs, it formats them correctly for BERT-style models.

```py
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

inputs = tokenizer("This is the first sentence.", "This is the second one.")
inputs
```

The output includes `input_ids`, `attention_mask`, and `token_type_ids`. The `token_type_ids` indicate which sentence each token belongs to in a pair input. The final structure follows:

```
[CLS] sentence1 [SEP] sentence2 [SEP]
```

Token type IDs assign 0 to the first sentence segment and 1 to the second. Not all models provide `token_type_ids`, as it depends on whether the model was pretrained with that mechanism.

For dataset preprocessing, tokenization is applied to all sentence pairs.

```py
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)
```

However, this approach loads everything into memory at once. A more scalable method uses the Hugging Face Datasets `map()` function.

```py
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
```

Using `batched=True` significantly improves speed because the tokenizer processes multiple examples at once. Padding is intentionally excluded at this stage to avoid inefficient uniform-length padding across the dataset.

The dataset is updated with new columns such as `input_ids`, `attention_mask`, and `token_type_ids`.

## Dynamic padding

Padding is deferred to batch creation time using a collate function. This ensures that sequences are only padded to the longest item within each batch rather than the maximum dataset length.

```py
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

A batch of samples shows variable sequence lengths before collation, but after applying the data collator, all sequences in the batch are padded to the same length dynamically.

This produces tensors such as:

```
input_ids:      [batch_size, max_length_in_batch]
attention_mask: [batch_size, max_length_in_batch]
token_type_ids: [batch_size, max_length_in_batch]
labels:         [batch_size]
```

Dynamic padding improves efficiency by reducing unnecessary computation on padding tokens, especially when sequence lengths vary widely.

## Section Quiz (concept summary)

Using `Dataset.map()` with `batched=True` improves preprocessing speed by processing multiple samples simultaneously. Dynamic padding reduces computation by padding only to the longest sequence in each batch rather than the entire dataset. `token_type_ids` identify which sentence each token belongs to in sentence-pair inputs. In GLUE loading, the second argument specifies the specific task subset. Removing raw text columns is necessary because models only accept numerical tensor inputs.

## Key takeaway

Tokenization and dataset preprocessing form the bridge between raw text and model training. Efficient pipelines rely on batched processing, dataset mapping, and dynamic padding to balance performance and memory usage.

## Resources

* [Tokenizer Summary](https://huggingface.co/docs/transformers/main/en/tokenizer_summary)
* [Transformers Performance and Scalability](https://huggingface.co/docs/transformers/main/en/performance)

# Fine-tuning a model with the Trainer API

The `Trainer` class in Hugging Face Transformers provides a high-level API for fine-tuning pretrained models on custom datasets. After preprocessing text into tokenized datasets, training mainly involves configuring arguments, defining a model, and passing everything into the `Trainer`.

```py
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

## Training

Training begins by defining `TrainingArguments`, which controls hyperparameters and output configuration. The only required parameter is the output directory for checkpoints and the final model.

```py
from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")
```

The model is typically loaded using a sequence classification head. In this case, a pretrained BERT model is adapted for a binary classification task, which replaces its original head with a randomly initialised classification layer.

```py
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

A `Trainer` object is then created by combining the model, training configuration, datasets, tokenizer, and data collator.

```py
from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
)
```

The training process is started with:

```py
trainer.train()
```

Without additional configuration, the trainer only reports training loss and does not perform evaluation. Evaluation must be explicitly enabled using evaluation settings and metric functions.

## Evaluation

Evaluation requires a function that converts model outputs into meaningful metrics. The `Trainer.predict()` method returns logits, labels, and basic runtime metrics.

```py
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)
```

Logits must be converted into class predictions using argmax over the output dimension.

```py
import numpy as np

preds = np.argmax(predictions.predictions, axis=-1)
```

Metrics are computed using the Hugging Face Evaluate library, which provides standard evaluation functions for datasets such as GLUE.

```py
import evaluate

metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)
```

A reusable evaluation function integrates this process into training:

```py
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```

To enable evaluation during training, a new trainer is defined with evaluation strategy enabled.

```py
training_args = TrainingArguments("test-trainer", eval_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)
```

Evaluation then runs automatically at the end of each epoch, reporting both loss and computed metrics such as accuracy and F1 score.

## Advanced Training Features

The Trainer supports multiple optimisations for efficiency and scalability.

Mixed precision training reduces memory usage and speeds up computation using 16-bit floating point operations.

```py
training_args = TrainingArguments(
    "test-trainer",
    eval_strategy="epoch",
    fp16=True,
)
```

Gradient accumulation allows simulation of larger batch sizes by accumulating gradients over multiple steps before updating weights.

```py
training_args = TrainingArguments(
    "test-trainer",
    eval_strategy="epoch",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
)
```

Learning rate scheduling can be adjusted to improve convergence behaviour.

```py
training_args = TrainingArguments(
    "test-trainer",
    eval_strategy="epoch",
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
)
```

The Trainer also supports distributed training across multiple GPUs or TPUs and integrates common best practices for large-scale model training.

## Section Quiz Summary

The `processing_class` defines which tokenizer is used for preprocessing inputs in the Trainer pipeline. Evaluation frequency is controlled by `eval_strategy`, not separate frequency parameters. Setting `fp16=True` enables mixed precision training for improved speed and memory efficiency. The `compute_metrics` function converts model outputs into evaluation metrics such as accuracy and F1 score. If no evaluation dataset is provided, training still runs but no evaluation metrics are produced. Gradient accumulation increases effective batch size by accumulating gradients over multiple steps before updating model weights.

## Key Concepts

The Trainer API abstracts most of the complexity of model training, including batching, optimisation, and evaluation. `TrainingArguments` centralises all configuration. Metrics are only computed if explicitly defined through a `compute_metrics` function. Performance can be improved using mixed precision and gradient accumulation.

# A full training loop

This section replicates Hugging Face `Trainer` functionality using a fully manual PyTorch training loop.

A dataset is loaded from GLUE (MRPC), tokenized using a BERT tokenizer, and converted into model-ready tensors.

```py
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

## Prepare for training

Dataset post-processing is required before using PyTorch dataloaders. This includes removing unused columns, renaming labels, and converting to tensor format.

```py
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names
```

The dataset is reduced to model-compatible fields such as input IDs, attention masks, token type IDs, and labels.

```python
["attention_mask", "input_ids", "labels", "token_type_ids"]
```

Dataloaders are then created for training and evaluation with batching and dynamic padding.

```py
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)

eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)
```

A single batch is inspected to verify shapes and preprocessing correctness.

```py
for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}
```

```python
{'attention_mask': torch.Size([8, 65]),
 'input_ids': torch.Size([8, 65]),
 'labels': torch.Size([8]),
 'token_type_ids': torch.Size([8, 65])}
```

## Model setup

A pretrained BERT model is loaded for sequence classification.

```py
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

Forward pass confirms that loss and logits are produced when labels are provided.

```py
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)
```

```python
tensor(0.5441, grad_fn=<NllLossBackward>) torch.Size([8, 2])
```

Loss is automatically computed by the model when labels are included, and logits represent class predictions per sample.

## Optimizer and scheduler

AdamW optimizer is used, combining Adam with decoupled weight decay regularization.

```py
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)
```

> 💡 Weight decay improves generalization by preventing overfitting through regularization applied separately from gradient updates.

A linear learning rate scheduler is configured to decay learning rate from its initial value to zero across training steps.

```py
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

print(num_training_steps)
```

```python
1377
```

## Device setup

The model is moved to GPU if available, otherwise CPU is used.

```py
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
device
```

```python
device(type='cuda')
```

## Training loop

The training loop iterates over epochs and batches, performing forward pass, loss computation, backpropagation, optimizer step, scheduler step, and gradient reset.

```py
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

> 💡 Common improvements include gradient clipping, mixed precision training, gradient accumulation, and periodic checkpointing to improve stability and efficiency.

## Evaluation loop

Evaluation uses the Hugging Face evaluation library to accumulate predictions over batches and compute final metrics.

```py
import evaluate

metric = evaluate.load("glue", "mrpc")
model.eval()

for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
```

```python
{'accuracy': 0.8431372549019608, 'f1': 0.8907849829351535}
```

Evaluation uses `model.eval()` to switch behavior of layers like dropout and batch norm, and `torch.no_grad()` to disable gradient tracking for efficiency.

## Accelerate-based distributed training

The Accelerate library enables multi-GPU/TPU training with minimal changes to standard PyTorch loops.

Core changes:

* Wrap model, dataloaders, and optimizer using `accelerator.prepare()`
* Replace `loss.backward()` with `accelerator.backward(loss)`
* Remove manual device handling

```py
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_scheduler

accelerator = Accelerator()

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)

train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dl)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

Accelerate abstracts device placement and distributed synchronization, enabling scalable training across hardware setups.

## Evaluation concepts

Evaluation relies on:

* `model.eval()` to disable training-specific behaviors
* `torch.no_grad()` to reduce memory and computation overhead
* metric accumulation across batches before final computation

## Training best practices

Modern training improvements include:

* Gradient clipping for stability
* Mixed precision training for speed and memory efficiency
* Gradient accumulation for larger effective batch sizes
* Checkpointing for fault tolerance
* Hyperparameter tuning for optimal performance

# Understanding Learning Curves

Learning curves represent model performance over time during training and are essential for diagnosing training behaviour and identifying issues early. They are commonly used when fine-tuning models using APIs such as `Trainer` or custom training loops.

## What are Learning Curves

Learning curves track model performance metrics across training steps or epochs.

The two primary metrics are:

* Loss curves: measure prediction error over time
* Accuracy curves: measure proportion of correct predictions over time

These metrics are computed per batch during training and logged for later visualisation using tools such as Weights & Biases.

## Loss Curves

Loss curves show how model error changes during training. A typical successful pattern shows a high initial loss followed by steady decrease and eventual stabilisation.

Key behaviour:

* Initial loss is high due to untrained weights
* Loss decreases as optimisation improves predictions
* Curve eventually converges and stabilises at a low value

Example training setup using Hugging Face Trainer with logging:

```python
from transformers import Trainer, TrainingArguments
import wandb

wandb.init(project="transformer-fine-tuning", name="bert-mrpc-analysis")

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=50,
    save_steps=100,
    logging_steps=10,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
```

## Accuracy Curves

Accuracy curves show classification performance over time and generally increase during training.

Key behaviour:

* Starts low due to untrained model
* Gradually increases as learning improves
* Often shows plateaus rather than smooth growth

> 💡 Accuracy behaves differently from loss because it depends on discrete predictions. Small confidence changes may not change the final class prediction, so accuracy only increases when a decision boundary is crossed.

## Convergence

Convergence occurs when both loss and accuracy stabilise. This indicates the model has likely learned the underlying patterns in the dataset.

At convergence:

* Loss levels off
* Accuracy stabilises
* Further training yields minimal improvements

## Interpreting Learning Curve Patterns

Learning curve shapes provide insight into model behaviour and training health.

## Healthy Learning Curves

A well-trained model typically shows:

* Smooth decrease in training and validation loss
* Increasing accuracy over time
* Small gap between training and validation metrics
* Stable convergence without instability

Accuracy curves may still show step-like behaviour due to discrete prediction thresholds.

Example intuition:

If a binary classifier outputs probabilities for class 1, small improvements in probability may reduce loss but not change accuracy until a threshold (e.g., 0.5) is crossed.

## Monitoring During Training

Key things to observe during training:

* Whether loss continues to decrease or plateaus
* Whether validation loss begins increasing (overfitting signal)
* Whether curves are stable or noisy (learning rate issues)
* Whether sudden spikes indicate instability

## Evaluation After Training

After training completes, learning curves help assess:

* Final model performance
* Whether training could have stopped earlier
* Generalisation gap between training and validation
* Whether further training would help

## Overfitting

Overfitting occurs when a model learns training data too well and fails to generalise.

Symptoms:

* Training loss decreases while validation loss increases
* Large gap between training and validation accuracy
* High training performance but poor validation performance

Solutions:

* Regularisation (dropout, weight decay)
* Early stopping
* Data augmentation
* Reducing model complexity

Early stopping example:

```python
from transformers import EarlyStoppingCallback

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    num_train_epochs=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
```

## Underfitting

Underfitting occurs when the model is too simple to learn patterns in the data.

Causes:

* Model lacks capacity
* Insufficient training time
* Poor learning rate choice
* Limited or poor-quality data

Symptoms:

* High training and validation loss
* Early plateau in performance
* Low training accuracy

Solution example:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
)
```

## Erratic Learning Curves

Erratic curves indicate unstable or ineffective training.

Causes:

* Learning rate too high
* Batch size too small
* Poor data preprocessing
* Gradient instability

Symptoms:

* Fluctuating loss and accuracy
* No clear downward or upward trend
* High variance in metrics

Solutions:

* Reduce learning rate
* Increase batch size
* Use gradient clipping
* Improve data preprocessing

Example adjustment:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-4,
    per_device_train_batch_size=32,
)
```

## Key Takeaways

Learning curves provide direct insight into model training behaviour.

Main points:

* Loss and accuracy curves reveal different aspects of learning
* Overfitting shows divergence between training and validation performance
* Underfitting shows poor performance across both training and validation
* Erratic curves often indicate optimisation issues
* Early stopping helps prevent overfitting
* Proper hyperparameter tuning stabilises training

## Resources

Weights & Biases: [https://wandb.ai/](https://wandb.ai/)
Weights & Biases Documentation: [https://docs.wandb.ai/](https://docs.wandb.ai/)