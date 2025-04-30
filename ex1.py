from datasets import load_dataset
import wandb
import os
import torch
from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np

# Load the MRPC dataset
dataset = load_dataset("glue", "mrpc")

# Load the tokenizer and model
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

#tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length", max_length=128
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load the model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define the compute_metrics function
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy.compute(predictions=predictions, references=labels)
    return {"accuracy": acc["accuracy"]}

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()



