from datasets import load_dataset
import wandb
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, set_seed, DataCollatorWithPadding
import evaluate
import numpy as np
import argparse


def get_args():
    # Define the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_train_samples', type=int, default=-1)
    parser.add_argument('--max_eval_samples', type=int, default=-1)
    parser.add_argument('--max_test_samples', type=int, default=-1)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    args = vars(parser.parse_args())
    return args

def tokenize_function(examples):
    return tokenizer(
        examples["sentence1"], examples["sentence2"], truncation=True
        )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy.compute(predictions=predictions, references=labels)
    return {"accuracy": acc["accuracy"]}

#arguments
args = get_args()



# Set the seed for reproducibility
set_seed(42)

# Load the MRPC dataset and select samples according to the values past in args
dataset = load_dataset("glue", "mrpc")
train_set = dataset['train']
train_set = train_set.select (args['max_train_samples']) if args['max_train_samples'] > -1 else train_set
validation_set = dataset['validation']
validation_set = validation_set.select (args['max_eval_samples']) if args['max_eval_samples'] > -1 else validation_set
test_set = dataset['test']
test_set = test_set.select (args['max_test_samples']) if args['max_test_samples'] > -1 else test_set

# Load the tokenizer and model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Define the compute_metrics function
accuracy = evaluate.load("accuracy")

if args['do_train']:
    # Initialize wandb
    wandb.init(project="mrpc-paraphrase-detection", config=args)


    tokenized_train_set = train_set.map(tokenize_function, batched=True)
    tokenized_validation_set = validation_set.map(tokenize_function, batched=True)

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=args['lr'],
        per_device_train_batch_size=args['batch_size'],
        per_device_eval_batch_size=args['batch_size'],
        num_train_epochs=args['num_train_epochs'],
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=1,
        report_to="wandb",
        load_best_model_at_end=False,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_set,
        eval_dataset=tokenized_validation_set,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    eval_accuracy = eval_results["eval_accuracy"]
    print("Validation Results:", eval_accuracy)

    # Save the model results to a file
    with open ('res.txt', 'a') as f:
        f.write(f'epochs: {args["num_train_epochs"]}, lr: {args["lr"]}, batch_size: {args["batch_size"]}, eval_acc: {eval_accuracy}\n')

    with open (f'predictions_on_validation_of_epoch_num_{args["num_train_epochs"]}_lr_{args["lr"]}_batch_size_{args["batch_size"]}.txt', 'w') as f:
        for i, pred in enumerate(tokenized_validation_set):
            sentence1 = tokenized_validation_set[i]["sentence1"]
            sentence2 = tokenized_validation_set[i]["sentence2"]
            f.write(f"{sentence1}###{sentence2}###{pred['label']}\n")

    # Save the model
    trainer.save_model(f'epoch_num_{args["num_train_epochs"]}_lr_{args["lr"]}_batch_size_{args["batch_size"]}')
    
    wandb.finish()

# define evaluation arguments
if args['do_predict']:

    # Tokenize the dataset
    tokenized_test_set = test_set.map(tokenize_function, batched=True)

    max_val_acc = 0
    max_test_acc = 0
    best_model_path = None
    best_model_path_of_test_set = None

    # Read the results from the file and find the best model
    with open('res.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            parameters_and_acc = line.split(', ')
            epochs = int(parameters_and_acc[0].split(': ')[1])
            lr = float(parameters_and_acc[1].split(': ')[1])
            batch_size = int(parameters_and_acc[2].split(': ')[1])
            eval_acc = float(parameters_and_acc[3].split(': ')[1])
            model_path = f'epoch_num_{epochs}_lr_{lr}_batch_size_{batch_size}'
            
            # Load the model
            model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
            # Load the tokenizer
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            model.eval()
            trainer = Trainer(model=model, tokenizer=tokenizer)
            predictions = trainer.predict(tokenized_test_set)
            preds = np.argmax(predictions.predictions, axis=-1)
            test_accuracy = accuracy.compute(predictions=preds, references=tokenized_test_set["label"])['accuracy']
            if eval_acc > max_val_acc:
                max_val_acc = eval_acc
                best_model_path = model_path
            if test_accuracy > max_test_acc:
                max_test_acc = test_accuracy
                best_model_path_of_test_set = model_path
    

    if best_model_path == best_model_path_of_test_set:
        print('same')
    else:
        print('different')

    best_model = AutoModelForSequenceClassification.from_pretrained(best_model_path, num_labels=2)
    trainer = Trainer(model=best_model, tokenizer=tokenizer)
    predictions = trainer.predict(tokenized_test_set)
    preds = np.argmax(predictions.predictions, axis=-1)

    with open('predictions.txt', 'w') as f:
        for i, pred in enumerate(preds):
            sentence1 = test_set[i]["sentence1"]
            sentence2 = test_set[i]["sentence2"]
            f.write(f"{sentence1}###{sentence2}###{preds[i]}\n")