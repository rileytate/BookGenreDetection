# Riley Tate and Harrison Niswander
# CS 59300 - Natural Language Processing
# 10-28-26
# NLP Research Project - Transformer Model

#--------------------------------------------------------------------------------------------
#                                 Data Preprocessing
#--------------------------------------------------------------------------------------------
# If throws error when trying to load, use pip install --upgrade kagglehub
# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import numpy as np
import evaluate
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding

def fine_tune_model():

    # Set the path to the file you'd like to load
    file_path = "data.csv"

    # Load CSV file
    df = pandas.read_csv(file_path)

    # drop all columns except for summary and genre
    df = df[['summary', 'genre']]

    # drop rows in dataset where value is null in summary or genre column
    df = df.dropna(subset=['summary', 'genre'])

    # Reduce the number of examples
    reducedDataset = df.sample(n=1000, random_state=42).reset_index(drop=True)

    X = df['summary']
    y = df['genre']

    # Split data into training and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    # Re-combine training and validation datasets
    train_dataframe = pandas.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    val_dataframe = pandas.concat([X_val.reset_index(drop=True), y_val.reset_index(drop=True)], axis=1)

    # Convert to huggingface datasets
    train_dataset = Dataset.from_pandas(train_dataframe, preserve_index=False)
    val_dataset = Dataset.from_pandas(val_dataframe, preserve_index=False)

    datasetDictionary = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
    })

    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

    # List the Categories
    categories = ['fantasy', 'science', 'crime', 'history', 'horror', 'thriller', 'psychology', 'romance', 'sports', 'travel']
    id2label = {i: label for i, label in enumerate(categories)}
    label2id = {label: i for i, label in enumerate(categories)}

    # Used the tokenization step found at:  
    # https://medium.com/thenextlayer/building-a-text-classification-model-using-distilbert-703c1409696c 
    def tokenize_data(examples):
        return tokenizer(examples['summary'], truncation=True)

    # Tokinize the dataset
    tokenized_datasets = datasetDictionary.map(tokenize_data, batched=True)

    # Pad the values
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create the model
    model = AutoModelForSequenceClassification.from_pretrained(
            'distilbert/distilbert-base-uncased', 
            num_labels=10, 
            id2label=id2label, 
            label2id=label2id
        )

    # Create training parameters
    training_args = TrainingArguments(
            output_dir="./distilbert-10-categories",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_steps=10,
        )

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    # Create the trainer for the model
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            processing_class=tokenizer,
            data_collator=data_collator
    )

    print('we are about to train the model')
    # Fine-tune the model!!
    trainer.train()

    print('we finished training the model')

    # Save the best models and tokenizers
    trainer.save_model("./fine_tuned_distilbert_model")
    tokenizer.save_pretrained("./fine_tuned_distilbert_tokenizer")
    print("Model saved successfully!")


fine_tune_model()