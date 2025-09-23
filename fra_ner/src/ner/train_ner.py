import json
import os
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import numpy as np
from seqeval.metrics import classification_report

# --- 1. Configuration ---
ANNOTATIONS_PATH = "../data/annotations/labeled_data.json"
MODEL_CHECKPOINT = "distilbert-base-uncased" # A good, fast baseline model
MODEL_OUTPUT_PATH = "../models/ner"

# --- 2. Load and Preprocess Data ---
def preprocess_label_studio_data(data):
    processed_data = []
    # Label Studio exports one big JSON object, which is a list of tasks
    for task in data:
        # We only care about tasks that were actually annotated
        if not task.get('annotations') or not task['annotations'][0].get('result'):
            continue
            
        text = task['data']['text']
        labels = []
        
        # Collect all labeled spans
        for annotation in task['annotations'][0]['result']:
            span = annotation['value']
            labels.append({
                'start': span['start'],
                'end': span['end'],
                'label': span['labels'][0]
            })
            
        processed_data.append({"text": text, "labels": labels})
    return processed_data

def convert_to_dataset_format(processed_data):
    dataset_formatted_data = {"tokens": [], "ner_tags": []}
    
    # Get all unique label names
    unique_labels = sorted(list(set(label['label'] for item in processed_data for label in item['labels'])))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}

    for item in processed_data:
        text = item['text']
        labels = item['labels']
        
        tokens = text.split()
        tags = ['O'] * len(tokens) # Default tag is 'O' (Outside)
        
        char_pos = 0
        for i, token in enumerate(tokens):
            start = char_pos
            end = start + len(token)
            
            for label in labels:
                if start >= label['start'] and end <= label['end']:
                    # Simple token-based alignment. More complex cases might need smarter logic.
                    tags[i] = label['label']
                    break # Move to the next token once a label is assigned
            
            char_pos = end + 1 # Account for the space

        dataset_formatted_data["tokens"].append(tokens)
        dataset_formatted_data["ner_tags"].append([label2id[tag] for tag in tags])

    return dataset_formatted_data, id2label, label2id


# --- 3. Tokenization and Label Alignment ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def tokenize_and_align_labels(examples, label2id):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100) # For special tokens like [CLS], [SEP]
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100) # Only label the first token of a multi-token word
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# --- 4. Metrics ---
def compute_metrics(p, id2label):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    report = classification_report(true_labels, true_predictions, output_dict=True)
    return {
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
        "accuracy": report["weighted avg"]["support"] # Note: support is used here just to show an example
    }


# --- 5. Main Training Execution ---
if __name__ == "__main__":
    # Load and process the data
    with open(ANNOTATIONS_PATH, 'r', encoding='utf-8') as f:
        ls_data = json.load(f)
    
    processed_data = preprocess_label_studio_data(ls_data)
    dataset_data, id2label, label2id = convert_to_dataset_format(processed_data)

    # Create Hugging Face Dataset
    dataset = Dataset.from_dict(dataset_data)
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True, fn_kwargs={"label2id": label2id})
    
    # Split into train and test sets
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
    dataset_dict = DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
    })

    # Load Model
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )

    # Training Arguments
    args = TrainingArguments(
        output_dir=MODEL_OUTPUT_PATH,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5, # Start with a few epochs
        weight_decay=0.01,
        push_to_hub=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, id2label)
    )

    # Train the model
    print("Starting model training...")
    trainer.train()
    print("Training complete.")

    # Save the final model
    trainer.save_model(MODEL_OUTPUT_PATH)
    print(f"Model saved to {MODEL_OUTPUT_PATH}")