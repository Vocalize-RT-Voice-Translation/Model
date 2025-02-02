from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

# Load the IITB Hindi-English dataset (for both en-hi and hi-en)
dataset_en_hi = load_dataset("opus100", "en-hi")
dataset_hi_en = load_dataset("ccmatrix", "hi-en")

model_name_en_hi = "Helsinki-NLP/opus-mt-en-hi"
model_name_hi_en = "Helsinki-NLP/opus-mt-hi-en"

tokenizer_en_hi = AutoTokenizer.from_pretrained(model_name_en_hi)
tokenizer_hi_en = AutoTokenizer.from_pretrained(model_name_hi_en)

model_en_hi = AutoModelForSeq2SeqLM.from_pretrained(model_name_en_hi)
model_hi_en = AutoModelForSeq2SeqLM.from_pretrained(model_name_hi_en)


# Tokenize datasets
def tokenize_function_en_hi(examples):
    return tokenizer_en_hi(
        examples["en"], truncation=True, padding="max_length", max_length=128
    )


def tokenize_function_hi_en(examples):
    return tokenizer_hi_en(
        examples["hi"], truncation=True, padding="max_length", max_length=128
    )


# Apply tokenization to datasets
tokenized_datasets_en_hi = dataset_en_hi.map(tokenize_function_en_hi, batched=True)
tokenized_datasets_hi_en = dataset_hi_en.map(tokenize_function_hi_en, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",  # output directory
    num_train_epochs=3,  # number of training epochs
    per_device_train_batch_size=8,  # batch size for training
    per_device_eval_batch_size=16,  # batch size for evaluation
    logging_dir="./logs",  # directory for storing logs
    logging_steps=10,  # log every 10 steps
    evaluation_strategy="epoch",  # evaluate at the end of each epoch
    save_strategy="epoch",  # save model at the end of each epoch
    load_best_model_at_end=True,  # load the best model when finished
)

# Fine-tune the English-to-Hindi model
trainer_en_hi = Trainer(
    model=model_en_hi,
    args=training_args,
    train_dataset=tokenized_datasets_en_hi["train"],
    eval_dataset=tokenized_datasets_en_hi["test"],
)

# Fine-tune the Hindi-to-English model
trainer_hi_en = Trainer(
    model=model_hi_en,
    args=training_args,
    train_dataset=tokenized_datasets_hi_en["train"],
    eval_dataset=tokenized_datasets_hi_en["test"],
)

# Train the models
trainer_en_hi.train()
trainer_hi_en.train()

# Save the fine-tuned models and tokenizers
model_en_hi.save_pretrained("fine_tuned_en_hi")
model_hi_en.save_pretrained("fine_tuned_hi_en")

tokenizer_en_hi.save_pretrained("fine_tuned_en_hi")
tokenizer_hi_en.save_pretrained("fine_tuned_hi_en")
