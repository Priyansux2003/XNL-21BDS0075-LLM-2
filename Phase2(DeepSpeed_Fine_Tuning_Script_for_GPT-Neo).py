!pip install datasets
!pip install deepspeed
!pip install -U accelerate
!pip install -U transformers
import pandas as pd
import datasets
from datasets import load_dataset
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import random
import torch
import deepspeed


def collect_datasets():
    """Load sentiment datasets from Hugging Face."""
    imdb = load_dataset("imdb")
    amazon = load_dataset("amazon_polarity")
    yelp = load_dataset("yelp_polarity")
    return imdb, amazon, yelp


def preprocess_and_combine(imdb, amazon, yelp, sample_size=5000):
    """Extract and standardize datasets, then merge into one."""
    df_imdb = pd.DataFrame(imdb['train']).sample(sample_size)[['text', 'label']]
    df_amazon = pd.DataFrame(amazon['train']).sample(sample_size)[['content', 'label']]
    df_yelp = pd.DataFrame(yelp['train']).sample(sample_size)[['text', 'label']]

    df_amazon.rename(columns={'content': 'text'}, inplace=True)
    df_combined = pd.concat([df_imdb, df_amazon, df_yelp], ignore_index=True)
    return df_combined


def augment_data(df, augmentation_factor=0.3):
    """Augment dataset by paraphrasing some sentences."""
    n_samples = int(len(df) * augmentation_factor)
    sample_texts = df.sample(n_samples)['text'].tolist()
    paraphraser = pipeline("text2text-generation", model="t5-small")

    augmented_texts = [paraphraser(text, max_length=100, do_sample=True)[0]['generated_text']
                      for text in sample_texts]
    augmented_labels = df.sample(n_samples)['label'].tolist()

    df_aug = pd.DataFrame({'text': augmented_texts, 'label': augmented_labels})
    df_final = pd.concat([df, df_aug], ignore_index=True)
    return df_final


if __name__ == "__main__":
    imdb, amazon, yelp = collect_datasets()
    df_combined = preprocess_and_combine(imdb, amazon, yelp)
    df_final = augment_data(df_combined)
    df_final.to_csv("sentiment_dataset.csv", index=False)
    print("Dataset saved as sentiment_dataset.csv")

# --- Training ---
dataset = load_dataset("csv", data_files="sentiment_dataset.csv")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token


def tokenize_function(examples):
    # Ensure examples["text"] is a list of strings
    texts = examples["text"]
    if isinstance(texts, str):
        texts = [texts]
    # Handle missing or empty text fields, including None
    texts = [text if text is not None else "" for text in texts]  # Replace None and empty text with ""
    return tokenizer(texts, padding="max_length", truncation=True, max_length=512)


dataset = dataset.map(tokenize_function, batched=False)
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,  # Mixed precision training
    deepspeed="ds_config.json",
)

# Configure DeepSpeed
ds_config = {
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 8,
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"}
    },
    "fp16": {"enabled": True}
}

with open("ds_config.json", "w") as f:
    import json
    json.dump(ds_config, f, indent=4)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]  # If you have a separate test set
)

# Train model
trainer.train()

# Save model
trainer.save_model("./fine_tuned_gpt_neo")
