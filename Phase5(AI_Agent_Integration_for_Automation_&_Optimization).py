import optuna
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import wandb
import psutil
import time

# Initialize Weights & Biases (wandb) for monitoring
wandb.init(project="llm-hyperparameter-tuning")

# Load dataset
dataset = load_dataset("csv", data_files="sentiment_dataset.csv")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

dataset = dataset.map(tokenize_function, batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

def monitor_resources():
    """Monitor system resources and log them to wandb."""
    while True:
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        wandb.log({"CPU Usage (%)": cpu_usage, "Memory Usage (%)": memory_usage})
        time.sleep(10)

# Start resource monitoring in a separate thread
import threading
monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
monitor_thread.start()

def objective(trial):
    """Define objective function for Optuna hyperparameter tuning."""
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 5e-4)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 0.1)
    gradient_accumulation_steps = trial.suggest_categorical("grad_acc", [4, 8, 16])
    num_train_epochs = trial.suggest_int("num_train_epochs", 3, 5)
    
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        fp16=True,
        deepspeed="ds_config.json",
        logging_dir="./logs"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"]
    )
    
    trainer.train()
    eval_result = trainer.evaluate()
    
    # Log evaluation loss and resource usage to wandb
    wandb.log({"eval_loss": eval_result["eval_loss"]})
    
    return eval_result["eval_loss"]

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)  # Increased trials for better optimization

# Save best hyperparameters
best_params = study.best_params
np.save("best_hyperparameters.npy", best_params)

print("Best Hyperparameters:", best_params)

# Mark wandb run as complete
wandb.finish()
