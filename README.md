# LLM Fine-Tuning and Optimization for Sentiment Analysis

## Overview
This project focuses on fine-tuning an open-source Large Language Model (LLM) for sentiment analysis using DeepSpeed. The objective is to enhance performance, accuracy, and resource efficiency through hyperparameter tuning, distributed training, and multi-cloud deployment.

## Features
- Fine-tuning of GPT models with sentiment analysis datasets
- Hyperparameter tuning using Optuna
- Distributed training with DeepSpeed
- AI-driven monitoring and auto-scaling
- Multi-cloud deployment on AWS, GCP, and Azure
- Continuous performance monitoring and evaluation

## Project Structure
```
├── data/                        # Dataset for fine-tuning
├── models/                      # Trained models and checkpoints
├── scripts/                     # Training, evaluation, and deployment scripts
├── configs/                     # DeepSpeed and training configuration files
├── docker/                      # Containerization setup
├── deployment/                  # Kubernetes and cloud deployment files
├── notebooks/                   # Jupyter notebooks for experimentation
└── README.md                    # Project documentation
```

## Setup Instructions
### 1. Clone Repository
```sh
git clone https://github.com/xnl-innovations/XNL-21BDS0075-LLM-2.git
cd XNL-21BDS0075-LLM-2
```

### 2. Install Dependencies
```sh
pip install -r requirements.txt
```

### 3. Configure DeepSpeed
Modify `ds_config.json` to set optimal parameters for distributed training.

### 4. Train the Model
```sh
python scripts/train.py --config configs/train_config.json
```

### 5. Deploy the Model
```sh
kubectl apply -f deployment/k8s_deployment.yaml
```

## Performance Evaluation
- Training logs and metrics are recorded using Weights & Biases (wandb)
- Evaluation is done using accuracy, precision, recall, and F1-score


