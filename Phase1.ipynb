!pip install datasets
import pandas as pd
import datasets
from datasets import load_dataset
from transformers import pipeline
import random

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
    
    augmented_texts = [paraphraser(text, max_length=100, do_sample=True)[0]['generated_text'] for text in sample_texts]
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
