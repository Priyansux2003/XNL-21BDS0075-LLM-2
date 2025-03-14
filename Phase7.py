from fastapi import FastAPI, Depends, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn
import os
from pydantic import BaseModel
from typing import Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")  

# Initialize FastAPI app
app = FastAPI()

# Load model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def verify_api_key(api_key: str):
    """API Key Authentication"""
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

class InputData(BaseModel):
    text: str
    api_key: str

@app.post("/predict")
def predict(data: InputData):
    """Generate sentiment analysis response"""
    verify_api_key(data.api_key)
    inputs = tokenizer(data.text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=512)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
