import streamlit as stl
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch as pt

from safetensors.torch import load_file, save_file
import openai
from dotenv import load_dotenv

import json

#with open("config.json") as f:
   # config = json.load(f)


#api_key = config["OPENAI_API_KEY"]
api_key = stl.secrets["OPENAI_API_KEY"]



openai.api_key = api_key


def explain_reason(news, label):
    justification =  openai.chat.completions.create(
        model="gpt-4o-mini", 
        messages = [
    {"role": "system", "content": "You are a part of a fake news detection application. Another model besides you determines whether or not a piece of news is fake or not. It automatically assigns a label to that piece of news, real or fake. Your job is to explain in detail that particular label is assigned to a piece of news. The explanation must to be detailed, but have a simple vocabulary as the target audience of the application is those who do are not literate in news linguo, basically the average person. Furthermore, if it is deemed fake, please recommend alternative articles regarding the news' subject matter that may provide real news regarding the matter. Provide the links to these articles and make sure they actually work."},
    {"role": "user", "content": f"Why is '{news}' {label}"},
    ]
    )
    return justification.choices[0].message.content

stl.title("FauxBuster")






model = AutoModelForSequenceClassification.from_pretrained("bert_fake_news_model", from_tf=True)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.save_pretrained("bert_fake_news_model")



input_handler = stl.text_area("Enter a piece of news: ")

if stl.button("Determine Validity"):
    inputs = tokenizer(input_handler, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = pt.argmax(logits, dim=-1).item()

    label = "Fake" if prediction == 0 else "Real"
    stl.success(f"Prediction: {label}")
    stl.subheader("Explanation:")
    stl.write(explain_reason(input_handler, label.lower()))
