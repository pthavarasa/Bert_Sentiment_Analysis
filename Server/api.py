from fastapi import FastAPI
import os
from bert_sentiment_predict import MyModel

app = FastAPI()
model = MyModel()

@app.get("/text/{text}")
async def text(text):
	MyModel()
	return model.predict(text)
