from fastapi import FastAPI
import os
from bert_sentiment_predict import Bert
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bert = Bert()

@app.get("/text/{text}")
async def text(text):
	# format du return : {emotion: value, precision: value}
	return bert.predict(text)
