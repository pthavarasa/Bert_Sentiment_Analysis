from fastapi import FastAPI
import os
from bert_sentiment_predict import MyModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bert = Bert()

@app.get("/text/{text}?emotion={bool}")
async def text(text):
	# format du return : {emotion: value, precision: value}
	return bert.predict(text)
