from fastapi import FastAPI
import os
from bert_sentiment_predict import Bert
from fastapi.middleware.cors import CORSMiddleware
from Database import DataBase
from clean_input_py import clean_input

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
db = DataBase("PROJET")
conn, dbCursor = db.connexionDB("localhost", "root", "")
db.selectDB(dbCursor)

@app.get("/text/{text}?emotion={emotion}")
async def text(text):
    text = clean_input(text)
    db.insertElem(conn, dbCursor, "data", "(name, sentiment, review)", (text, emotion))
    bertPrediction = bert.predict(text)
    return {
        emotion: emotion,
        bert: bertPrediction
    }



