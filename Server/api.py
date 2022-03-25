from fastapi import FastAPI
import os
#from bert_sentiment_predict import Bert
from fastapi.middleware.cors import CORSMiddleware
from DB import DataBase
from utils import format_dataset
#from utils import clean_input
"""
from Model import LR
from Model import Bert
"""
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
originalDatasetPath = "./Server/DB/dataset.txt"
formatDatasetPath = "./Server/DB/formatDataset.txt"
format_dataset.format_dataset(originalDatasetPath)
tbName = "sentiment"
dbName = "PROJET"
columns = "(sentiment VARCHAR(255), review VARCHAR(255))"
db = DataBase.Database("localhost", "root", "", dbName, tbName)
db.createTable(tbName, columns)
db.injectFile(formatDatasetPath, tbName)


"""

sentimentTable = db.getAll("data")

bert = Bert(sentimentTable)
linearRegression = LR(sentimentTable)

@app.get("/text/{text}/{emotion}")
async def text(text, emotion):
    text = clean_input(text)
    db.insertElem(conn, dbCursor, "data", "(name, sentiment, review)", (text, emotion))
    bertPrediction = bert.predict(text)
    lrPrediction = LR.predict(text)
    return {
        emotion: emotion,
        bert: bertPrediction
    }


"""