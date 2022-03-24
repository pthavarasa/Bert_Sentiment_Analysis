from fastapi import FastAPI
import os
#from bert_sentiment_predict import Bert
from fastapi.middleware.cors import CORSMiddleware
from DB import Database
from utils import clean_input_py
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


tbName = "sentiment"
dbName = "PROJET"
columns = "(sentiment VARCHAR(255), review VARCHAR(255))"
db = Database.DataBase("localhost","root","",dbName, tbName)
db.createTable(tbName, columns)
#db.injectDataset(tbName, columns, dataset)



"""

db = DataBase("PROJET")
conn, dbCursor = db.connexionDB("localhost", "root", "")
db.selectDB(dbCursor)

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