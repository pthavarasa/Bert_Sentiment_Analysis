from fastapi import FastAPI
import os
from Model import bert
from Model import logistic_regression
from fastapi.middleware.cors import CORSMiddleware
from DB import DataBase
from utils import format_dataset
#from utils import clean_input

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
#format_dataset.format_dataset(originalDatasetPath)
tbName = "sentiment"
dbName = "PROJET"
columns = "(sentiment VARCHAR(255), review VARCHAR(255))"
db = DataBase.Database("localhost", "root", "", dbName, tbName)
db.createTable(tbName, columns)
db.injectFile(formatDatasetPath, tbName)
dataset = db.getDfOfDataset(tbName)

#bert = bert.Bert(dataset)
linearRegression = logistic_regression.Logistic_regression(dataset)

nInput = 0

"""

@app.get("/text/{text}/{emotion}")
async def text(text, emotion):
    if (nInput == 1000):
        dataset = db.getDfOfDataset(tbName)
        linearRegression = logistic_regression.Logistic_regression(dataset)
    text = clean_input(text)
    db.insertElem(conn, dbCursor, "data", "(name, sentiment, review)", (text, emotion))
    bertPrediction = bert.predict(text)
    lrPrediction = linearRegression.predict(text)
    nInput += 1
    return {
        emotion: emotion,
        bert: bertPrediction,
        lr: lrPrediction
    }

"""