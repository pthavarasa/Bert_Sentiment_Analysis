from fastapi import FastAPI
import os
from Model import bert
from Model import logistic_regression
from fastapi.middleware.cors import CORSMiddleware
from DB import DataBase
from utils import format_dataset

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=origins,
    allow_headers=origins,
)

bertModelPicklePath = './Model/bert.pickle'
originalDatasetPath = "./DB/dataset.txt"
formatDatasetPath = "./DB/formatDataset.txt"
format_dataset.format_dataset(originalDatasetPath, formatDatasetPath)
tbName = "sentiment"
dbName = "PROJET"
columns = "(review TEXT, sentiment VARCHAR(255))"
db = DataBase.Database("localhost", "root", "", dbName, tbName)
db.createTable(tbName, columns)
db.injectFile(formatDatasetPath, tbName)
dataset = db.getDfOfDataset(tbName)

bert = bert.Bert(bertModelPicklePath)
linearRegression = logistic_regression.Logistic_regression(dataset)

nInput = 0

@app.get("/text/{text}/{emotion}")
async def text(text, emotion):
    global nInput
    global bert
    global linearRegression
    global dataset
    global db
    if (nInput == 1000):
        nInput = 0
        dataset = db.getDfOfDataset(tbName)
        linearRegression = logistic_regression.Logistic_regression(dataset)
        await bert.train_model(bertModelPicklePath, dataset)
    #text = clean_input(text)
    db.insertElem(tbName, "(review, sentiment)", (text, emotion))
    bertPrediction = bert.predict(text)
    lrPrediction = "Positif" if linearRegression.predict(text) == 1 else "Negatif"
    nInput += 1
    return {
        "emotion": "Positif" if int(emotion) == 1 else "Negatif",
        "text": text,
        "bert": bertPrediction["sentiment"],
        "lr": lrPrediction
    }
