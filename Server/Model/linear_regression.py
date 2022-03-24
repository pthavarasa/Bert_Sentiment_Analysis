# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-


# importation des bibliotheques sklearn required

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import mysql.connector as connector
import pymysql

"""importation du dataset"""

# importation des donnees a partir de la bdd sur mysql
connexion = pymysql.connect(host='localhost',
                        user='root',
                        password='root')
df = pd.read_sql('Select * From PROJET.dataset', connexion)
connexion.close()   

df['sentiment'] = df.iloc[:,2]      


# attibution d'une etiquette pour les sentiments positifs:1 et negatifs:0
df['sentiment'] = df['sentiment'].map({'positive\r':1,'negative\r':0,'positive':1,'negative':0})

# Suppresssion des commentaires vides
dff = df[df['review'] == ""]
df = df.drop(dff.index)

print(df['sentiment'].sum())


""" Preparation du dataset"""

#separation etiquette / dataset à étudier
X = df['review']  #seul la colonne contenant le commentaire sera retenue car relevant
y = df['sentiment']

#separation du dataset pour la phase training et validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
cv = CountVectorizer()
ctmTr = cv.fit_transform(X_train)
X_test_dtm = cv.transform(X_test)

# entrainement du dataset par regression linear
model = LogisticRegression()
model.fit(ctmTr, y_train)
LogisticRegression(C=1.0)
# phase de test pour validation du modele
y_pred_class = model.predict(X_test_dtm)


# creation dataframe contenant les donnees predites par rapport aux etiquettes 
X_test = pd.Series(X_test).reset_index(drop=True)
y_test = pd.Series(y_test).reset_index(drop=True)
y_pred_class = pd.Series(y_pred_class)
dp = pd.concat([X_test,y_test,y_pred_class],axis=1)


#impression de la  matrice de confusion
predictions = (model.predict(X_test_dtm) > 0.5).astype("int32")
pcl =  classification_report(y_test,predictions)
cfx = confusion_matrix(y_test, y_pred_class)


"""Prédiction sur une nouvelle phrase"""

def LR_model(phrase,ressenti) :
    
    phrase_trans = cv.transform([phrase])
    
    connexion = pymysql.connect(host='localhost',
                        user='root',
                        password='root')
    cursor = connexion.cursor()
    
    sql = "INSERT INTO PROJET.dataset (name, review,sentiment) VALUES (%s, %s,%s)"
    val = ("new title", phrase,ressenti)
    cursor.execute(sql, val)
    connexion.commit()
    connexion.close()   
    
    return(model.predict(phrase_trans).astype("int32")[0])

