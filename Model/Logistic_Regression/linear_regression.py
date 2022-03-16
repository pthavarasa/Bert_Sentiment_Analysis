# -*- coding: utf-8 -*-


# importation des bibliotheques sklearn required

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

"""importation du dataset"""

#ouverture du fichier txt
contenu = open("dataset.txt", "r").read()

contenu = contenu.split("\n")
for i in range (len(contenu)) :
    contenu[i] = contenu[i].split(";")


# definition du nom des colonnes
df = pd.DataFrame(contenu)
df.columns = contenu[0]
df= df.drop(0)


# attibution d'une etiquette pour les sentiments positifs:1 et negatifs:0
df['sentiment'] = df['sentiment'].map({'positive':1,'negative':0})

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

def LR_model(phrase) :
    phrase_trans = cv.transform([phrase])
    return(model.predict(phrase_trans).astype("int32")[0])

