''' Modele de regression logistique '''

# importation des bibliotheques sklearn required

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


class Logistic_regression : 
    # classe permettant d'entrainer un modele de regression logistique et de de prédire à partir du modèle
    # si une phrase a un sentiment positif ou négatif
    def __init__(self,df): 
        # Le modele prend en arguement un dataframe
        
        #separation etiquette / dataset à étudier
        df['sentiment'] = df['sentiment'].map({'positive':1,'negative':0})
        self.X = df['review']  #seul la colonne contenant le commentaire sera retenue car relevant
        self.y = df['sentiment']
        
        #separation du dataset pour la phase training et validation
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=101)
        
        # Transformation des données textuelles en matrices prenant en compte le nombre d'occurence de chaque mot
        self.cv = CountVectorizer()     
        # Transformation des données d'entrainement
        self.ctmTr = self.cv.fit_transform(self.X_train)
         # Transformation des données de test
        self.X_test_dtm = self.cv.transform(self.X_test)
        
        # entrainement du dataset par regression linear
        self.model = LogisticRegression()
        self.model.fit(self.ctmTr, self.y_train)
        LogisticRegression(C=1.0,max_iter=1e5,solver='saga')

        
        
    """Prédiction sur une nouvelle phrase"""
    
    def predict(self,phrase) :
        # instance permettant de prédire sur une phrase le sentiment en se basant sur le modèle entraîné ci dessus
        
        phrase_trans = self.cv.transform([phrase])
        return(self.model.predict(phrase_trans).astype("int32")[0])
        #si le retour est 1 la phrase est positive sinon la phrase est negative
        
        
        


