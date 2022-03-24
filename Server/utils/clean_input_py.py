import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def clean_input(text):
	tokens = word_tokenize(text)
	# conversion en minuscule
	tokens = [w.lower() for w in tokens]
	# suppresion des ponctuations
	table = str.maketrans('', '', string.punctuation)
	stripped = [w.translate(table) for w in tokens]
	# supprimer les mots non alphabétique
	words = [word for word in stripped if word.isalpha()]
	# filtrer les stopwords
	stop_words = set(stopwords.words('english'))
	words = [w for w in words if w not in stop_words]
	return words