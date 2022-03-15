def clean_input(text):
	from nltk.tokenize import word_tokenize
	tokens = word_tokenize(text)
	# conversion en minuscule
	tokens = [w.lower() for w in tokens]
	# suppresion des ponctuations
	import string
	table = str.maketrans('', '', string.punctuation)
	stripped = [w.translate(table) for w in tokens]
	# supprimer les mots non alphabétique
	words = [word for word in stripped if word.isalpha()]
	# filtrer les stopwords
	from nltk.corpus import stopwords
	stop_words = set(stopwords.words('english'))
	words = [w for w in words if not w in stop_words]

	return words