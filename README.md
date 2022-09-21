<div align="center">
  <h1>
    <br>
    <a href=""><img src="favicon.png" alt="Bad or Good" width="200"></a>
    <br>
      Detection emotion 
    <br>
  </h1>
</div>

<div align="center">
  <a href="#a-propos">A propos</a> •
  <a href="#installation">Installation</a> •
  <a href="#technologies">Technologie utilisé</a> 
</div>

## A propos

Ce repository est composé d'un serveur et d'un client.
Projet universitaire en groupe 

## Try online
https://thavarasa.name/heroku/?name=bert-sentiment-analysis-2022

https://bert-sentiment-analysis-2022.herokuapp.com/

## Technologies

Client : 
```
# No framework (vanilla)
```

Server :
```
  # Environnement : 
    - Python
  
  # Base de donnée : 
    - MySQL

  # Container : 
    - Docker
```

Model & Accuracy: 
```
  # Traditional methods : 
    - Logistic Regression 87.40
    - Naïve Bayes (in process)

  # Deep Learning methods :
    - RNN (LSTM) 77.07
    - DNN (BERT) 93.02
```
## Installation

Depuis votre terminal de commande : 

```bash
# Cloner le repertoire
$ git clone https://github.com/Andy-d-g/Predicteur-d-emotions.git

# Aller dans le repertoire
$ cd Predicteur-d-emotions/

# Install dependencies
$ pip3 install --no-cache-dir -r ./Server/requirements.txt
$ pip3 install --default-timeout=100 --no-cache-dir torch==1.10.2

# Install bert model
$ python3 ./Server/utils/download_model.py 1KVo4Z1vThfHI732Asg-OeIYTISwV1kpe ./Server/Model/bert.pickle

# Lancer Mysql
$ mysql.server start (macOS)

# Permettre à MySql de charger des fichiers 
$ mysql
$ set global local_infile=true;
$ exit;

# Changement les identifiants de connexion à la base de donnée (Server/api.py:28)

# Lancer le projet
$ cd Server
$ uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Ouvrir index.html
$ open Client/index.html
```

Build & Run model with Docker (not working for moment : probleme with DB) : 
- [How to install docker](https://docs.docker.com/engine/install/)

```bash
# build docker image
docker build -t mymodel -f Dockerfile .

# run docker image 
docker run -p 8000:8000 mymodel

# use : 
http://localhost:8000/text/{text to predict}/{sentiment}
```

## Screenshot
<img src="Capture.PNG" alt="Bad or Good">

## Sources
- [Bert Model](https://skimai.com/fine-tuning-bert-for-sentiment-analysis/)

## Contributeurs

@Andy-d-g
@pthavarasa
@rolemoine
@layaida
@musescorecontributor
