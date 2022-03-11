FROM python:3.7

COPY Server/requirements.txt ./requirements.txt
COPY utils/download_model.py ./download_model.py

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install --default-timeout=100 --no-cache-dir torch==1.10.2

COPY Model/Bert/bert_sentiment_predict.py ./bert_sentiment_predict.py
COPY Server/api.py ./api.py

EXPOSE 8000

RUN pip install gdown
COPY utils/download_model.py ./download_model.py
RUN python3 download_model.py 1KVo4Z1vThfHI732Asg-OeIYTISwV1kpe bert_sentiment_model.pickle

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
