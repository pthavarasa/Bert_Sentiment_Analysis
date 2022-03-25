FROM python:3.7

COPY Server/requirements.txt ./requirements.txt
COPY Server/utils/download_model.py ./download_model.py

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install --default-timeout=100 --no-cache-dir torch==1.10.2

COPY Server/Model/bert.py ./bert.py
COPY Server/api.py ./api.py

EXPOSE 8000

RUN pip install gdown
COPY Server/utils/download_model.py ./download_model.py
RUN python3 download_model.py 1KVo4Z1vThfHI732Asg-OeIYTISwV1kpe bert.pickle

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
