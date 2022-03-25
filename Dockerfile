FROM python:3.7

COPY Server/requirements.txt ./requirements.txt
COPY Server/utils/download_model.py ./download_model.py

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install --default-timeout=100 --no-cache-dir torch==1.10.2

# Get database
RUN mkdir DB
COPY Server/DB/DataBase.py ./DB/DataBase.py

# Get utils
RUN mkdir utils
COPY Server/utils/format_dataset.py ./utils/format_dataset.py
COPY Server/utils/clean_input.py ./utils/clean_input.py
COPY Server/utils/download_model.py ./utils/download_model.py

# Get model
RUN mkdir Model
COPY Server/Model/bert.py ./Model/bert.py
COPY Server/Model/logistic_regression.py ./Model/logistic_regression.py

# Get server
COPY Server/api.py ./api.py

EXPOSE 8000

RUN pip install gdown
RUN python3 ./utils/download_model.py 1KVo4Z1vThfHI732Asg-OeIYTISwV1kpe ./Model/bert.pickle

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
