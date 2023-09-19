FROM python:3.10

# This is the default Docker app folder
WORKDIR /usr/src/app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY app /app/app
COPY create_schema.py /app/create_schema.py
