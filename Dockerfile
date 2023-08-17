FROM python:3.10

# This is the default Docker app folder
WORKDIR /usr/src/app

COPY requirements.txt requirements.txt

#RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]
#CMD ["python", "-m", "main.py"]
