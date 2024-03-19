FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY app .
EXPOSE 88
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "88", "--proxy-headers"]
