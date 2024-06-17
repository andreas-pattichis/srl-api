FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements.txt
# Update the package list and install build-essential
# build-essential includes gcc, g++ and make
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY app /app/app
EXPOSE 80
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--proxy-headers"]

