FROM python:3.12-slim
WORKDIR /app
## Update the package list and install build-essential
# build-essential includes gcc, g++ and make
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY app /app/app
EXPOSE 88
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "88", "--proxy-headers"]
## Start with a slim version of Python 3.12
#FROM python:3.12-slim
#
## Set the working directory in the container
#WORKDIR /app
#
## Update the package list and install build-essential
## build-essential includes gcc, g++ and make
#RUN apt-get update && \
#    apt-get install -y --no-install-recommends build-essential && \
#    rm -rf /var/lib/apt/lists/*
#
## Copy the requirements file into the container
#COPY requirements.txt requirements.txt
#
## Install Python dependencies from requirements.txt
#RUN pip install --no-cache-dir --upgrade -r requirements.txt
#
## Copy the rest of your application's code into the container
#COPY . .
#
## Expose port 88 to be accessible from the host
#EXPOSE 88
#
## Set the command to start the uvicorn server
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "88", "--proxy-headers"]


