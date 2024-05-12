# srl-api

## Prerequisites
Before you begin, ensure your computer meets the following requirements:

- **Docker**: [Install Docker](https://docs.docker.com/get-docker/) on your machine. Docker will handle the installation of other required software, including Python.
- **Python**: It's recommended to have Python 3.12 installed for local testing and development, though not strictly necessary if you're using Docker exclusively. [Install Python](https://www.python.org/downloads/).
- **Operating System**: Compatible with Windows, macOS, and Linux. Specific Docker configurations might be needed based on your operating system.


## Setup
To set up the project, follow these steps:
1. **Clone the Repository**
   ```bash
   git clone https://github.com/andreas-pattichis/srl-api.git
   cd srl-api
    ```
   
2. **Build and Run with Docker**
   ```bash
   docker compose -f docker-compose.dev.yml up --build
    ```
## Usage
Once the API is up and running, you can interact with it using the following endpoints:

- **PHPMyAdmin**
  - Access PHPMyAdmin at: [http://localhost:81/](http://localhost:81/)
  - Use PHPMyAdmin to manage the database directly through a web interface.

- **Trace Data**
  - Retrieve trace data for a specific user:
    ```bash
    curl http://localhost/api/tracedata/[username]
    ```
    Replace `[username]` with the actual username to fetch the trace data for that user.

### Example
Here is an example of how to use the `curl` command to get trace data for a user named "john":
   ```bash
   curl http://localhost/api/tracedata/john