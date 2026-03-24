# Containerized Signature Verification API (Signet)

A fast, lightweight REST API for signature verification using the **Signet** model and PyTorch. This application is fully containerized with Docker, making it incredibly easy to deploy and run anywhere without worrying about environment dependencies or complex PyTorch setups.

## Prerequisites

Before you begin, ensure you have the following installed on your machine:
- [Docker Desktop](https://www.docker.com/products/docker-desktop) (or Docker Engine)
- [model params](https://drive.google.com/file/d/1pO1chhe3utHtTIJCYHT8Qkubvo4uMMbi/view?usp=sharing)

## How to Run the App

To run the containerized version of the application, follow these steps in your terminal:

**1. Build the Docker image:**
This command reads the Dockerfile and builds your isolated environment.
docker build -t sigver-api .

**2. Run the container:**
docker run -d -p 8000:8000 --name running-sigver sigver-api

**3. swagger ui:** 
Open your browser and navigate to: http://localhost:8000/docs

**4. upload your signature images**

**5. Stopping the Container:**
docker stop running-sigver
docker rm running-sigver
 
