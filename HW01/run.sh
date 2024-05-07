#!/bin/bash

# Echo and change permissions of the Docker socket
echo "Changing permissions of the Docker socket..."
sudo chmod 666 /var/run/docker.sock  

# Echo and build the Docker image
echo "Building the Docker image..."
docker build -t jupyter .

# Echo and run the Docker container
echo "Running the Docker container..."
docker run -p 8888:8888 -v /home/ubuntu/de300/DE300:/home/jovyan/ jupyter 

# Echo and change permissions back to secure the Docker data directory
echo "Securing the Docker data directory..."
sudo chmod 755 postgres-data

# Echo and start the Docker Compose services
echo "Starting services using Docker Compose..."
docker-compose -f DE300/HW01/dockerfiles/docker-compose-file.yml up -d
