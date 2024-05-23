echo "Starting Docker container from the image 'pyspark-image'"

docker run -v /home/ubuntu/de300/DE300/Lab07/ml:/tmp/ml -it \
           -p 8888:8888 \
           --name spark-sql-container \
           pyspark-image

echo "Docker container 'spark-sql-container' is now running."

