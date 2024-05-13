
echo "Starting Docker container from the image 'pyspark-image'"

docker run -v /home/ubuntu/de300/DE300/Lab06/spark-sql:/tmp/spark-sql -it \
           -p 8888:8888 \
           --name spark-sql \
	   pyspark-image

echo "Docker container 'spark-sql' is now running."

