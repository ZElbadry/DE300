echo "Starting Docker container from the image 'pyspark-image'"

docker run -v /home/ubuntu/de300/DE300/HW03:/tmp/HW03 -it \
           -p 8888:8888 \
           --name spark-sql-heart-disease-container \
           pyspark-image
			
echo "Docker container 'spark-sql-heart-disease-container' is now running."
