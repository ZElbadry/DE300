
echo "Starting Docker container from the image 'pyspark-image'"

docker run -v /home/ubuntu/de300/DE300/Lab06/word-count:/tmp/word-count-c -it \
           -p 8888:8888 \
           --name word-count-container \
	   pyspark-image

echo "Docker container 'word-count' is now running."

