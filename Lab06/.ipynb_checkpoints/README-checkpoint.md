# PySpark Docker Setup and Execution

## Steps

1. **Build the Docker File**
   - Create a Docker file to build the PySpark image. This includes setting up the necessary environment, installing PySpark, and configuring any additional dependencies.

2. **Create Containers Using `run.sh`**
   - Execute the `run.sh` script located in the `spark-sql` `word-count` directories. This script will create containers based on the previously built PySpark image.

3. **Run PySpark Scripts**
   - Use the `run-py-spark.sh` script in each directory to execute the PySpark code. This script handles the initialization and execution of the PySpark applications.
