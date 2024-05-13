#!/bin/bash
# Script to run PySpark job

# Ensure the environment knows where Spark is
export SPARK_HOME=/opt/spark
export PATH=$SPARK_HOME/bin:$PATH

# Run the PySpark script
spark-submit --master local[4] /tmp/spark-sql/pyspark_script.py

# Echo a completion message
echo "PySpark job completed and the data is saved to './data/output.csv'"
