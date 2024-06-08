# Summary of Airflow DAG for Heart Disease Data Processing

This Airflow Directed Acyclic Graph (DAG) is designed to handle data processing for a heart disease dataset. The DAG includes various stages such as data ingestion, cleaning, normalization, exploratory data analysis, feature engineering, model training, model selection, and evaluation.

## DAG Structure

- **DAG Name**: `Ziad`
- **Schedule**: Runs as configured in `PARAMS['workflow']['workflow_schedule_interval']`.
- **Tags**: `HW4`

## Tasks Description

### Data Preparation

1. **Initialize Schema (`initialize_schema`)**:
   - Drops and recreates the public schema to ensure a clean environment.

2. **Ingest Data (`ingest_data`)**:
   - Loads original data into the system from a configured source.

3. **Data Cleaning and Imputation**:
   - **Stage 1 (`prepare_data_stage1`)**: Cleans and imputes missing values for the first set of data.
   - **Stage 2 (`prepare_data_stage2`)**: Applies similar operations for a second dataset.

### Data Normalization

- Normalizes data to ensure uniformity, important for accurate model performance.

### Exploratory Data Analysis (EDA)

- Conducts basic statistical analysis to understand data distributions and relationships.

### Feature Engineering

- Generates new features that could help improve model predictions from the data.

### Model Training

- Logistic Regression and Support Vector Machine (SVM) models are trained using the processed features.

### Model Selection

- Evaluates all models and selects the best performing model based on accuracy.

### Data Scraping

- Scrapes additional data which may enhance model insights and predictions.

### Data Merging

- Combines scraped data with existing datasets to provide a richer dataset for analysis.

### Model Evaluation

- Evaluates the extended models on the merged dataset to assess performance improvements.

## Task Dependencies

- Proper sequencing of tasks ensures that data flows through cleaning, analysis, and modeling stages effectively.

## Features

- Modular design allowing easy modifications and testing of different stages.
- Use of PythonOperators for flexibility in executing Python functions.
- Use of PostgresOperator for database interactions.

## Configuration

- All configurations are read from a `toml` file, ensuring that the DAG is adaptable to changes in parameters without modifying the DAG code itself.

## Output

- The models generated and selected by this DAG could potentially be used to inform healthcare decisions or provide insights into heart disease research.

---

Check out the running dag on the MWAA dashboard of the course