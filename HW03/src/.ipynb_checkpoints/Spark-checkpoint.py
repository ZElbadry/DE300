#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[ ]:


# I am only insatlling pandas to plot the data easily. I haven't used in cleaning, imputing or building the machine learning model
get_ipython().system('pip install pandas')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install seaborn')
from pyspark.sql.functions import col as spark_col, when
from pyspark.ml.feature import Imputer
from pyspark.sql import functions as F
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup

from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.types import StringType
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import FloatType, StringType

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler


# ## Create Spark Session & Read the data

# In[1]:


# Initialize Spark session
spark = SparkSession.builder \
    .appName("Heart-Disease-Prediction") \
    .getOrCreate()

# Load the dataset
df = spark.read.csv('s3://ziadelbadry/HW3/data/heart_disease.csv', header=True, inferSchema=True)


# In[2]:


df.show(5)


# ## drop unused columns

# In[3]:


# List of columns to retain
columns_to_keep = [
    'age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'smoke', 'fbs', 
    'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 'exang', 
    'oldpeak', 'slope', 'target'
]

# Select the columns to retain
df = df.select(columns_to_keep)
df.show(5)


# ## Cleaning & Imputing

# In[4]:


# Define the columns explicitly
binary_categorical_cols = ['sex', 'painloc', 'painexer', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'exang', 'slope', 'target']
numerical_cols = ['age', 'thaldur', 'trestbps', 'oldpeak', 'thalach']

# Cleaning 'trestbps': Replace values less than 100 mm Hg with the median of values >= 100
median_trestbps = df.filter(spark_col('trestbps') >= 100).approxQuantile('trestbps', [0.5], 0.001)[0]
df = df.withColumn('trestbps', when(spark_col('trestbps') < 100, median_trestbps).otherwise(spark_col('trestbps')))

# Cleaning 'oldpeak': Replace values less than 0 and those greater than 4 with the median of values within range
median_oldpeak = df.filter((spark_col('oldpeak') >= 0) & (spark_col('oldpeak') <= 4)).approxQuantile('oldpeak', [0.5], 0.001)[0]
df = df.withColumn('oldpeak', when((spark_col('oldpeak') < 0) | (spark_col('oldpeak') > 4), median_oldpeak).otherwise(spark_col('oldpeak')))

# Impute remaining numerical columns using median
imputer = Imputer(inputCols=numerical_cols, outputCols=[f"{col}_imputed" for col in numerical_cols]).setStrategy("median")
df_imputed = imputer.fit(df).transform(df)

# Drop the original columns and rename imputed columns
for num_col in numerical_cols:
    df_imputed = df_imputed.drop(num_col).withColumnRenamed(f"{num_col}_imputed", num_col)

# Ensure valid binary values (0 or 1) where applicable for specific columns
for binary_col in ['fbs', 'prop', 'nitr', 'pro', 'diuretic']:
    df_imputed = df_imputed.withColumn(binary_col, when(spark_col(binary_col) > 1, None).otherwise(spark_col(binary_col)))


# ## Analysis of the data

# In[5]:


# Function to get the mode of a column
def get_mode(df, col_name):
    return df.groupBy(col_name).count().orderBy(F.desc('count')).first()[0]

# Fill missing values in categorical columns with the mode
for col in binary_categorical_cols:
    mode_value = get_mode(df_imputed, col)
    df_imputed = df_imputed.fillna({col: mode_value})

# Verify if there are any missing values left
for col in binary_categorical_cols:
    print(f"Missing values in {col}: {df_imputed.filter(F.col(col).isNull()).count()}")

# Drop rows where the slope value is 0
df_imputed = df_imputed.filter(df_imputed.slope != 0)

# Verify if there are any slope values of 0 left
print(f"Remaining rows with slope value 0: {df_imputed.filter(F.col('slope') == 0).count()}")


# In[6]:


# Identify binary and categorical columns
binary_categorical_cols = ['sex', 'painloc', 'painexer', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'exang', 'slope', 'target']

# Display value counts for each binary or categorical column
for column in binary_categorical_cols:
    print(f"\nValue Counts for {column}:")
    df_imputed.groupBy(column).count().show()


# In[7]:


numerical_cols = ['age', 'thaldur', 'trestbps', 'oldpeak', 'thalach']

# Display basic statistics for numerical columns
print("Basic Statistics for Numerical Columns:")
df_imputed.select(numerical_cols).describe().show()


# In[8]:


# Configure the notebook to show plots inline
get_ipython().run_line_magic('matplotlib', 'inline')

# Optional: Configure the style for seaborn plots
sns.set(style="whitegrid")

# Convert the DataFrame to Pandas for plotting
numerical_df = df_imputed.select(numerical_cols).toPandas()

# Creating a combined box plot
plt.figure(figsize=(12, 6))  # Set the figure size
sns.boxplot(data=numerical_df, orient="v")  # Vertical box plots
plt.title('Combined Box Plot for Numerical Columns')
plt.show()


# In[9]:


# Convert the DataFrame to Pandas for plotting
binary_df = df_imputed.select(binary_categorical_cols).toPandas()

# Determine the layout size
n_cols = 4  # Increase number of columns in the grid for a more compact display
n_rows = (len(binary_categorical_cols) + n_cols - 1) // n_cols  # Calculate required number of rows in the grid

# Create figure and axes for the subplots
plt.figure(figsize=(12, 10))  # Adjust overall figure size to reduce plot width

for i, col in enumerate(binary_categorical_cols):
    plt.subplot(n_rows, n_cols, i + 1)  # Create a subplot for each column
    sns.countplot(x=binary_df[col], color='skyblue')  # Specify the color here
    plt.title(f'Value Counts for {col}')
    plt.xticks(rotation=0)  # Rotate x labels for better readability if necessary
    plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Adjust the space between plots

# Add a central title for the entire figure
plt.suptitle('Distribution of Binary and Categorical Variables', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()


# # Analysis of Basic Statistics
# 
# ## Numerical Columns Analysis
# 
# ### 1. Age:
# - **Mean**: The average age in the dataset is approximately 53.5 years.
# - **Standard Deviation**: The age data shows a moderate spread around the mean with a standard deviation of about 9.4 years, indicating variability in the patient age group.
# - **Range**: The ages range from 28 to 77 years, suggesting a wide distribution from young adults to elderly patients.
# - **Quartiles**: The 25th percentile is at 47 years and the 75th percentile is at 60 years, highlighting that half of the patients are between these ages.
# 
# ### 2. Thaldur (Exercise Duration):
# - **Mean**: The mean exercise duration is approximately 8.6 minutes.
# - **Standard Deviation**: There is a variability of 3.63 minutes in exercise duration among the patients.
# - **Range**: The duration varies from 1 to 24 minutes, indicating diverse physical capacities or exercise tests of varying intensities.
# - **Quartiles**: 50% of the data falls between 6 and 10.3 minutes, suggesting that most exercise durations are within a reasonable range for standard stress tests.
# 
# ### 3. Trestbps (Resting Blood Pressure):
# - **Mean**: The average resting blood pressure is about 132.6 mm Hg.
# - **Standard Deviation**: With a standard deviation of 17.4 mm Hg, this indicates variability in patient blood pressure levels.
# - **Minimum**: The minimum blood pressure recorded is 100 mm Hg, aligned with clinical interventions to address values below this threshold.
# - **Quartiles**: The interquartile range from 120 to 140 mm Hg represents typical blood pressure readings for a general adult population.
# 
# ### 4. Oldpeak (ST Depression Induced by Exercise):
# - **Mean**: The mean ST depression is 0.84, which can indicate varying degrees of heart stress during exercise.
# - **Standard Deviation**: A standard deviation of 0.97 suggests significant variability, which could correlate with different levels of coronary artery disease.
# - **Range**: The values range from 0 to 4, with the upper range typically indicating more severe exercise-induced cardiac stress.
# 
# ### 5. Thalach (Maximum Heart Rate Achieved):
# - **Mean**: The average maximum heart rate achieved is approximately 137.5 beats per minute.
# - **Standard Deviation**: The standard deviation of 25.17 bpm indicates varied fitness levels or cardiac responses among individuals.
# - **Range**: The range from 60 to 202 bpm shows a wide variation, likely reflecting diverse age groups and health conditions.
# 
# ## Categorical Columns Analysis (Value Counts)
# 
# ### Binary Categories:
# - **Sex**: Predominantly male (711 out of 899), highlighting a male-heavy dataset which might affect gender-specific analysis.
# - **Pain Location (Painloc)**: Most patients (850 out of 899) reported pain located in a typical region associated with heart disease.
# - **Exercise Induced Pain (Painexer)**: A majority (648 out of 899) experienced pain during exercise, a critical indicator of exercise-related cardiac issues.
# - **Fasting Blood Sugar (FBS) above 120 mg/dl**: A smaller portion (135 out of 899) had elevated fasting blood sugar, a risk factor for cardiovascular disease.
# - **Use of Properties (Prop)**: Less than a quarter (214 out of 899) are on properties, possibly indicating medication or interventions.
# - **Use of Nitrates (Nitr)**: Around a quarter (222 out of 899) use nitrates, often used for managing heart pain.
# - **Pro (Prophylactic use of Aspirin)**: A minority (144 out of 899) use aspirin prophylactically, suggesting preventive measures in a subset of patients.
# - **Diuretic Use**: Relatively few (92 out of 899) are on diuretics, indicating specific treatment strategies.
# - **Exercise Induced Angina (Exang)**: About 37% (330 out of 899) experienced angina induced by exercise, significant for assessing exercise tolerance.
# - **Slope of Peak Exercise ST Segment (Slope)**: Most common is a slope of 2 (643 out of 899), which can relate to specific heart function during exercise.
# - **Heart Disease Presence (Target)**: Almost evenly split with 495 positive and 404 negative cases, indicating a balanced dataset regarding the outcome variable.
# 
# ## Box Plot Analysis and Reasoning for Retaining Outliers
# 
# The provided box plot visualizes the distributions of several numerical variables including age, thaldur (duration of the exercise test in minutes), trestbps (resting blood pressure), oldpeak (ST depression induced by exercise relative to rest), and thalach (maximum heart rate achieved). Here's an in-depth analysis of each variable and the reasoning behind the decision to retain outliers:
# 
# ### Detailed Analysis
# 
# #### Age:
# - Shows a symmetric distribution with a slightly wider interquartile range, typical for an adult population. There are no outliers, indicating a normal age distribution for the study population.
# 
# #### Thaldur:
# - The exercise duration shows a compact interquartile range with outliers on the higher end. These outliers could represent individuals with exceptional endurance, and retaining them provides a fuller picture of physical capabilities across populations.
# 
# #### Trestbps (Resting Blood Pressure):
# - This variable has a normal distribution with outliers mainly on the higher side, likely indicating individuals with potential hypertension. Such outliers are crucial for understanding the upper extremes of blood pressure variations.
# 
# #### Oldpeak:
# - Notable for its significant outliers, especially on the lower side. This variability is typical as individual responses to exercise can vary dramatically based on cardiac health, making it essential to include these outliers for a comprehensive model.
# 
# #### Thalach (Maximum Heart Rate Achieved):
# - Displays a broad range with several outliers both low and high, reflecting the physiological diversity in heart rate responses among different individuals, possibly influenced by age, fitness level, or cardiac conditions.
# 
# ## Reasoning for Keeping Outliers
# 
# ### Better Representation:
# - Retaining outliers ensures that the model reflects a realistic spectrum of patient responses and conditions, enhancing its accuracy and generalizability in clinical settings.
# 
# ### Population Diversity:
# - Including outliers in the dataset helps the model learn from a wide range of physiological data, improving its performance and robustness when deployed across diverse patient groups.
# 
# ### Clinical Insights:
# - Outliers can often highlight unique or extreme conditions that may be clinically significant. By analyzing these cases, we can gain deeper insights into unusual health patterns or responses, which could inform more effective treatment strategies.
# 
# ## Conclusion
# 
# The choice to keep outliers in the training dataset is driven by the need to develop a model that not only performs well statistically but also is capable of handling the full range of real-world clinical scenarios. This approach ensures that the predictive models we develop are both robust and sensitive to the nuances of individual patient data, thereby improving their utility and reliability in healthcare applications.
# 

# ## Web-Scrabing to Impute Smoke Column

# In[10]:


# URL of the page to be scraped
url = "https://www.abs.gov.au/statistics/health/health-conditions-and-risks/smoking/latest-release"

# Send a GET request
response = requests.get(url)
if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')

    # Attempt to find the table by a unique characteristic such as a caption
    caption_text = "Proportion of people 15 years and over who were current daily smokers by age, 2011–12 to 2022"
    table = soup.find("caption", string=lambda text: caption_text in text if text else False).find_parent('table') if soup.find("caption", string=lambda text: caption_text in text if text else False) else None

    if table:
        # Extracting headers from the table's header row
        headers = [th.get_text(strip=True) for th in table.find('thead').find_all('th')[1:]]  # Skip the first empty <th>
        headers.insert(0, "Age Group")  # Manually add "Age Group" as the first header

        # Collecting data rows
        data = []
        for row in table.find('tbody').find_all('tr'):
            cells = [td.get_text(strip=True) for td in row.find_all('td')]
            age_group = row.find('th').get_text(strip=True) if row.find('th') else ""
            cells.insert(0, age_group)  # Insert the age group at the start of the list of cells

            if len(cells) == len(headers):  # Ensure the row has the correct number of elements
                data.append(cells)

        # Creating a Spark DataFrame
        df_age_group = spark.createDataFrame(data, schema=headers)

        # Convert relevant columns to float
        for col_name in ['2011–12 (%)', '2014–15 (%)', '2017–18 (%)', '2022 (%)']:
            df_age_group = df_age_group.withColumn(col_name, col(col_name).cast('float'))

        # Compute the average smoking rate across all specified years for each row
        df_age_group = df_age_group.withColumn(
            'Average Smoking Rate (%)',
            (col('2011–12 (%)') + col('2014–15 (%)') + col('2017–18 (%)') + col('2022 (%)')) / 4
        )

        # Select only the age group and the average smoking rate
        df_age_group = df_age_group.select('Age Group', 'Average Smoking Rate (%)')

        df_age_group.show(8)
    else:
        print("Table not found.")
else:
    print("Failed to retrieve data, status code:", response.status_code)


# In[11]:


# URL of the page to be scraped
url = "https://www.abs.gov.au/statistics/health/health-conditions-and-risks/smoking/latest-release"

# Send a GET request to the URL
response = requests.get(url)
if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')

    # Attempt to find the table by searching for a specific caption
    caption_text = "Proportion of people 15 years and over who were current daily smokers by age and sex, 2022"
    table = soup.find("caption", string=lambda text: caption_text in text if text else False).find_parent('table') if soup.find("caption", string=lambda text: caption_text in text if text else False) else None

    if table:
        # Extract headers from the table's thead element
        headers = [th.get_text(strip=True) for th in table.find('thead').find_all('th')[1:]]  # Skip the first empty <th>
        headers.insert(0, "Age Group")  # Manually add "Age Group" as the first header

        # Collecting data rows
        data = []
        for row in table.find('tbody').find_all('tr'):
            cells = [td.get_text(strip=True) for td in row.find_all('td')]
            age_group = row.find('th').get_text(strip=True) if row.find('th') else ""
            cells.insert(0, age_group)  # Insert the age group at the start of the list of cells

            if len(cells) == len(headers):  # Ensure the row has the correct number of elements
                data.append(cells)

        # Creating a Spark DataFrame
        df_age_sex_group = spark.createDataFrame(data, schema=headers)

        # Convert relevant columns to float
        for col_name in ['Males (%)', 'Females (%)']:
            df_age_sex_group = df_age_sex_group.withColumn(col_name, col(col_name).cast('float'))

        # Select only the age group and the converted columns
        df_age_sex_group = df_age_sex_group.select('Age Group', 'Males (%)', 'Females (%)')

        df_age_sex_group.show(8)
    else:
        print("Table not found under this caption.")
else:
    print("Failed to retrieve data, status code:", response.status_code)


# In[13]:


sc = SparkContext.getOrCreate()


# Broadcast the data required for the UDF
age_group_broadcast = sc.broadcast({row['Age Group']: row['Average Smoking Rate (%)'] for row in df_age_group.collect()})
age_sex_group_broadcast = sc.broadcast({row['Age Group']: (row['Males (%)'], row['Females (%)']) for row in df_age_sex_group.collect()})

# Define the impute_smoking function
def impute_smoking(age_group, gender):
    age_group_dict = age_group_broadcast.value
    age_sex_group_dict = age_sex_group_broadcast.value
    
    if gender == 1:  # Male
        if age_group in age_sex_group_dict:
            age_rate = age_group_dict.get(age_group, None)
            male_rate, female_rate = age_sex_group_dict[age_group]
            rate = (age_rate * male_rate) / female_rate if female_rate != 0 else male_rate
        else:
            rate = None 
    else:  # Female
        if age_group in age_group_dict:
            rate = age_group_dict.get(age_group, None)
        else:
            rate = None  

    return rate

# Convert the function to a UDF
impute_smoking_udf = udf(impute_smoking, FloatType())

# Assuming df_imputed is your main DataFrame with 'Age Group' and 'sex' columns

# Define the function to categorize age
def categorize_age(age):
    if age < 18:
        return "15–17"
    elif age < 25:
        return "18–24"
    elif age < 35:
        return "25–34"
    elif age < 45:
        return "35–44"
    elif age < 55:
        return "45–54"
    elif age < 65:
        return "55–64"
    elif age < 75:
        return "65–74"
    else:
        return "75 years and over"

# Convert the function to a UDF
categorize_age_udf = udf(categorize_age, StringType())

# Load your actual main DataFrame
# df_imputed = spark.read.csv('path_to_your_actual_data.csv', header=True, inferSchema=True)

# Apply the UDF to create a new column 'Age Group'
df_imputed = df_imputed.withColumn('Age Group', categorize_age_udf(col('age')))

# Apply the UDF to create a new column 'smoke'
df_imputed = df_imputed.withColumn('smoke', impute_smoking_udf(col('Age Group'), col('sex')))

# Round and convert the 'smoke' column to integer
df_imputed = df_imputed.withColumn('smoke', col('smoke').cast('int'))

# Show the result
df_imputed.show(5)


# ## Build the Machine Learning Model

# In[14]:


# Drop the 'Age Group' column from df_imputed
df_imputed = df_imputed.drop('Age Group')


# In[15]:


from pyspark.sql.functions import col
from pyspark.sql import functions as F

# Add an index column to use for stratified sampling
df_imputed = df_imputed.withColumn("index", F.monotonically_increasing_id())

# Compute the proportion of positive labels
label_counts = df_imputed.groupBy('target').count().collect()
positive_ratio = label_counts[1]['count'] / (label_counts[0]['count'] + label_counts[1]['count'])

# Split the data into training and test sets with stratification
train_df = df_imputed.sampleBy("target", fractions={0: 0.9, 1: 0.9}, seed=42)
test_df = df_imputed.subtract(train_df)

train_df.show(5)
test_df.show(5)


# In[16]:


# Assemble feature columns into a single vector column
feature_cols = [col for col in df_imputed.columns if col != 'target']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

train_df = assembler.transform(train_df)
test_df = assembler.transform(test_df)

# Define the models
model_lr = LogisticRegression(featuresCol='features', labelCol='target')
model_rf = RandomForestClassifier(featuresCol='features', labelCol='target')
model_gb = GBTClassifier(featuresCol='features', labelCol='target')

# Define parameter grids
param_grid_lr = (ParamGridBuilder()
                 .addGrid(model_lr.regParam, [0.01, 0.1, 1])
                 .addGrid(model_lr.elasticNetParam, [0.0, 0.5, 1.0])
                 .build())

param_grid_rf = (ParamGridBuilder()
                 .addGrid(model_rf.numTrees, [50, 100, 200])
                 .addGrid(model_rf.maxDepth, [5, 10, 20])
                 .build())

param_grid_gb = (ParamGridBuilder()
                 .addGrid(model_gb.maxIter, [50, 100])
                 .addGrid(model_gb.maxDepth, [3, 5])
                 .addGrid(model_gb.stepSize, [0.01, 0.1])
                 .build())

# Define evaluator
evaluator = BinaryClassificationEvaluator(labelCol='target', metricName='areaUnderROC')

# Setup CrossValidator for each model
crossval_lr = CrossValidator(estimator=model_lr, estimatorParamMaps=param_grid_lr, evaluator=evaluator, numFolds=5)
crossval_rf = CrossValidator(estimator=model_rf, estimatorParamMaps=param_grid_rf, evaluator=evaluator, numFolds=5)
crossval_gb = CrossValidator(estimator=model_gb, estimatorParamMaps=param_grid_gb, evaluator=evaluator, numFolds=5)

# Fit CrossValidator models
cv_model_lr = crossval_lr.fit(train_df)
cv_model_rf = crossval_rf.fit(train_df)
cv_model_gb = crossval_gb.fit(train_df)

# Retrieve the best models
best_model_lr = cv_model_lr.bestModel
best_model_rf = cv_model_rf.bestModel
best_model_gb = cv_model_gb.bestModel


# In[17]:


# Evaluate the models on the test set
results_lr = best_model_lr.transform(test_df)
results_rf = best_model_rf.transform(test_df)
results_gb = best_model_gb.transform(test_df)

auc_lr = evaluator.evaluate(results_lr)
auc_rf = evaluator.evaluate(results_rf)
auc_gb = evaluator.evaluate(results_gb)

print(f"Logistic Regression AUC: {auc_lr:.4f}")
print(f"Random Forest AUC: {auc_rf:.4f}")
print(f"Gradient Boosting AUC: {auc_gb:.4f}")


# # Analysis of Model Performance
# 
# After training and evaluating three different binary classification models on the heart disease dataset, we obtained the following AUC scores on the test set:
# 
# - **Logistic Regression AUC**: 0.8841
# - **Random Forest AUC**: 0.8929
# - **Gradient Boosting AUC**: 0.8516
# 
# ## Interpretation of Results
# 
# 1. **Logistic Regression**:
#    - **AUC**: 0.8841
#    - Logistic Regression is a linear model that is often used as a baseline for binary classification tasks. It is easy to interpret and quick to train. The model achieved a high AUC, indicating good performance in distinguishing between positive and negative classes.
# 
# 2. **Random Forest**:
#    - **AUC**: 0.8929
#    - Random Forest is an ensemble learning method that combines multiple decision trees to improve classification performance. It tends to perform well with minimal parameter tuning and is robust to overfitting. The Random Forest model achieved the highest AUC among the three models, indicating the best performance in distinguishing between classes.
# 
# 3. **Gradient Boosting**:
#    - **AUC**: 0.8516
#    - Gradient Boosting is another ensemble learning method that builds models sequentially, with each new model attempting to correct the errors of the previous one. While Gradient Boosting can achieve high performance with careful tuning, in this case, it achieved the lowest AUC among the three models.
# 
# ## Model Selection
# 
# The AUC (Area Under the ROC Curve) is a performance metric that indicates how well the model can distinguish between the positive and negative classes. A higher AUC value indicates better performance. Based on the AUC scores:
# 
# - **Random Forest** achieved the highest AUC score (0.8929), making it the best performing model in terms of distinguishing between positive and negative cases of heart disease.
# 
# ## Conclusion
# 
# Based on the AUC scores obtained from the cross-validation and evaluation on the test set, the **Random Forest** model is the best choice for predicting heart disease in this dataset. It provides the highest discriminative power among the models evaluated.
# 
