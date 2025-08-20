Assignment Help

Step1 (Directory 2): To download two datasets from 2 website using API
Example 1) Kaggle 2) Hugging Face

2025-03-02 16:14:58,359 - INFO - Starting Kaggle data ingestion...
2025-03-02 16:15:01,861 - INFO - Kaggle data successfully downloaded and
stored in raw_data/ 2025-03-02 16:15:01,861 - INFO - Starting Hugging
Face data ingestion... 2025-03-02 16:15:14,942 - INFO - Hugging Face
data successfully downloaded and stored in
raw_data/huggingface_churn.csv

Step 2 : (Directory 3): Storing the downloaded data into your local
machine

Step 3 : (Directory 4): Data Validation.

Data validation is the process of ensuring that data is accurate, clean,
and useful before it is processed or stored. It checks that the data
entered into a system meets certain rules or constraints, which are
defined based on the type of data and the business logic involved.

An excel file indicating what are the missing, duplicate, data_types and
negative values

Step 4 : (Directory 5): Data preparation is the process of collecting,
cleaning, organising, and transforming raw data into a format that is
ready for analysis, reporting, or machine learning.

It's a critical step in the data lifecycle because high-quality
decisions and models require high-quality data.

In this case a single code for both Kaggle and hugging face.

Step 5 : (Directory 6): Data transformation is the process of converting
data from its original format into a new structure or format that is
more appropriate for analysis, reporting, or machine learning.

Example """Perform feature engineering and transformations."""

✨ Common Data Transformation Tasks

  -----------------------------------------------------------------------------
  Task                  Description                      Example
  --------------------- -------------------------------- ----------------------
  Normalisation         Scaling numeric data to a        Scale income to a 0--1
                        standard range                   range

  Standardisation       Shifting and scaling data to     Standardise test
                        have mean = 0, std dev = 1       scores

  Encoding              Converting categorical data into "Yes"/"No" → 1/0
                        numerical form                   

  Aggregation           Summarising data                 Total sales per month

  Pivoting/Unpivoting   Restructuring data tables        Rows to columns and
                                                         vice versa

  Filtering             Removing irrelevant data         Only keep rows where
                                                         status = "active"

  Date-Time Conversion  Changing date formats or         "2025-08-18" → year:
                        extracting components            2025
  -----------------------------------------------------------------------------

Step 6 : (Directory 7): A Feature Store is a centralised system or
platform used to store, manage, and serve features for machine learning
models---both during training and in production.

Sample Code:

``` python
# Create a table to store engineered features
cursor.execute('''
    CREATE TABLE IF NOT EXISTS feature_store (
        customerID TEXT PRIMARY KEY,
        tenure INTEGER,
        MonthlyCharges REAL,
        TotalCharges REAL,
        Contract_OneYear INTEGER,
        Contract_TwoYear INTEGER,
        PaymentMethod_CreditCard INTEGER,
        PaymentMethod_ElectronicCheck INTEGER,
        PaymentMethod_MailedCheck INTEGER,
        Churn INTEGER
    )
''')
```

Step 7 : (Directory 8): Data versioning refers to the process of
tracking, managing, and controlling changes made to datasets over time.
Just like software version control (e.g., Git for code), data versioning
ensures that every modification, addition, or deletion in a dataset is
recorded, allowing users to reproduce experiments, roll back to previous
states, and maintain consistency across projects.

Write a code in python get upload the files in GIT

👉 https://github.com/n1000/ml_pipeline/tree/main

Step 8 : (Directory 9): Model Building is the process of developing a
machine learning (ML) or statistical model that can learn from data and
make predictions or decisions.

It involves preparing the data, choosing the right algorithm, training
the model, evaluating its performance, and tuning it for accuracy.

👉 https://github.com/n1000/ml_pipeline/tree/main

Step 9 : (Directory 10): Sample Python code

The word **"Orchestrate"** in data science / ML / cloud computing
contexts means: Coordinating and automating multiple tasks, processes,
or services so they work together smoothly as one system.

It's like being a conductor of an orchestra --- ensuring each instrument
(data pipeline, model training, deployment, monitoring) plays at the
right time in harmony.

Step 10 : (A word document): Content as follows

**Detailed Documentation**
