# Project Title: End-to-End Data Management Pipeline for Machine Learning

# Project Brief: Predicting Customer Churn

## 1. Business Problem

The business is facing a significant challenge with customer churn, where existing customers are discontinuing their services. This leads to revenue loss and increased costs associated with acquiring new customers. The goal is to proactively identify customers who are at a high risk of churning so that targeted retention strategies can be implemented to keep them. This project focuses on building an automated data pipeline to train a model that predicts this "addressable churn."

## 2. Business Objectives

*   **Reduce Customer Churn:** Decrease the quarterly churn rate by 5% within the next six months.
*   **Increase Customer Retention:** Improve the overall customer retention rate by implementing targeted marketing campaigns for at-risk customers.
*   **Identify Churn Drivers:** Understand the key factors contributing to customer churn to inform business strategy and product improvements.

## 3. Key Data Sources and Attributes

For this project, we will use two primary data sources to simulate a real-world scenario.

*   **Source 1: Telco Customer Data (CSV File)**
    *   **customerID:** Unique identifier for each customer.
    *   **gender, SeniorCitizen, Partner, Dependents:** Demographic information.
    *   **tenure:** How many months the customer has been with the company.
    *   **PhoneService, MultipleLines, InternetService, OnlineSecurity, etc.:** Details of the services each customer has signed up for.
    *   **Contract, PaperlessBilling, PaymentMethod:** Customer account and payment information.
    *   **MonthlyCharges, TotalCharges:** Financial information.
    *   **Churn:** The target variable; whether the customer churned or not.

*   **Source 2: Hugging Face Dataset (JSON via API)**
    *   Dataset: `scikit-learn/churn-prediction` (retrieved via Hugging Face Datasets Server API)
    *   Format: JSON (top-level object with `rows`, each `row` containing fields below)
    *   Schema (matches Telco CSV):
        *   **customerID**, **gender**, **SeniorCitizen**, **Partner**, **Dependents**
        *   **tenure**, **PhoneService**, **MultipleLines**, **InternetService**, **OnlineSecurity**
        *   **OnlineBackup**, **DeviceProtection**, **TechSupport**, **StreamingTV**, **StreamingMovies**
        *   **Contract**, **PaperlessBilling**, **PaymentMethod**, **MonthlyCharges**, **TotalCharges**
        *   **Churn**

## 4. Expected Pipeline Outputs

1.  **Clean Datasets for EDA:** A clean, validated dataset available for exploratory data analysis to uncover initial insights.
2.  **Transformed Features for Machine Learning:** A processed dataset with normalized numerical features, encoded categorical features, and newly engineered features ready for model training.
3.  **A Deployable Model to Predict Customer Churn:** A serialized, versioned machine learning model file (`.pkl`) that can be loaded to make predictions on new customer data.

## 5. Measurable Evaluation Metrics

The performance of the churn prediction model will be evaluated using the following metrics:

*   **Accuracy:** The overall percentage of correct predictions.
*   **Precision:** Of all the customers the model predicted would churn, what percentage actually churned. This is important to avoid wasting resources on customers who were never going to leave.
*   **Recall:** Of all the customers who actually churned, what percentage did the model correctly identify. This is crucial for identifying as many at-risk customers as possible.
*   **F1-Score:** The harmonic mean of Precision and Recall, providing a single score that balances both metrics.