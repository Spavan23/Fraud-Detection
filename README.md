# Fraud-Detection
Problem Statement

The problem statement chosen for this project is to predict fraudulent credit card transactions using machine learning models. This analysis will involve customer-level data collected during a research collaboration between Worldline and the Machine Learning Group. The dataset comprises a total of 284,807 transactions, out of which 492 are labeled as fraudulent, highlighting a significant class imbalance that needs to be addressed prior to model building.

The dataset is taken from the Kaggle Website website and it has a total of 2,84,807 transactions, out of which 492 are fraudulent. Since the dataset is highly imbalanced, so it needs to be handled before model building.

# Business Problem Overview

For banks, retaining highly profitable customers is a top priority. However, banking fraud poses a significant threat to achieving this goal. Banking fraud can lead to substantial financial losses, damage to trust, and a loss of credibility for both banks and their customers.

A Nilson Report estimated that by 2020, banking frauds could account for $30 billion globally. The rise in digital payment channels has increased the number and sophistication of fraudulent transactions, making fraud detection a necessity rather than a luxury.

Credit card fraud detection using machine learning provides a proactive way to combat these frauds. Machine learning reduces the need for time-consuming manual reviews, minimizes chargebacks and fees, and ensures legitimate transactions are not incorrectly denied. These advancements enable institutions to protect both their customers and their financial assets effectively.

# Understanding and Defining Fraud

Credit card fraud involves any dishonest act or behavior to obtain financial information or funds without the proper authorization of the account holder. Common methods of credit card fraud include:

Skimming: Duplication of the magnetic strip information on a card.

Manipulation/Alteration: Altering genuine cards to conduct fraudulent activities.

Counterfeit Cards: Creating fake cards with legitimate account details.

Stolen or Lost Cards: Using lost or stolen cards to make unauthorized transactions.

Fraudulent Telemarketing: Collecting card details through deceptive telemarketing schemes.

# Data Dictionary

The dataset used in this project contains credit card transactions made by European cardholders over a two-day period in September 2013. It consists of 284,807 transactions, of which 492 are labeled as fraudulent. The dataset is highly imbalanced, with fraudulent transactions accounting for just 0.172% of the total. Principal Component Analysis (PCA) has been applied to most features to maintain confidentiality.

Time: Seconds elapsed between the first transaction and subsequent transactions.

Amount: Transaction amount.

V1 to V28: Principal components derived using PCA.

Class: Labels indicating fraud (1) or non-fraud (0).

# Project Pipeline

The project pipeline consists of the following stages:

1. Data Understanding

  Load the dataset and examine its features to gain insights.
  Identify relevant features for model building.

2. Exploratory Data Analysis (EDA)

  Perform univariate and bivariate analysis to understand the distribution and relationships between features.
  Address skewness in the data to improve model performance.
  Verify if feature transformations are required.

3. Train/Test Split

  Split the dataset into training and testing sets to evaluate model performance on unseen data.
  Use k-fold cross-validation to ensure robust validation, particularly for the minority class.
  Select an appropriate value of k to ensure the minority class is adequately represented.

4. Model Building and Hyperparameter Tuning

  Experiment with various machine learning models, such as logistic regression, random forest, and gradient boosting.
  Apply hyperparameter tuning to optimize model performance.
  Use sampling techniques, such as oversampling (SMOTE) or undersampling, to address class imbalance.

5. Model Evaluation

  Evaluate models using metrics that are sensitive to the imbalanced nature of the dataset.
  Focus on metrics such as precision, recall, F1-score, and the area under the Receiver Operating Characteristic (ROC-AUC) curve to assess the ability to correctly identify fraudulent transactions.
  Analyze false positives and false negatives to understand the trade-offs and improve the model’s decision-making process.
  Select the model that achieves the best balance between detecting fraudulent transactions (minimizing false negatives) and maintaining a low rate of false positives.

# Enhanced Model Evaluation

Given the high stakes of credit card fraud detection, evaluation metrics must align with business objectives. Accurately identifying fraudulent transactions is more critical than classifying non-fraudulent ones. Therefore:

1.Precision: Measures the proportion of correctly identified fraudulent transactions among all transactions classified as fraud. High precision reduces the risk of falsely accusing customers of fraud (false positives).

2.Recall (Sensitivity): Measures the proportion of actual fraudulent transactions that are correctly identified. High recall ensures that most fraudulent activities are detected (minimizing false negatives).

3.F1-Score: The harmonic mean of precision and recall, providing a single metric to balance the trade-off between the two.

4.ROC-AUC Curve: Visualizes the trade-off between the true positive rate and the false positive rate. A higher AUC value indicates better model performance across different thresholds.

5.Confusion Matrix: Provides a clear breakdown of true positives, true negatives, false positives, and false negatives to assess model reliability.

6.Cost Analysis: Integrate a cost-sensitive evaluation framework to account for the financial and reputational impact of false positives and false negatives, ensuring that the chosen model aligns with real-world priorities.

By incorporating these comprehensive metrics, the project aims to deliver a reliable and actionable solution for credit card fraud detection that aligns with business goals and customer satisfaction.

