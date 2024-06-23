# Text Classification for Customer Complaint Topics

## Overview
This project aims to classify customer complaints into different topics using machine learning techniques. The dataset contains text data related to customer complaints, and the goal is to predict the category or topic of each complaint. Various models such as Logistic Regression, Decision Tree, and Random Forest have been evaluated for their effectiveness in classifying the complaints accurately.

## Dataset Description
The dataset consists of customer complaints categorized into several topics:
- Bank account services
- Credit Card/Prepaid Card
- Mortgages/loans
- Theft/Dispute reporting
- Others

Each complaint is associated with text data that describes the issue faced by the customer.

## Methodology
### Data Preprocessing
- Text cleaning: Tokenization, removing stopwords, punctuation, and stemming/lemmatization.
- Vectorization: Transforming text data into numerical features using TF-IDF vectorization.

### Model Building and Evaluation
1. **Logistic Regression**
   - Model trained using `LogisticRegression` from scikit-learn.
   - Evaluation metrics: Accuracy, Confusion Matrix, Classification Report.
   
2. **Decision Tree**
   - Model trained using `DecisionTreeClassifier` with and without hyperparameter tuning.
   - Hyperparameters tuned: `max_depth`, `min_samples_leaf`, `criterion`.
   - Evaluation metrics: Accuracy, Confusion Matrix, Classification Report.
   
3. **Random Forest**
   - Model trained using `RandomForestClassifier` with and without hyperparameter tuning.
   - Hyperparameters tuned: `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`.
   - Evaluation metrics: Accuracy, Confusion Matrix, Classification Report.
   
4. **Naive Bayes (Optional)**
   - Model trained using `MultinomialNB`.
   - Evaluation metrics: Accuracy, Confusion Matrix, Classification Report.
   - Hyperparameter tuning: Alpha parameter for Laplace smoothing.

### Model Evaluation
Several machine learning models were evaluated for their effectiveness in classifying complaints:
1. **Logistic Regression**
   - **Training Accuracy:** 92.92%
   - **Test Accuracy:** 88.37%
   
2. **Decision Tree**
   - **Training Accuracy:** 73.43%
   - **Test Accuracy:** 70.23%
   - Hyperparameter tuning improved performance:
     - **Training Accuracy (Tuned):** 89.76%
     - **Test Accuracy (Tuned):** 76.84%
   
3. **Random Forest**
   - **Training Accuracy:** 67.69%
   - **Test Accuracy:** 67.16%
   - Hyperparameter tuning did not significantly improve performance.

4. **Naive Bayes**
   - **Training Accuracy:** 68.05%
   - **Test Accuracy:** 67.38%
   - Naive Bayes with hyperparameter tuning:
     - **Training Accuracy (Tuned):** 86.77%
     - **Test Accuracy (Tuned):** 76.01%

### Model Selection
- **Logistic Regression** performed the best overall with the highest test accuracy of 88.37%, indicating robust performance in classifying customer complaints.
- **Decision Tree** showed improvement after hyperparameter tuning but did not match Logistic Regression's performance.
- **Random Forest** and **Naive Bayes** models demonstrated lower accuracies compared to Logistic Regression and Decision Tree.

## Conclusion
Based on the evaluation results, **Logistic Regression** is recommended for predicting customer complaint topics due to its superior performance in both training and test sets. Further optimizations could involve:
- Exploring additional text preprocessing techniques.
- Collecting more diverse complaint data to enhance model generalization.
- Experimenting with ensemble methods or deep learning architectures for potentially better performance.

