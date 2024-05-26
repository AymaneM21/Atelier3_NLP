# Lab 3: Mastering NLP Techniques with Sklearn

## Objective
The main purpose of this lab is to familiarize ourselves with NLP language models using the Sklearn library. We worked on both regression and classification tasks using different datasets to understand how various NLP techniques and machine learning models can be applied to textual data.

## Work Overview

### Part 1: Language Modeling / Regression
**Dataset:** [Short Answer Grading Dataset](https://github.com/dbbrandt/short_answer_granding_capstone_project/blob/master/data/sag/answers.csv)

1. **Preprocessing Pipeline:**
   - Tokenization
   - Stemming
   - Lemmatization
   - Stop words removal

2. **Data Encoding:**
   - Word2Vec (CBOW and Skip Gram)
   - Bag of Words (BoW)
   - TF-IDF

3. **Models Trained:**
   - Support Vector Regression (SVR)
   - Naive Bayes
   - Linear Regression
   - Decision Tree Regression

4. **Evaluation Metrics:**
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)

### Part 2: Language Modeling / Classification
**Dataset:** [Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)

1. **Preprocessing Pipeline:**
   - Tokenization
   - Stemming
   - Lemmatization
   - Stop words removal

2. **Data Encoding:**
   - Word2Vec (CBOW and Skip Gram)
   - Bag of Words (BoW)
   - TF-IDF

3. **Models Trained:**
   - Support Vector Machine (SVM)
   - Naive Bayes
   - Logistic Regression
   - AdaBoost

4. **Evaluation Metrics:**
   - Accuracy
   - F1 Score
   - Log Loss

## Summary of Results
### Regression
- **Best Model using Skip Gram:** Linear Regression
  - **MSE:** 1.0848
  - **RMSE:** 1.0415

- **Best Model using CBOW:** Linear Regression
  - **MSE:** 222.2731
  - **RMSE:** 14.9088

### Classification
- **Best Model:** SVM with Skip Gram Word2Vec
  - **Accuracy:** 0.605
  - **F1 Score:** 0.595
  - **Log Loss:** N/A for SVM

## Interpretation of Results
Throughout this lab, we explored various preprocessing, encoding, and modeling techniques in NLP. The Skip Gram Word2Vec model consistently provided better contextual understanding, leading to improved performance in both regression and classification tasks. This lab enhanced our proficiency in handling textual data, applying machine learning models, and interpreting results using standard metrics.

## Tools Used
- **Python Libraries:** Spacy, NLTK, Gensim, Sklearn, Pandas, Numpy
- **Development Platforms:** Google Colab, Kaggle

## What We Learned
- Effective preprocessing of text data is crucial for better model performance.
- Word embeddings like Word2Vec can capture context, leading to improved model accuracy and interpretability.
- Different models and metrics provide varied insights, highlighting the importance of choosing appropriate techniques based on the task.

## Conclusion
This lab provided a comprehensive understanding of NLP techniques and their application in machine learning. We successfully applied various models to real-world datasets, gaining practical experience in handling regression and classification problems using NLP.

---

**Université Abdelmalek Essaadi** 
- Faculté des Sciences et Techniques de Tanger
- Département Génie Informatique
- Master : AISD
- NLP
- Pr . ELAACHAK LOTFI
