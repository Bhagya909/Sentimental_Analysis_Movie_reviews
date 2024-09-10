# Sentimental_Analysis_Movie_reviews
# Project Overview
This project focuses on performing Sentiment Analysis on a dataset of movie reviews using the Naive Bayes algorithm. The goal is to classify each review as either positive or negative based on its content, and the model achieves an accuracy of 84%.

Sentiment analysis is a key application in Natural Language Processing (NLP) that helps to extract emotions, opinions, and sentiments from text. In this project, we use machine learning to determine whether movie reviews have a positive or negative sentiment.

# Features
**Machine Learning Algorithm:** Naive Bayes (Multinomial or Bernoulli)

**Accuracy:** 84%

**Dataset:** https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

**Preprocessing:** Tokenization, stop word removal, stemming/lemmatization (optional)

**Evaluation Metrics:** Accuracy, Precision, Recall, F1 Score

**Tools/Libraries:**
Python
Scikit-learn
NLTK (Natural Language Toolkit) or similar for text preprocessing
Pandas, NumPy for data handling
Dataset
The dataset contains movie reviews labeled as either positive or negative. Each review is a free-text entry, and the labels are used to train the Naive Bayes model. You can use public datasets like the IMDB movie reviews dataset or any other labeled review data.

# Project Workflow
**1. Data Preprocessing**
Before applying the Naive Bayes classifier, the raw text data needs to be preprocessed:

Tokenization: Splitting the text into individual words or tokens.

Stop word removal: Removing common words like "and", "is", "the" that do not contribute much to sentiment analysis.

Stemming/Lemmatization (optional): Reducing words to their root forms.

Vectorization: Converting text into a numerical format using techniques like TF-IDF or Count Vectorization.

**2. Model Training**
The Naive Bayes algorithm is applied after vectorizing the text data.
The data is split into training and testing sets (e.g., 80% training, 20% testing).
The model learns the relationship between words and sentiment from the training set.

**3. Model Evaluation**
The model's performance is measured using metrics such as accuracy (achieved 84%), precision, recall, and F1 score.
Accuracy refers to the percentage of reviews that are correctly classified as either positive or negative.

![Screenshot 2024-09-09 230836](https://github.com/user-attachments/assets/9be57e2a-1458-47b2-9118-29d349d15f5f)
