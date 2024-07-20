# CODTECH-DS-Task1
# NLP-Task
## Customer Review Sentiment Analysis Project

## Overview

This project involves the development of a machine learning model for sentiment analysis on customer reviews, focusing primarily on drug reviews. The goal is to classify these reviews into different sentiment categories such as positive, negative, or neutral. The project encompasses data preprocessing, model training, and deployment of the model using a web application built with Streamlit.

## Data Preprocessing
Data Collection: The dataset consists of customer reviews related to various medications. Each review includes text data that reflects the customer’s sentiment towards the medication.

Tokenization: Tokenization is the process of splitting the text into individual words or tokens. This was achieved using SpaCy, a powerful NLP library in Python.

Lemmatization: Lemmatization was employed to reduce words to their base or root form. This step helps in normalizing the text, making it easier for the model to learn patterns.

Stop Words Removal: Common words that do not contribute significantly to the sentiment of the text, such as "and", "the", "is", were removed. This helps in reducing noise in the data.

Cleaning the Text: Additional cleaning steps included removing HTML tags, and non-alphabetic characters, and converting text to lowercase. These steps ensure that the text is in a standardized format for the model to process.

Vectorization: The text data was converted into a numerical format using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. This technique helps in representing the importance of words in the text, making it suitable for machine learning algorithms.

## Model Training
Model Selection: Various machine learning models were considered, including Logistic Regression, Support Vector Machines (SVM), and Random Forests. After experimentation, the Logistic Regression model was chosen due to its performance and interpretability.

Training: The model was trained on a labeled dataset where each review was tagged with its corresponding sentiment. The training process involved splitting the data into training and testing sets to evaluate the model’s performance.

Evaluation: The model’s performance was evaluated using metrics such as accuracy, precision, recall, and F1-score. An accuracy of 84% was achieved, indicating that the model is fairly accurate in predicting the sentiment of customer reviews.

## Deployment with Streamlight
Web Application Development: A web application was developed using Streamlit, a popular framework for building interactive web apps with Python. This application allows users to input a customer review and get an immediate sentiment analysis result.

Model Integration: The trained model was integrated into the Streamlit app. The model was loaded using Joblib, a library for serializing Python objects, ensuring that the model can be efficiently loaded and used in the app.

User Interface: The app features a simple and intuitive interface where users can enter a review in a text box and click a button to analyze the sentiment. The result is displayed on the screen, showing whether the sentiment is positive, negative, or neutral.

## Challenges and Solutions
Imbalanced Data: One challenge encountered was the imbalance in the sentiment categories, with more positive reviews than negative or neutral ones. This was addressed by experimenting with different sampling techniques and ensuring the model was trained on a balanced dataset.

Text Preprocessing: Handling various forms of text data, including slang, abbreviations, and typos, was another challenge. Comprehensive preprocessing steps and the use of robust NLP techniques like lemmatization helped in mitigating these issues.

## Conclusion
This project successfully demonstrates the application of machine learning for sentiment analysis on customer reviews. By leveraging powerful NLP techniques and machine learning algorithms, we developed a model that accurately predicts the sentiment of reviews. The deployment of this model using Streamlit makes it accessible and easy to use, providing real-time sentiment analysis for customer reviews. This project showcases the potential of sentiment analysis in understanding customer opinions and improving product and service offerings based on feedback.
