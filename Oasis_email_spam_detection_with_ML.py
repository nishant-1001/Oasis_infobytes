#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("encoded-spam.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df["Unnamed: 2"].value_counts


# In[6]:


df.shape


# In[7]:


df.tail()


# In[8]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[9]:


# Load the dataset (replace 'your_dataset.csv' with the actual path to your dataset)
data = pd.read_csv('encoded-spam.csv')


# In[10]:


# Assuming your dataset has two columns: 'text' and 'label' where 'text' contains the email
#content and 'label' contains the class (spam or not spam)
X = data['v2']
y = data['v1']


# In[11]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[12]:


# Convert the text data to a bag-of-words representation
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# In[13]:


# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)


# In[14]:


# Make predictions on the test set
predictions = classifier.predict(X_test_vectorized)


# In[15]:


# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
classification_rep = classification_report(y_test, predictions)


# In[16]:


print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')


# In[17]:


import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


# In[18]:


import nltk
nltk.download('stopwords')


# In[19]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust the number of features
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# In[20]:


from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)


# In[21]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

predictions = classifier.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
classification_rep = classification_report(y_test, predictions)

print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')


# In[22]:


def preprocess_text(text):
    text = re.sub(r'\b\d+\b', '', text)  # Remove numbers
    text = re.sub(r'\W', ' ', text)      # Remove non-word characters
    text = re.sub(r'\s+', ' ', text)     # Remove extra whitespaces
    text = text.lower()                   # Convert to lowercase
    
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text


# In[23]:


new_emails = ["You win a good award", "Another email to check"]

# Preprocess the new emails
new_emails_processed = [preprocess_text(email) for email in new_emails]

# Vectorize the new emails
new_emails_vectorized = vectorizer.transform(new_emails_processed)

# Make predictions
new_predictions = classifier.predict(new_emails_vectorized)
print(f'Predictions for new emails: {new_predictions}')


# In[24]:


new_emails = ["you won award", "Another email to check"]

# Preprocess the new emails
new_emails_processed = [preprocess_text(email) for email in new_emails]

# Vectorize the new emails
new_emails_vectorized = vectorizer.transform(new_emails_processed)

# Make predictions
new_predictions = classifier.predict(new_emails_vectorized)
print(f'Predictions for new emails: {new_predictions}')


# In[25]:


new_emails = ["official mail", "Another email to check"]

# Preprocess the new emails
new_emails_processed = [preprocess_text(email) for email in new_emails]

# Vectorize the new emails
new_emails_vectorized = vectorizer.transform(new_emails_processed)

# Make predictions
new_predictions = classifier.predict(new_emails_vectorized)
print(f'Predictions for new emails: {new_predictions}')

