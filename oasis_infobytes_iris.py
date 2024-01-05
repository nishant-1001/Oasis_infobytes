#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


df = pd.read_csv("iris.csv")


# In[3]:


df.head()


# # Preprocessing 

# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df = df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","Species"]]


# In[7]:


df.head()


# In[8]:


df["Species"].value_counts()


# In[9]:


sns.pairplot(df,hue = 'Species' )


# # Slicing and Indexing:

# In[10]:


df = df.values
X = df[:,0:4]
y = df[:,4]


# # Splitting the data set into training and testing:

# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


# # Model1:Support Vector Machine

# In[13]:


from sklearn.svm import SVC
Model1_SVC = SVC()
Model1_SVC.fit(X_train,y_train)


# In[14]:


pred_y1 = Model1_SVC.predict(X_test)


# In[15]:


from sklearn.metrics import accuracy_score


# In[16]:


print(accuracy_score(y_test,pred_y1)*100)


# # Model2:Logistic Regression

# In[17]:


from sklearn.linear_model import LogisticRegression
Model2_LR = LogisticRegression()
Model2_LR.fit(X_train,y_train)


# In[18]:


pred_y2 = Model2_LR.predict(X_test)


# In[19]:


from sklearn.metrics import accuracy_score


# In[20]:


print(accuracy_score(y_test,Model2_LR.predict(X_test))*100)


# # Model3: Decision Tree Classifier

# In[21]:


from sklearn.tree import DecisionTreeClassifier


# In[22]:


Model3_dc = DecisionTreeClassifier(random_state=0)


# In[23]:


Model3_dc =  DecisionTreeClassifier()
Model3_dc.fit(X_train,y_train)


# In[24]:


pred_y3 = Model3_dc.predict(X_test)


# In[25]:


from sklearn.metrics import accuracy_score


# In[26]:


print(accuracy_score(y_test,Model3_dc.predict(X_test))*100)


# In[27]:


from sklearn.metrics import classification_report


# In[28]:


print(classification_report(y_test,pred_y3))


# In[29]:


print(classification_report(y_test,pred_y2))


# In[30]:


print(classification_report(y_test,pred_y1))


# # Verification of Model:

# In[31]:


X_new = np.array([[1,2,0.3,0.9],[0.6,1.2,2.0,3.1],[5.2,4.6,2.3,3.5]])


# In[32]:


Model1_SVC.fit(X_train,y_train)


# In[33]:


prediction = Model1_SVC.predict(X_new) 


# In[34]:


print("Prediction of Species:{}".format(prediction))

