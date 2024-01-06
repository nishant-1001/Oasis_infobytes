#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import warnings


# In[9]:


# Load your sales dataset (replace 'your_dataset.csv' with the actual path)
df = pd.read_csv('Advertising.csv')


# In[10]:


# Inspect the data
print(df.head())


# In[11]:


warnings.simplefilter(action='ignore', category=FutureWarning)
os.getcwd()


# In[12]:


df.head()


# In[13]:


df.columns


# In[14]:


df.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)


# In[15]:


df


# In[16]:


df.info()


# In[17]:


df.describe().T


# In[18]:


df.describe()


# In[19]:


df.isnull().values.any()
df.isnull().sum()


# In[20]:


sns.pairplot(df, x_vars=["TV", "Radio", "Newspaper"], y_vars="Sales", kind="reg")


# In[21]:


df.hist(bins=20)


# In[22]:


sns.lmplot(x='TV', y='Sales', data=df)
sns.lmplot(x='Radio', y='Sales', data=df)
sns.lmplot(x='Newspaper',y= 'Sales', data=df)


# In[23]:


corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmin=0, vmax=1, square=True, cmap="YlGnBu", ax=ax)
plt.show()


# In[24]:


X = df.drop('Sales', axis=1)
y = df[["Sales"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40)


# In[26]:


lin_model = sm.ols(formula="Sales ~ TV + Radio + Newspaper", data=df).fit()


# In[27]:


print(lin_model.params, "\n")


# In[28]:


print(lin_model.summary())


# In[29]:


# Evaluate the model

results = []
names = []


# In[30]:


models = [('LinearRegression', LinearRegression())]


# In[31]:


for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    result = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append(result)
    names.append(name)
    msg = "%s: %f" % (name, result)
    print(msg)


# In[32]:


new_data = pd.DataFrame({'TV': [100], 'Radio': [50], 'Newspaper': [25]})
predicted_sales = lin_model.predict(new_data)
print("Predicted Sales:", predicted_sales)


# # Make predictions on new data

# In[33]:


new_data = pd.DataFrame({'TV': [25], 'Radio': [63], 'Newspaper': [80]})
predicted_sales = lin_model.predict(new_data)
print("Predicted Sales:", predicted_sales)


# In[ ]:




