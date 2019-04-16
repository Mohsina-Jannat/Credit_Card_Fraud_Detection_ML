#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

#Reading the CSV file
df = pd.read_csv('./Downloads/CreditData/creditcard.csv')
print(df)


# In[2]:


print(df.describe())


# In[3]:


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns


# In[4]:


features = ['Amount'] + ['V%d' % number for number in range(1, 29)]
target = 'Class'
X = df[features]
y = df[target]


# In[5]:


model = LogisticRegression()

#Splitting the train test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

#Normalizing the data

#X_train = normalize(X_train)
#X_test = normalize(X_test)

#fitting the model
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(classification_report(y_test, pred))


# In[ ]:




