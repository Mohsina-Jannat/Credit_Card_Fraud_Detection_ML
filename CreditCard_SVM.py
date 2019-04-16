#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[34]:


df = pd.read_csv('./Downloads/CreditData/creditcard.csv')


# In[25]:


df.shape


# In[26]:


df.head()


# In[27]:


features = ['Amount'] + ['V%d' % number for number in range(1, 29)]
target = 'Class'
X = df[features]
y = df[target]


# In[28]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


# In[29]:


from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train) 

y_pred = svclassifier.predict(X_test)
# In[30]:


y_pred = svclassifier.predict(X_test)


# In[31]:


from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))


# In[ ]:




