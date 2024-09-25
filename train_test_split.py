#!/usr/bin/env python
# coding: utf-8

# In[26]:


from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
min_max_scaler = preprocessing.MinMaxScaler()
X = df_cleaned
y = df_cleaned.classification # Classes: 0-3
X_minmax = min_max_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_minmax, y, test_size=0.2, random_state=42)# are these good options?
print(X.columns)
print(y)


# In[ ]:




