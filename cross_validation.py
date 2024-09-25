#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import cross_val_score

# Cross-validation for Decision Tree
dt_cv_scores = cross_val_score(dt_model, X_train, y_train, cv=5)
print(f"Decision Tree Cross-Validation Accuracy: {np.mean(dt_cv_scores)}")

# Cross-validation for KNN
knn_cv_scores = cross_val_score(knn_model, X_train, y_train, cv=5)
print(f"KNN Cross-Validation Accuracy: {np.mean(knn_cv_scores)}")

