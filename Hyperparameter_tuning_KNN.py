#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning for KNN
param_grid = {'n_neighbors': range(1, 20)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best KNN model
best_knn = grid_search.best_estimator_
y_pred_best_knn = best_knn.predict(X_test)

print("Tuned KNN Accuracy: ", accuracy_score(y_test, y_pred_best_knn))
print(classification_report(y_test, y_pred_best_knn))

# Confusion Matrix for the Tuned KNN
cm_best_knn = confusion_matrix(y_test, y_pred_best_knn)
sns.heatmap(cm_best_knn, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Tuned KNN')
plt.show()

