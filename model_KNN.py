#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

# Train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Predict using the KNN model
y_pred_knn = knn_model.predict(X_test)

# Evaluate the KNN model
print("KNN Accuracy: ", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# Overall metrics for the tuned Decision Tree model
print("Tuned Decision Tree Performance:")
print("Overall Accuracy:", metrics.accuracy_score(y_test, y_pred_knn))
print("Overall Precision:", metrics.precision_score(y_test, y_pred_knn, average='weighted'))
print("Overall Recall:", metrics.recall_score(y_test, y_pred_knn, average='weighted'))
print("Overall F1 Score:", metrics.f1_score(y_test, y_pred_knn, average='weighted'))


# Confusion Matrix for KNN
cm_knn = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - KNN')
plt.show()

