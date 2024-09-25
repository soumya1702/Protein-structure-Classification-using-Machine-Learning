#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Train the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predict using the Decision Tree model
y_pred_dt = dt_model.predict(X_test)

# Evaluate the Decision Tree model
print("Decision Tree Accuracy: ", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# Overall metrics for the tuned Decision Tree model
print("Tuned Decision Tree Performance:")
print("Overall Accuracy:", metrics.accuracy_score(y_test, y_pred_dt))
print("Overall Precision:", metrics.precision_score(y_test, y_pred_dt, average='weighted'))
print("Overall Recall:", metrics.recall_score(y_test, y_pred_dt, average='weighted'))
print("Overall F1 Score:", metrics.f1_score(y_test, y_pred_dt, average='weighted'))


# Confusion Matrix for Decision Tree
cm_dt = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Decision Tree')
plt.show()

