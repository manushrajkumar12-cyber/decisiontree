🌳 Decision Tree Classification

📌 Overview

This project demonstrates the implementation of a Decision Tree Classifier for a classification problem using Python and Scikit-learn.

---

🎯 Objective

- Understand how Decision Trees work
- Build a classification model
- Visualize and evaluate model performance

---

🧠 What is a Decision Tree?

A Decision Tree is a supervised machine learning algorithm used for classification and regression.
It splits data into branches based on feature values to make predictions.

---

⚙️ Technologies Used

- Python 🐍
- Pandas
- Matplotlib
- Scikit-learn

---

🔄 Workflow

1. Load dataset
2. Preprocess data
3. Split into training and testing sets
4. Train Decision Tree model
5. Make predictions
6. Evaluate performance

---

🧪 Implementation

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Model
model = DecisionTreeClassifier(max_depth=3)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

---

📈 Output

- Model Accuracy: ~85–95% (depends on dataset)



📌 Key Concepts

- Tree structure (nodes, branches, leaves)
- Overfitting and max_depth
- Feature-based splitting

---

▶️ How to Run

pip install pandas matplotlib scikit-learn
jupyter notebook

---

👨‍💻 Author

Manush
