import pickle
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load and split data
df = pd.read_csv("data/anemia.csv")
X = df.drop('Result', axis=1)
Y = df['Result']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

# Train and evaluate models
models = {
    'Logistic Regression': LogisticRegression(random_state=20),
    'Random Forest': RandomForestClassifier(random_state=20),
    'Decision Tree': DecisionTreeClassifier(random_state=20),
    'Gaussian Naive Bayes': GaussianNB(),
    'SVM': SVC(random_state=20),
    'Gradient Boosting': GradientBoostingClassifier(random_state=20)
}

results = {}

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    results[name] = {'Accuracy': acc, 'Report': report}

# Print comparison of models
compare_models = pd.DataFrame.from_dict({name: data['Accuracy'] for name, data in results.items()}, orient='index', columns=['Accuracy'])
print(compare_models)

# Save the best model (Gradient Boosting Classifier)
best_model = models['Gradient Boosting']
with open("model.pkl", "wb") as model_file:
    pickle.dump(best_model, model_file)

# Test prediction
test_input = [[0, 12.4, 23, 32.2, 76.1]]
prediction = best_model.predict(test_input)
print(f"Test prediction for input {test_input}: {prediction}")

# Handle potential warnings
warnings.warn("Ensure the input data has valid feature names when making predictions.")
