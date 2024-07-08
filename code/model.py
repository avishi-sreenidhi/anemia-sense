import pickle 
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split

# splitting into train and test data
df = pd.read_csv("data/anemia.csv")
X =  df.drop('Result',axis=1)
Y = df['Result']
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=20)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

logistic_regression = LogisticRegression()
logistic_regression.fit(x_train,y_train)
y_pred = logistic_regression.predict(x_test)

acc_lr = accuracy_score(y_test,y_pred)
c_lr= classification_report(y_test,y_pred)

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier()
random_forest.fit(x_train,y_train)
y_pred= random_forest.predict(x_test)

acc_rf = accuracy_score(y_test,y_pred)
c_rf= classification_report(y_test,y_pred)


from sklearn.tree import DecisionTreeClassifier

decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(x_train,y_train)
y_pred = decision_tree_model.predict(x_test)

acc_dt = accuracy_score(y_test,y_pred)
c_dt= classification_report(y_test,y_pred)

from sklearn.naive_bayes import GaussianNB

NB = GaussianNB()
NB.fit(x_train,y_train)
y_pred = NB.predict(x_test)

acc_nb = accuracy_score(y_test,y_pred)
c_nb= classification_report(y_test,y_pred)

from sklearn.svm import SVC

support_vector = SVC()
support_vector.fit(x_train,y_train)
y_pred = support_vector.predict(x_test)

acc_svc = accuracy_score(y_test,y_pred)
c_svc= classification_report(y_test,y_pred)

from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier()
GBC.fit(x_train,y_train)
y_pred = GBC.predict(x_test)

acc_gbc = accuracy_score(y_test,y_pred)
c_gbc= classification_report(y_test,y_pred)

# Testing model with multiple evaluation metrics

compareModels = pd.DataFrame({'Model' : ['Logistic Regression','Random Forest model', 'Decision Tree', 'Gaussian Naive Bayes', 'SVM', 'GBC'],'Score':[acc_lr,acc_rf,acc_dt,acc_nb,acc_svc,acc_gbc]})
print(compareModels)

#saving the desired model and testing out the model

pickle.dump(GBC,open("model.pkl","wb"))
prediciton = GBC.predict([[0,12.4,23,32.2,76.1]])
print(prediciton)
warnings.warn("Model might not have valid feature names for predictions")

