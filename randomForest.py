import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.preprocessing import StandardScaler
blooddata = pd.read_csv("C:/Courses/bio project/Blood Glucose Dataset_final.csv")
blooddata.head()
X = blooddata.drop('Class', axis=1)
X = StandardScaler().fit_transform(X)
y = blooddata['Class']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40,random_state=0)
#print(np.shape(X_train))

# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
sfm = SelectFromModel(clf, threshold=0.01)
sfm.fit(X_train, y_train)
X_important_train = sfm.transform(X_train)
X_important_test = sfm.transform(X_test)
#print(np.shape(X_important_t))

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_important_train, y_train)
y_pred = svclassifier.predict(X_important_test)
#clf_important = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
#clf_important.fit(X_important_train, y_train)
# Train the classifier
#clf.fit(X_train, y_train)
#y_pred = clf.predict(X_test)+

print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

with plt.style.context(('ggplot')):
    fig, ax = plt.subplots()
    ax.scatter(y_pred, y_test, c='red', edgecolors='k')
    plt.xlabel('Predicted_class')
    plt.ylabel('Actual_class')
    plt.title('Actual vs Predicted for method 1')
    plt.show()
print(confusion_matrix(y_test,y_pred))