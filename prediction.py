
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
blooddata = pd.read_csv("C:/Courses/bio project/Blood Glucose Dataset_final.csv")
blooddata.head()
X = blooddata.drop('Class', axis=1)
X = StandardScaler().fit_transform(X)
y = blooddata['Class']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40,random_state=0)
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)



from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

print(accuracy_score(y_test,y_pred))



with plt.style.context(('ggplot')):
    fig, ax = plt.subplots()
    ax.scatter(y_pred, y_test, c='red', edgecolors='k')
    plt.xlabel('Predicted')
    plt.ylabel('Measured')
    plt.title('Prediction')
    plt.show()
