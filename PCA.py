
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


data = pd.read_csv('C:/Courses/bio project/Blood Glucose Dataset_final.csv')

X = data.drop('Class', axis=1)
print(X.shape)
X = X.iloc[:, 30:84]
print(X.shape)
y = data['Class']
X = StandardScaler().fit_transform(X)
pca = PCA(n_components=.95)
X = pca.fit_transform(X)
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40,random_state=0)
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(y_pred.shape)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(y_test,y_pred.T)

with plt.style.context(('ggplot')):
    fig, ax = plt.subplots()
    ax.scatter(y_pred, y_test, c='red', edgecolors='k')
    plt.xlabel('Predicted')
    plt.ylabel('Measured')
    plt.title('Prediction')
    plt.show()






