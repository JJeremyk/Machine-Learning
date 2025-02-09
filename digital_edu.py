#create your individual project here!
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv('train.csv')
print(df.info())

X = df.drop('result', axis = 1)
y = df['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = KNeighborsClassifier(n_neighbours = 5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print('Percentage of coorectly predicted outcomes:', accuracy_score(y_test, y_pred)*100)
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))