import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

df = pd.read_csv('train.csv')
print(df.info())

df['bdate'] = pd.to_datetime(df['bdate'], errors = 'coerce', dayfirst = True)
df['age'] = 2024 - df['bdate'].dt.year
df.drop(columns = ['bdate', 'id', 'occupation_name'], inplace = True)

df.fillna({'education_form' : 'unknown', 'city' : 'unknown' , 'occupation_time' : 'Unknown'}, inplace = True)
df.fillna(df.median(numeric_only = True), inplace = True)

data = pd.get_dummies(df, drop_first = True)

#split data into features and target
X = data.drop(columns = ['result'])
y = data['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
prinnt('Classification Report:\n', classification_report(y_test, y_pred))