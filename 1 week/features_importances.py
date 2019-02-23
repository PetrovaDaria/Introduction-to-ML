import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas

clf = DecisionTreeClassifier(random_state=241)
data = pandas.read_csv('titanic.csv')
frame = pandas.DataFrame(data=data, columns=['Pclass', 'Fare', 'Age', 'Sex', 'Survived'])
frame = frame[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']].dropna()
filtered = frame[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']].replace('female', 0).replace('male', 1)
Y = frame['Survived']
X = filtered.drop(('Survived'), axis=1)
print(X.columns)
clf.fit(X, Y)
print(clf.feature_importances_)

#data = pandas.read_csv('titanic.csv', index_col='PassengerId', names=['Pclass', 'Fare', 'Age', 'Sex'])


