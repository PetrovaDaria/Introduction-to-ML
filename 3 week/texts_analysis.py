from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np


newsgroups = fetch_20newsgroups(data_home='./', subset='all', categories=['alt.atheism', 'sci.space'])
texts = newsgroups.data
tfidfvectorizer = TfidfVectorizer()
X = tfidfvectorizer.fit_transform(texts)
y = newsgroups.target

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, random_state=241, shuffle=True)
svc = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(estimator=svc, param_grid=grid, scoring='accuracy', cv=cv)
gs.fit(X, y)

features_names = tfidfvectorizer.get_feature_names()
results = gs.best_estimator_.coef_
row = results.getrow(0).toarray()[0].ravel()
top_ten_indicies = np.argsort(abs(row))[-10:]
print(" ".join(features_names[i] for i in top_ten_indicies))


