from sklearn.svm import SVC
from pandas import read_csv


data = read_csv('svm-data.csv', header=None)
y = data[data.columns[0]]
X = data[data.columns[1:]]
svc = SVC(kernel='linear', C=100000, random_state=241)
svc.fit(X, y)
print(svc.support_)
#add 1