from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from pandas import read_csv
from numpy import mean


def get_neighbors_amount_and_result(features, cls):
    amount = 0
    result = 0
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for k in range(1, 51):
        classifier = KNeighborsClassifier(n_neighbors=k)
        current = mean(cross_val_score(estimator=classifier, cv=kf, X=features, y=cls))
        if current > result:
            result = current
            amount = k
    return amount, result


data = read_csv('wine.data', header=None)
cls = data.iloc[:, 0]
features = data.iloc[:, 1:]

print('Not scaled features')
amount, result = get_neighbors_amount_and_result(features, cls)
print(amount)
print(result)

print('Scaled features')
features = scale(features)
amount, result = get_neighbors_amount_and_result(features, cls)
print(amount)
print(result)


