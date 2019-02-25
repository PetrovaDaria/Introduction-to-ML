from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from numpy import linspace, mean


boston_data = load_boston()
X = scale(boston_data.data)
y = scale(boston_data.target)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
max = -1
p = 0
for i in linspace(1, 10, 200):
    regressor = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=i)
    current = mean(cross_val_score(estimator=regressor, cv=kf, scoring='neg_mean_squared_error', X=X, y=y))
    if current > max:
        p = i
        max = current
print(p)
print(max)