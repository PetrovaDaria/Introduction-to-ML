from pandas import read_csv
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


perceptron_train = read_csv('perceptron-train.csv', header=None)
perceptron_test = read_csv('perceptron-test.csv', header=None)

train_data = perceptron_train.iloc[:, 1:]
train_target = perceptron_train.iloc[:, 0]
test_data = perceptron_test.iloc[:, 1:]
test_target = perceptron_test.iloc[:, 0]

perceptron = Perceptron(random_state=241)
perceptron.fit(train_data, train_target)
predictions = perceptron.predict(test_data)
accuracy = accuracy_score(test_target, predictions)
print(accuracy)

scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)
#train_target_scaled = scaler.fit_transform(train_target)
#test_target_scaled = scaler.fit_transform(test_target)

perceptron.fit(train_data_scaled, train_target)
predictions_scaled = perceptron.predict(test_data_scaled)
accuracy_scaled = accuracy_score(test_target, predictions_scaled)
print(accuracy_scaled)

print(abs(accuracy - accuracy_scaled))
