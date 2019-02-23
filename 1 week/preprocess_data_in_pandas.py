import pandas
import re


data = pandas.read_csv('titanic.csv', index_col='PassengerId')

# 1. Какое количество мужчин и женщин ехало на корабле?
print(data['Sex'].value_counts())

# 2. Какой части пассажиров удалось выжить?
count = data['Survived'].count()
survived_count = data['Survived'].gt(0).sum(axis=0)
print(str(round(survived_count / count * 100, 2)) + '%')

# 3. Какую долю пассажиры первого класса составляли среди всех пассажиров?
first_class_count = data[data['Pclass'] == 1]['Pclass'].count()
print(str(round(first_class_count / count * 100, 2)) + '%')

# 4. Какого возраста были пассажиры? Посчитайте среднее и медиану возраста пассажиров.
ages = data[data['Age'].notnull()]['Age']
mean = ages.mean()
print(round(mean, 2))
median = ages.median()
print(round(median, 2))

# 5. Коррелируют ли число братьев/сестер/супругов с числом родителей/детей?
# Посчитайте корреляцию Пирсона между признаками SibSp и Parch.
sibsp_parch_pearson_corr = data['SibSp'].corr(data['Parch'], method='pearson')
print(round(sibsp_parch_pearson_corr, 2))

# 6. Какое самое популярное женское имя на корабле?
# Извлеките из полного имени пассажира (колонка Name) его личное имя (First Name).
female_names = data[data['Sex'] == 'female']['Name']
first_names = {}
for name in female_names:
    if '(' in name:
        found = re.search('(?<=\()[a-zA-Z]+', name)
    else:
        found = re.search('(?<=Miss. )[a-zA-Z]+|(?<=Mrs. )[a-zA-Z]+|(?<=Mme. )[a-zA-Z]+'
                          '|(?<=Ms. )[a-zA-Z]+ |(?<=Lady. )[a-zA-Z]+', name)
    if found is not None:
        first_name = found.group(0)
        if first_name not in first_names.keys():
            first_names[first_name] = 0
        first_names[first_name] += 1
print(sorted(first_names.items(), key=lambda kv: kv[1], reverse=True))

