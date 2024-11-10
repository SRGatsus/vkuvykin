import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Загрузка данных
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic_data = pd.read_csv(url)

# Просмотр первых строк данных
print(titanic_data.head())

# Предварительная обработка данных
# Заполнение пропущенных значений
titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].median())
titanic_data['Embarked'] = titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0])
titanic_data['Fare'] = titanic_data['Fare'].fillna(titanic_data['Fare'].median())

# Удаление столбца 'Cabin' из-за большого количества пропущенных значений
titanic_data = titanic_data.drop(columns=['Cabin', 'Name', 'Ticket'])

# Преобразование категориальных признаков в числовые с использованием One-Hot Encoding
titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)

# Масштабирование числовых признаков с использованием MinMaxScaler
scaler = MinMaxScaler()
numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']

titanic_data[numerical_features] = scaler.fit_transform(titanic_data[numerical_features])

# Просмотр первых строк преобразованных данных
print(titanic_data.head())