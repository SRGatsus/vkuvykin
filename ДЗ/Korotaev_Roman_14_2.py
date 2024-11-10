import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

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

# Преобразование категориальных признаков в числовые
titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)

# Разделение данных на обучающие и тестовые
X = titanic_data.drop(columns=['Survived'])
y = titanic_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Функция для построения и оценки дерева классификации
def build_and_evaluate_tree(criterion):

    clf = DecisionTreeClassifier(criterion=criterion, random_state=42)
    clf.fit(X_train, y_train)

    # Оценка модели
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Важность переменных
    feature_importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)

    # Визуализация дерева
    plt.figure(figsize=(20, 10))
    plot_tree(clf, filled=True, feature_names=X.columns, class_names=['Not Survived', 'Survived'], rounded=True)
    plt.title(f'Decision Tree using {criterion.capitalize()}')
    plt.show()

    return train_accuracy, test_accuracy, feature_importances

# Построение и оценка дерева с критерием Джини
train_accuracy_gini, test_accuracy_gini, feature_importances_gini = build_and_evaluate_tree('gini')

# Построение и оценка дерева с критерием энтропии
train_accuracy_entropy, test_accuracy_entropy, feature_importances_entropy = build_and_evaluate_tree('entropy')

# Вывод результатов
print("Критерий Джини")
print(f"Средняя точность по обучающим данным: {train_accuracy_gini:.4f}")
print(f"Средняя точность по тестовым данным: {test_accuracy_gini:.4f}")
print("Важность переменных:")
print(feature_importances_gini)

print("\nКритерий Энтропии")
print(f"Средняя точность по обучающим данным: {train_accuracy_entropy:.4f}")
print(f"Средняя точность по тестовым данным: {test_accuracy_entropy:.4f}")
print("Важность переменных:")
print(feature_importances_entropy)
