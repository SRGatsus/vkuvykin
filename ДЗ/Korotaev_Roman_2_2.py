import random

random.seed(22)
rand_float = random.random()
print("Случайное число с плавающей запятой:", rand_float)

rand_uniform = random.uniform(1, 10)
print("Случайное однородное число от 1 до 10::", rand_uniform)

rand_int = random.randint(100, 200)
print("Случайное целое число в диапазоне от 100 до 200:", rand_int)

colors = ['red', 'blue', 'green', 'yellow']
rand_choice = random.choice(colors)
print("Случайный выбор из цветов:", rand_choice)

name = "Roman"
surname = "Korotaev"
characters = list(name) + list(surname) + [str(i) for i in range(10)]
password = ''.join(random.choice(characters) for _ in range(10))

print("Сгенерированный пароль:", password)