import csv
from urllib.parse import urlparse

import requests


# Функция для извлечения домена из URL
def get_domain_from_url(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc


# Функция для обработки CSV файла и отправки запроса
def process_csv_and_send_requests(file_path):
    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t')

        # Чтение каждой строки CSV
        for row in csvreader:
            if len(row) < 2:
                continue  # Пропускаем строки, которые не содержат двух элементов

            hash_value = row[0]
            url = row[1]

            # Извлечение домена из URL
            domain = get_domain_from_url(url)

            response = requests.get(f"https://proscena.space:8443/user_service/api/users/{hash_value}")
            if response.status_code != 200:
                print(f"Ошибка при отправке запроса для {hash_value} к {domain}: {response.status_code}")
                response = requests.post(f"https://proscena.space:8443/user_service/api/users",
                                         json={
                                             "third_party_ID": hash_value,
                                             "domain": domain,
                                             "is_integrator": False
                                         })
                print(response.json())


# Путь к вашему CSV файлу
file_path = 'trial_calls_data (2).csv'

# Вызов основной функции
process_csv_and_send_requests(file_path)
