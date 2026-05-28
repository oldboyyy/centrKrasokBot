from bs4 import BeautifulSoup
import requests
import pandas as pd

link = "https://centr-krasok.kz/catalog/shtukaturki/"
response = requests.get(link)
html_content = response.text
soup = BeautifulSoup(html_content, 'html.parser')

products_data = []
items = soup.find_all('div', class_='product-item')

for item in items:
    title_tag = item.find('h3', class_='card_item_title')
    title = title_tag.text.strip() if title_tag else "Нет названия"

    stock_tag = item.find('div', class_='card-item-stores-amount-store')
    stock = " ".join(stock_tag.text.split()) if stock_tag else "Нет данных"

    price_tag = item.find('strong', class_='card_item_price_current')
    price = price_tag.text.strip() if price_tag else "Нет цены"

    sku_tag = item.find('span', class_='card_item_property_value')
    sku = sku_tag.text.strip() if sku_tag else "Нет артикула"

    labels = item.find_all('span', class_='card_item_label_no_mobile')
    brand = labels[0].text.strip() if len(labels) > 0 else "Не указан"
    country = labels[1].text.strip() if len(labels) > 1 else "Не указана"
    volume = labels[2].text.strip() if len(labels) > 2 else "Не указан"

    products_data.append({
        'Название': title,
        'Остаток': stock,
        'Цена': price,
        'Артикул': sku,
        'Компания': brand,
        'Откуда': country,
        'Объем': volume
    })

df = pd.DataFrame(products_data)
df.to_csv('shtukaturki.csv', index=False)
print("Данные успешно сохранены!")