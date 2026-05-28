import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Загружаем наш объединенный каталог
try:
    df = pd.read_csv('catalog.csv')
except FileNotFoundError:
    print("Ошибка: Файл catalog.csv не найден! Сначала запустите merge_data.py")
    exit()

# 2. Создаем "Умную колонку" для поиска
# Мы склеиваем категорию, название, компанию и откуда товар в одну строку, 
# чтобы алгоритм искал по всем этим словам сразу, переведя их в нижний регистр.
df['search_text'] = (df['Категория'] + ' ' + 
                     df['Название'] + ' ' + 
                     df['Компания'] + ' ' + 
                     df['Откуда']).str.lower()

# 3. Обучаем алгоритм превращать текст в математические векторы (TF-IDF)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['search_text'].fillna(''))

def find_products(query, top_k=5):
    """
    Функция принимает запрос пользователя (query) и возвращает top_k лучших совпадений.
    """
    # Переводим запрос пользователя в нижний регистр и векторизуем
    query_vec = vectorizer.transform([query.lower()])
    
    # Считаем похожесть запроса со всеми товарами в базе (от 0.0 до 1.0)
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Берем индексы самых похожих товаров
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        score = similarities[idx]
        if score > 0.05:  # Отсеиваем совсем неподходящий мусор (порог похожести)
            item = df.iloc[idx]
            # Формируем красивую строчку с информацией о товаре
            product_info = (f"[{item['Категория']}] {item['Название']} ({item['Компания']}) | "
                            f"Цена: {item['Цена']} | {item['Остаток']}")
            results.append(product_info)
            
    if not results:
        return "В каталоге ничего не найдено по этому запросу."
        
    # Соединяем найденные товары в один текст
    return "\n".join(results)

# ==========================================
# ТЕСТОВЫЙ БЛОК (Сработает, если запустить этот файл напрямую)
# ==========================================
if __name__ == "__main__":
    print("🤖 Поисковик запущен. Напишите, что вы ищете (или 'выход' для завершения):")
    while True:
        user_query = input("Вы ищете: ")
        if user_query.lower() == 'выход':
            break
            
        print("\n--- Результаты поиска ---")
        print(find_products(user_query))
        print("-------------------------\n")