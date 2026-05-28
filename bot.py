"""
Telegram AI-ассистент для «Центр Красок #1»
Использует бесплатную открытую модель Llama 3 (через Groq API) и RAG.
"""

import logging
import os
from collections import defaultdict

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from keep_alive import keep_alive  

from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from openai import OpenAI

# Убедитесь, что файлы называются knowledge_base.py и search_engine.py
from knowledge import SYSTEM_PROMPT
from search import find_products
keep_alive()

load_dotenv()

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.getLogger().setLevel(logging.INFO),
)
logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Используем самую умную и мощную бесплатную модель Llama от Meta
MODEL = "llama-3.3-70b-versatile" 
MAX_HISTORY_TURNS = 10   

# ─── State ───────────────────────────────────────────────────────────────────
conversation_history: dict[int, list[dict]] = defaultdict(list)

# ─── Groq (OpenAI Client) Setup ──────────────────────────────────────────────
# Платформа Groq полностью совместима с библиотекой OpenAI
ai_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)


def get_ai_response(user_id: int, user_message: str) -> str:
    """Отправляет сообщение в Groq вместе с найденными товарами из CSV."""
    history = conversation_history[user_id]
    history.append({"role": "user", "content": user_message})

    if len(history) > MAX_HISTORY_TURNS * 2:
        history = history[-(MAX_HISTORY_TURNS * 2):]
        conversation_history[user_id] = history

    # 1. Ищем товары в catalog.csv по тексту сообщения
    search_results = find_products(user_message, top_k=5)
    
    # 2. Формируем финальный промпт для нейросети
    dynamic_system_prompt = f"{SYSTEM_PROMPT}\n\n=== РЕЗУЛЬТАТЫ ПОИСКА ПО КАТАЛОГУ (CSV) ===\n{search_results}"

    # 3. Собираем сообщения (системный промпт всегда первый)
    messages = [{"role": "system", "content": dynamic_system_prompt}] + history

    try:
        response = ai_client.chat.completions.create(
            model=MODEL,
            temperature=0.3, # Меньше фантазий, больше точности
            messages=messages,
        )

        assistant_message = response.choices[0].message.content
        history.append({"role": "assistant", "content": assistant_message})
        return assistant_message

    except Exception as e:
        logger.error("Groq API error: %s", e)
        return "Извините, произошла техническая ошибка. Пожалуйста, позвоните нам: +7 (778) 061-50-00."


# ─── Handlers ────────────────────────────────────────────────────────────────

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    conversation_history[user_id].clear()
    welcome_text = (
        "Здравствуйте! Я AI-ассистент магазина «Центр Красок #1». 🎨\n\n"
        "Я могу помочь вам с выбором краски, подсказать цены, остатки на складе, "
        "а также информацию о доставке и адресах магазинов.\n\n"
        "Напишите, что вы ищете (например: 'Нужна белая краска для потолка' или 'Сколько стоит грунтовка Milq?')"
    )
    await update.message.reply_text(welcome_text)

async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "Я могу ответить на вопросы о:\n"
        "- Наличии и ценах на товары (краски, обои, инструменты и др.)\n"
        "- Адресах и контактах\n"
        "- Условиях доставки и оплаты\n\n"
        "Просто напишите ваш вопрос обычным текстом!\n"
        "Если хотите начать диалог заново, нажмите /reset."
    )
    await update.message.reply_text(help_text)

async def reset_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    conversation_history[user_id].clear()
    await update.message.reply_text("История диалога очищена. Чем я могу помочь вам сейчас?")

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_message = update.message.text.strip()

    if not user_message:
        return

    logger.info("User %s: %s", user_id, user_message[:80])

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING,
    )

    response = get_ai_response(user_id, user_message)

    logger.info("Bot → User %s: %s", user_id, response[:80])
    await update.message.reply_text(response)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Update %s caused error %s", update, context.error)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN не задан в .env")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY не задан в .env")

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("reset", reset_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    app.add_error_handler(error_handler)

    logger.info("Бот запущен на серверах Groq! Нажмите Ctrl+C для остановки.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()