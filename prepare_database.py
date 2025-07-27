import os
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import config

def main():
    """
    Основная функция для подготовки и сохранения векторной базы данных.
    """
    # --- 1. Загрузка документов ---
    knowledge_base_dir = "knowledge_base"
    documents = []
    for filename in os.listdir(knowledge_base_dir):
        if filename.endswith(".md"):
            filepath = os.path.join(knowledge_base_dir, filename)
            loader = UnstructuredMarkdownLoader(filepath)
            documents.extend(loader.load())

    if not documents:
        print("Не найдены документы в папке knowledge_base. Завершение работы.")
        return

    print(f"Загружено {len(documents)} фрагментов из документов.")

    # --- 2. Разбиение текста на фрагменты (chunks) ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Размер фрагмента в символах
        chunk_overlap=200,  # Перекрытие между фрагментами
        length_function=len
    )
    all_splits = text_splitter.split_documents(documents)
    print(f"Документы разделены на {len(all_splits)} фрагментов.")

    # --- 3. Создание векторных представлений и сохранение в ChromaDB ---
    # Инициализируем модель для создания эмбеддингов
    # Важно: мы используем тот же API, что и для генерации,
    # но LangChain будет вызывать эндпоинт для эмбеддингов.
    embeddings = OpenAIEmbeddings(
        openai_api_key=config.API_KEY,
        base_url=config.BASE_URL
    )

    # Указываем папку для хранения локальной векторной базы
    persist_directory = 'db'

    # Создаем и сохраняем векторную базу данных
    # Chroma.from_documents создаст эмбеддинги для всех фрагментов и сохранит их
    print("Создание и сохранение векторной базы данных... Это может занять некоторое время.")
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    print(f"Векторная база данных успешно создана и сохранена в папке '{persist_directory}'.")
    print("Этап подготовки базы знаний завершен.")


if __name__ == "__main__":
    main()