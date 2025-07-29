import json
import re
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict

import config

class Ability(BaseModel):
    name: str = Field(description="Название способности")
    cell_type: str = Field(description="Тип Ячейки ('Нулевая', 'Малая (I)', 'Значительная (II)', 'Предельная (III)')")
    cell_cost: int = Field(description="Стоимость способности в ячейках")
    description: str = Field(description="Описание эффекта способности")
    tags: Dict[str, str] = Field(description="Теги способности и их ранги")

class Contract(BaseModel):
    contract_name: str = Field(description="Название Контракта")
    creature_name: str = Field(description="Имя/Название Существа")
    creature_rank: str = Field(description="Ранг Существа (F, E, D, C, B, A, S, SS, SSS)")
    creature_spectrum: str = Field(description="Спектр/Тематика существа")
    creature_description: str = Field(description="Описание внешности и характера существа")
    gift: str = Field(description="Дар (пассивный эффект), который получает владелец контракта")
    sync_level: int = Field(description="Уровень Синхронизации (от 0 до 100)", default=0)
    unity_stage: str = Field(description="Ступень Единения", default="Ступень I - Активация")
    abilities: List[Ability] = Field(description="Список способностей контракта")

class Assistant:
    """
    Класс ИИ-Ассистента для проекта SDP_AI.
    Использует RAG для ответов на основе базы знаний.
    """
    def __init__(self, model_id):
        """
        Инициализирует ассистента, загружая векторную базу и настраивая LLM.
        """
        # Инициализация LLM
        self.llm = ChatOpenAI(
            model=model_id,
            openai_api_key=config.API_KEY,
            base_url=config.BASE_URL,
            temperature=0.7  # Температура для более креативных ответов
        )

        # Загрузка существующей векторной базы данных
        embeddings = OpenAIEmbeddings(
            openai_api_key=config.API_KEY,
            base_url=config.BASE_URL
        )
        vectorstore = Chroma(
            persist_directory="db", 
            embedding_function=embeddings
        )
        self.retriever = vectorstore.as_retriever()

        # Подготовка шаблонов промптов
        self._prepare_prompts()

    def _prepare_prompts(self):
        """
        Готовит и компилирует шаблоны промптов для разных задач.
        """
        # Загрузка новых промптов из файлов
        with open('sdp_ai/new_creative_prompt.md', 'r', encoding='utf-8') as f:
            creative_idea_template = f.read()
        self.creative_idea_prompt = ChatPromptTemplate.from_template(creative_idea_template)

        with open('sdp_ai/new_json_formatter_prompt.md', 'r', encoding='utf-8') as f:
            json_formatter_template = f.read()
        self.json_formatter_prompt = ChatPromptTemplate.from_template(json_formatter_template)

        # Старые промпты (валидация и общая генерация идей) пока оставляем без изменений
        # Промпт для валидации анкеты (версия 2.0 для ГМ)
        validation_template = """
        Ты — опытный и очень внимательный к деталям Гейм-Мастер (ГМ). Твоя задача — провести глубокий анализ анкеты персонажа, уделяя особое внимание балансу сил.
        Ты должен действовать как эксперт по игровой системе.

        ### Ключевые правила для анализа баланса:

        **1. Уровни силы Проводников по Рангам:**
        - **Ранг A (уровень анкеты):** Оперирует сложными концепциями в масштабе района города. Может создавать масштабные эффекты, меняющие поле боя (зоны с измененной гравитацией, массовые ментальные внушения).
        - **Ранг B:** Влияние на квартал, пробивает тяжелую броню.
        - **Ранг C:** Влияние на здание/улицу, пробивает легкую броню.

        **2. Бюджеты Ячеек и Стоимость Тегов:**
        | Тип Ячейки | Бюджет Мощи | Макс. Ранг Тега |
        |---|---|---|
        | Малая (I) | 10 очков | C |
        | Значительная (II) | 15 очков | A |
        | Предельная (III) | 30 очков | SSS |

        | Ранг Тега | Стоимость |
        |---|---|
        | F: 1 | E: 2 | D: 3 | C: 5 | B: 8 | A: 10 | S: 20 |

        ---

        **Контекст из базы знаний (лор и общие правила):**
        {context}

        **Анкета для анализа:**
        {question}

        ---

        **Твоя задача:**

        1.  **Анализ Биографии и Характера:** Кратко оцени, насколько логична история и характер, и как они вписываются в мир. Есть ли потенциал для отыгрыша?
        2.  **Анализ Синергии:** Оцени, как сочетаются архетипы, атрибуты, контракты и синки. Создают ли они целостный и интересный стиль игры?
        3.  **Глубокий Анализ Баланса (самый важный пункт):**
            *   **Способности:** Для каждой способности сравни ее описанный эффект с ее тегами и рангом Проводника. Соответствует ли мощь способности (например, "скручивая и разрывая все") ее бюджету и уровню силы Проводника ранга А? Если способность кажется слишком сильной или слабой для своей стоимости, укажи на это.
            *   **Синки:** Оцени мощь эффектов Синки. Соответствует ли эффект "Серьги Сатурна" рангу D? Не слишком ли силен эффект "Бутыли Блэкмора" для ранга C?
            *   **Манифестация и Доминион:** Внимательно изучи описание Предельной Техники. Соответствует ли ее разрушительная мощь ("вырывая здания с фундаментом") тегам S-ранга и общей концепции мира? Дай оценку ее балансу.
        4.  **Рекомендации и Вердикт:** Дай конкретные советы по улучшению баланса и общей концепции. В конце вынеси четкий вердикт: что хорошо, а что требует обязательной доработки для сохранения баланса в игре.

        Структурируй свой ответ по этим четырем пунктам. Будь конструктивен и точен в своих оценках.
        """
        self.validation_prompt = ChatPromptTemplate.from_template(validation_template)

        # Промпт для генерации идей (версия 2.0)
        idea_template = """
        Ты — креативный Гейм-Мастер (ГМ), создающий полноценные концепции персонажей.
        Твоя задача — сгенерировать ОДНУ детально проработанную идею для анкеты на основе запроса игрока.
        Ты должен строго следовать правилам создания способностей.

        ### Ключевые правила для генерации:

        **1. Бюджеты Ячеек и Стоимость Тегов:**
        | Тип Ячейки | Бюджет Мощи | Макс. Ранг Тега |
        |---|---|---|
        | Малая (I) | 10 очков | C |
        | Значительная (II) | 15 очков | A |
        | Предельная (III) | 30 очков | SSS |

        | Ранг Тега | Стоимость |
        |---|---|
        | F: 1 | E: 2 | D: 3 | C: 5 | B: 8 | A: 10 | S: 20 |

        **2. Шаблон для способности:**
        - Название:
        - Тип Ячейки:
        - Описание:
        - Расчет бюджета: Тег1 (Ранг) X очков + Тег2 (Ранг) Y очков = Итого Z очков (не больше бюджета).

        ---

        **Контекст из базы знаний (лор и общие правила):**
        {context}

        **Запрос игрока:**
        {question}

        ---

        **Твоя задача:**

        Создай одну полную концепцию персонажа (например, для ранга C), включая:
        1.  **Общая идея:** Имя, фракция, архетип.
        2.  **Контракт:** Название, описание Существа, Дар.
        3.  **Способности (2-3 шт.):** Придумай способности, опиши их и **обязательно рассчитай их стоимость по тегам**, укладываясь в бюджет ячейки.
        4.  **Синки (1-2 шт.):** Придумай пример Осколка или Эха, который подходит персонажу.
        5.  **Манифестация (если персонаж высокого ранга):** Кратко опиши идею для Манифестации.

        Твоя сгенерированная анкета должна быть креативной, сбалансированной и полностью готовой для игры.
        """
        self.idea_prompt = ChatPromptTemplate.from_template(idea_template)

    def _extract_json(self, text: str) -> dict:
        """
        Извлекает JSON-объект из строки, очищая от мусора.
        """
        # Ищем JSON, который может быть обернут в ```json ... ```
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Если обертки нет, ищем первый попавшийся объект
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                raise ValueError("Не удалось найти JSON в ответе модели.")

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Ошибка декодирования JSON: {e}\nПолученный текст: {json_str}")

    def validate_character_sheet(self, sheet_json: str) -> str:
        """
        Проверяет анкету персонажа на соответствие правилам.
        """
        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.validation_prompt
            | self.llm
            | StrOutputParser()
        )
        # Переводим JSON в строку для передачи в модель
        sheet_str = json.dumps(sheet_json, ensure_ascii=False, indent=2)
        return chain.invoke(sheet_str)

    def generate_character_ideas(self, user_prompt: str) -> str:
        """
        Генерирует идеи для персонажа на основе запроса пользователя.
        """
        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.idea_prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(user_prompt)

    def generate_contract_idea(self, sheet_and_prompt: dict) -> dict:
        """
        Генерирует идею для контракта в два этапа: креатив и форматирование.
        """
        # Шаг 1: Генерация креативной идеи
        creative_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.creative_idea_prompt
            | self.llm
            | StrOutputParser()
        )
        sheet_str = json.dumps(sheet_and_prompt, ensure_ascii=False, indent=2)
        creative_text = creative_chain.invoke(sheet_str)

        # Шаг 2: Форматирование в JSON
        formatting_chain = (
            self.json_formatter_prompt
            | self.llm
            | StrOutputParser()
        )
        
        json_response_str = formatting_chain.invoke({"creative_text": creative_text})
        
        # Шаг 3: Безопасный парсинг JSON
        try:
            # Пробуем сначала стандартный парсинг
            return json.loads(json_response_str)
        except json.JSONDecodeError:
            # Если не удалось, используем наш "умный" экстрактор
            return self._extract_json(json_response_str)

if __name__ == '__main__':
    # Пример использования (для демонстрации)
    # Этот блок не будет выполняться при импорте, только при прямом запуске файла
    
    print("Инициализация ассистента для демонстрации...")
    # Используем одну из моделей для примера
    assistant = Assistant(model_id=config.MODEL_LIST[0])
    print("Ассистент готов к работе.")

    # --- Пример 1: Валидация анкеты ---
    print("\n--- Тест валидации анкеты ---")
    test_sheet = {
        "character_name": "Тест",
        "nickname": "Тестер",
        "age": 25,
        "rank": "F",
        "faction": "Порядок",
        "home_island": "Кага",
        "attributes": {"Сила": "Мастер"}, # Явное нарушение правил (7 очков на старте)
        "contracts": [{
            "creature_rank": "SSS" # Явное нарушение правил
        }]
    }
    validation_result = assistant.validate_character_sheet(test_sheet)
    print("Результат валидации:")
    print(validation_result)

    # --- Пример 2: Генерация идей ---
    print("\n--- Тест генерации идей ---")
    idea_prompt = "Хочу создать персонажа-медика, который работает в трущобах Куро, но тайно связан с Отраженным Светом Солнца."
    idea_result = assistant.generate_character_ideas(idea_prompt)
    print("Результат генерации идей:")
    print(idea_result)