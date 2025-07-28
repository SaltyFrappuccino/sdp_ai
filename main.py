from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from sdp_ai_assistant import Assistant
from sqlite3 import Connection
import sqlite3
import config

app = FastAPI()

# CORS configuration
origins = [
    "https://sdp-ten-sand.vercel.app",
    "http://localhost:5173",
    r"https:\/\/prod-app53964840-.*\.pages-ac\.vk-apps\.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https:\/\/prod-app53964840-.*\.pages-ac\.vk-apps\.com",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup for rate limiting
def get_db():
    conn = sqlite3.connect('db/ai_requests.db', check_same_thread=False)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS ai_requests (
            vk_id INTEGER PRIMARY KEY,
            last_request TEXT NOT NULL
        )
    ''')
    try:
        yield conn
    finally:
        conn.close()

# Initialize AI assistant
assistant = Assistant(config.MODEL_LIST[1])  # Используем gemini-2.5-flash

@app.post("/")
async def validate_character(
    character_data: dict,
    vk_id: int = 0, # Добавляем значение по умолчанию
    db: Connection = Depends(get_db)
):
    # Check rate limit
    cursor = db.execute(
        "SELECT last_request FROM ai_requests WHERE vk_id = ?",
        (vk_id,)
    )
    result = cursor.fetchone()
    
    if result and datetime.now() - datetime.fromisoformat(result[0]) < timedelta(hours=3):
        raise HTTPException(
            status_code=429,
            detail="You can only make one request every 3 hours"
        )

    # Validate character sheet
    validation_result = assistant.validate_character_sheet(character_data)
    
    # Update request timestamp
    db.execute(
        "INSERT OR REPLACE INTO ai_requests (vk_id, last_request) VALUES (?, ?)",
        (vk_id, datetime.now().isoformat())
    )
    db.commit()

    return {"result": validation_result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)