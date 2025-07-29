from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from sdp_ai_assistant import Assistant
from sqlite3 import Connection
import sqlite3
import config
import uuid
import logging
import requests
from pydantic import BaseModel
import argparse

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

tasks = {}

class CharacterData(BaseModel):
    character_data: dict
    vk_id: int = 0
    admin_id: str | None = None
    character_id: int | None = None

class ContractIdeaData(BaseModel):
    character_data: dict
    user_prompt: str

def process_validation(task_id: str, character_data: dict, character_id: int = None):
    logger.info(f"Starting validation for task {task_id}")
    try:
        result = assistant.validate_character_sheet(character_data)
        tasks[task_id] = {"status": "completed", "result": result}
        logger.info(f"Validation for task {task_id} completed successfully")
        
        if character_id:
            try:
                requests.post(
                    f"http://localhost:3000/api/characters/{character_id}/ai-analysis",
                    json={"result": result}
                )
                logger.info(f"Successfully sent AI analysis for character {character_id} to backend")
            except Exception as e:
                logger.error(f"Failed to send AI analysis for character {character_id} to backend: {e}")

    except Exception as e:
        tasks[task_id] = {"status": "error", "detail": str(e)}
        logger.error(f"Validation for task {task_id} failed: {e}")

def process_contract_generation(task_id: str, character_data: dict, user_prompt: str):
    logger.info(f"Starting contract generation for task {task_id}")
    try:
        result = assistant.generate_contract_idea({
            "character_data": character_data,
            "user_prompt": user_prompt
        })
        tasks[task_id] = {"status": "completed", "result": json.dumps(result)}
        logger.info(f"Contract generation for task {task_id} completed successfully")
    except Exception as e:
        tasks[task_id] = {"status": "error", "detail": str(e)}
        logger.error(f"Contract generation for task {task_id} failed: {e}")

@app.post("/validate")
async def start_validation(
    data: CharacterData,
    background_tasks: BackgroundTasks,
    db: Connection = Depends(get_db)
):
    # Check rate limit
    if not no_cd and not data.admin_id:
        cursor = db.execute(
            "SELECT last_request FROM ai_requests WHERE vk_id = ?",
            (data.vk_id,)
        )
        result = cursor.fetchone()
        
        if result and datetime.now() - datetime.fromisoformat(result[0]) < timedelta(hours=3):
            raise HTTPException(
                status_code=429,
                detail="You can only make one request every 3 hours"
            )

    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing"}
    
    background_tasks.add_task(process_validation, task_id, data.character_data, data.character_id)
    
    # Update request timestamp
    db.execute(
        "INSERT OR REPLACE INTO ai_requests (vk_id, last_request) VALUES (?, ?)",
        (data.vk_id, datetime.now().isoformat())
    )
    db.commit()

    return {"task_id": task_id}

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.post("/generate-contract")
async def start_contract_generation(
    data: ContractIdeaData,
    background_tasks: BackgroundTasks
):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing"}
    
    background_tasks.add_task(process_contract_generation, task_id, data.character_data, data.user_prompt)
    
    return {"task_id": task_id}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cd", action="store_true", help="Disable cooldown")
    args = parser.parse_args()
    no_cd = args.no_cd

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)