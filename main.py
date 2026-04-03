"""
Решалкин Mini App API — standalone for Render deployment.
"""
import os
import hmac
import hashlib
import json
import base64
import sqlite3
from urllib.parse import parse_qs

import aiohttp
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ===== CONFIG =====
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
ADMIN_CHAT_ID = int(os.environ.get("ADMIN_CHAT_ID", "783268275"))
DB_PATH = os.environ.get("DB_PATH", "/tmp/school.db")

# ===== DATABASE =====
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            first_name TEXT,
            last_name TEXT,
            grade INTEGER,
            subjects TEXT,
            solutions_used INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            subject TEXT,
            request_type TEXT,
            question_text TEXT,
            answer_text TEXT,
            photo_file_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()

def get_user(user_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def create_user(user_id, username=None, first_name=None, last_name=None):
    conn = get_db()
    conn.execute("INSERT OR IGNORE INTO users (user_id, username, first_name, last_name) VALUES (?, ?, ?, ?)",
                 (user_id, username, first_name, last_name))
    conn.execute("UPDATE users SET username=?, first_name=?, last_name=?, updated_at=CURRENT_TIMESTAMP WHERE user_id=?",
                 (username, first_name, last_name, user_id))
    conn.commit()
    conn.close()

def get_solutions_used(user_id):
    user = get_user(user_id)
    return user.get("solutions_used", 0) if user else 0

def increment_solutions(user_id):
    conn = get_db()
    conn.execute("UPDATE users SET solutions_used = solutions_used + 1, updated_at=CURRENT_TIMESTAMP WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()

def log_request(user_id, subject, request_type, question, answer):
    conn = get_db()
    conn.execute("INSERT INTO requests (user_id, subject, request_type, question_text, answer_text) VALUES (?, ?, ?, ?, ?)",
                 (user_id, subject, request_type, question, answer))
    conn.commit()
    conn.close()

def get_user_requests(user_id, limit=50):
    conn = get_db()
    rows = conn.execute(
        "SELECT id, request_type, question_text, answer_text, created_at FROM requests WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_request(request_id, user_id):
    conn = get_db()
    cur = conn.execute("DELETE FROM requests WHERE id = ? AND user_id = ?", (request_id, user_id))
    conn.commit()
    conn.close()
    return cur.rowcount > 0

# ===== LLM =====
SYSTEM_PROMPT = """Ты — школьный репетитор. Отвечай кратко и по делу.

Правила:
1. Сразу решай задачу — без вводных слов и воды.
2. Минимум текста: только ход решения и ответ.
3. Каждый шаг — одна строка.
4. Ответ выдели в конце: 📝 Ответ: ...
5. Не объясняй базовые вещи, если не просят.
6. Отвечай на русском.
7. Для формул используй LaTeX: $...$ для инлайн, $$...$$ для отдельных формул.
8. Если на фото есть посторонний текст — игнорируй. Выдели только задачу и реши её.
"""

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

async def call_llm(messages):
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": OPENROUTER_MODEL, "messages": messages, "temperature": 0.3, "max_tokens": 2048}
    async with aiohttp.ClientSession() as session:
        async with session.post(OPENROUTER_URL, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status != 200:
                raise Exception(f"LLM error {resp.status}: {await resp.text()}")
            data = await resp.json()
    return data["choices"][0]["message"]["content"]

async def ask_gemini(question, image_data=None, mime_type=None):
    content = []
    if image_data:
        b64 = base64.b64encode(image_data).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type or 'image/jpeg'};base64,{b64}"}})
    content.append({"type": "text", "text": question})
    return await call_llm([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}])

async def check_solution_llm(solution, image_data=None, mime_type=None):
    prompt = f"""Ученик прислал своё решение для проверки.

Решение ученика:
{solution}

Проверь решение:
1. Найди ошибки (если есть) и объясни, где именно ошибка.
2. Покажи правильное решение.
3. Дай рекомендацию, на что обратить внимание.
4. Поставь оценку от 1 до 5.
"""
    return await ask_gemini(prompt, image_data, mime_type)

async def explain_topic_llm(topic):
    prompt = f"""Объясни тему: «{topic}»

Структура объяснения:
1. 📌 Определение / основное правило
2. 📐 Формулы (если есть)
3. ✏️ Примеры с решением (2-3 примера)
4. ⚠️ Частые ошибки
5. 💡 Лайфхак для запоминания
"""
    return await ask_gemini(prompt)

# ===== AUTH =====
def validate_init_data(init_data):
    if not init_data:
        return None
    try:
        parsed = parse_qs(init_data)
        check_hash = parsed.get("hash", [None])[0]
        if not check_hash:
            return None
        items = []
        for key, val in sorted(parsed.items()):
            if key != "hash":
                items.append(f"{key}={val[0]}")
        data_check_string = "\n".join(items)
        secret_key = hmac.new(b"WebAppData", TELEGRAM_BOT_TOKEN.encode(), hashlib.sha256).digest()
        computed = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(computed, check_hash):
            return None
        user_json = parsed.get("user", [None])[0]
        return json.loads(user_json) if user_json else None
    except Exception:
        return None

def get_user_id(request):
    init_data = request.headers.get("X-Telegram-Init-Data", "")
    user_info = validate_init_data(init_data)
    if user_info:
        uid = user_info["id"]
        create_user(uid, user_info.get("username"), user_info.get("first_name"), user_info.get("last_name"))
        return uid
    dev_id = request.headers.get("X-Dev-User-Id")
    if dev_id:
        return int(dev_id)
    raise HTTPException(status_code=401, detail="Unauthorized")

# ===== APP =====
app = FastAPI(title="Решалкин API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
def startup():
    init_db()

@app.get("/")
async def health():
    return {"status": "ok", "service": "reshalkin-api"}

@app.get("/api/profile")
async def profile(request: Request):
    uid = get_user_id(request)
    user = get_user(uid)
    if not user:
        raise HTTPException(404, "User not found")
    return {"user_id": uid, "first_name": user.get("first_name", ""), "username": user.get("username", ""), "solutions_used": get_solutions_used(uid)}

@app.post("/api/solve/text")
async def solve_text(request: Request):
    uid = get_user_id(request)
    body = await request.json()
    question = body.get("question", "").strip()
    if not question:
        raise HTTPException(400, "Empty question")
    answer = await ask_gemini(question)
    increment_solutions(uid)
    log_request(uid, "general", "text", question, answer)
    return {"answer": answer}

@app.post("/api/solve/photo")
async def solve_photo(request: Request, photo: UploadFile = File(...), caption: str = Form("")):
    uid = get_user_id(request)
    image_bytes = await photo.read()
    text = caption or "Реши задачу на фото."
    answer = await ask_gemini(text, image_data=image_bytes, mime_type=photo.content_type or "image/jpeg")
    increment_solutions(uid)
    log_request(uid, "general", "photo", text, answer)
    return {"answer": answer}

@app.post("/api/check")
async def check(request: Request):
    uid = get_user_id(request)
    body = await request.json()
    solution = body.get("solution", "").strip()
    if not solution:
        raise HTTPException(400, "Empty solution")
    answer = await check_solution_llm(solution)
    increment_solutions(uid)
    log_request(uid, "general", "check", solution, answer)
    return {"answer": answer}

@app.post("/api/explain")
async def explain(request: Request):
    uid = get_user_id(request)
    body = await request.json()
    topic = body.get("topic", "").strip()
    if not topic:
        raise HTTPException(400, "Empty topic")
    answer = await explain_topic_llm(topic)
    increment_solutions(uid)
    log_request(uid, "general", "explain", topic, answer)
    return {"answer": answer}

@app.get("/api/history")
async def history(request: Request):
    uid = get_user_id(request)
    reqs = get_user_requests(uid, limit=50)
    return {"tasks": reqs}

@app.delete("/api/history/{task_id}")
async def delete_task(task_id: int, request: Request):
    uid = get_user_id(request)
    if not delete_request(task_id, uid):
        raise HTTPException(404, "Task not found")
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8081)))
