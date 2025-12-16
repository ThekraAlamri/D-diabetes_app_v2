import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = None

def init_db(db_path: Path):
    global DB_PATH
    DB_PATH = str(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        patient_id TEXT,
        age REAL,
        gender TEXT,
        height_cm REAL,
        weight_kg REAL,
        bmi REAL,
        waist_cm REAL,
        diastolic_bp REAL,
        systolic_bp REAL,
        fasting_glucose REAL,
        random_glucose REAL,
        choles REAL,
        hba1c REAL,
        family_history TEXT,
        smoker TEXT,
        physically_active TEXT,
        predicted_label INTEGER,
        risk_percentage REAL,
        model_name TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    conn.commit()
    conn.close()

def _conn():
    if DB_PATH is None:
        raise RuntimeError("DB not initialized. Call init_db() first.")
    return sqlite3.connect(DB_PATH)

def create_user(username: str, password_hash: str) -> bool:
    conn = _conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?,?,?)",
            (username, password_hash, datetime.utcnow().isoformat())
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_user_by_username(username: str):
    conn = _conn()
    cur = conn.cursor()
    cur.execute("SELECT id, username, password_hash FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "username": row[1], "password_hash": row[2]}

def insert_prediction(user_id: int, payload: dict):
    conn = _conn()
    cur = conn.cursor()

    cols = [
        "user_id","patient_id","age","gender","height_cm","weight_kg","bmi","waist_cm",
        "diastolic_bp","systolic_bp","fasting_glucose","random_glucose","choles","hba1c",
        "family_history","smoker","physically_active",
        "predicted_label","risk_percentage","model_name","created_at"
    ]
    values = [user_id] + [payload.get(c) for c in cols[1:-1]] + [datetime.utcnow().isoformat()]

    cur.execute(f"""
        INSERT INTO predictions ({",".join(cols)})
        VALUES ({",".join(["?"]*len(cols))})
    """, values)

    conn.commit()
    conn.close()

def fetch_predictions(user_id: int, limit: int = 500):
    conn = _conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT patient_id, age, gender, bmi, fasting_glucose, random_glucose, hba1c,
               predicted_label, risk_percentage, model_name, created_at
        FROM predictions
        WHERE user_id=?
        ORDER BY id DESC
        LIMIT ?
    """, (user_id, limit))
    rows = cur.fetchall()
    conn.close()
    return rows
