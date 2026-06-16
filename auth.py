# auth.py
import os
import json
import hashlib
from datetime import datetime

USERS_FILE = "data/users.json"

def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def _load_users() -> dict:
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(USERS_FILE):
        # Create default admin account on first run
        default = {"admin": {"password": _hash("admin123"),
                              "role": "admin",
                              "created": str(datetime.now())}}
        _save_users(default)
        return default
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def _save_users(users: dict):
    os.makedirs("data", exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def login(username: str, password: str) -> dict | None:
    """Returns user dict if credentials valid, else None."""
    users = _load_users()
    user  = users.get(username)
    if user and user["password"] == _hash(password):
        return {"username": username, "role": user["role"]}
    return None

def register(username: str, password: str, role: str = "teacher") -> bool:
    """Returns True if registered successfully, False if username taken."""
    users = _load_users()
    if username in users:
        return False
    users[username] = {"password": _hash(password),
                       "role": role,
                       "created": str(datetime.now())}
    _save_users(users)
    return True

def change_password(username: str, old_pw: str, new_pw: str) -> bool:
    users = _load_users()
    user  = users.get(username)
    if user and user["password"] == _hash(old_pw):
        users[username]["password"] = _hash(new_pw)
        _save_users(users)
        return True
    return False

def get_all_users() -> dict:
    return {k: {i: v[i] for i in v if i != "password"}
            for k, v in _load_users().items()}