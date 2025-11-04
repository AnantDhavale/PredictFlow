from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, validator

router = APIRouter(prefix="/users", tags=["users"])

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "predictflow.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Auth
# -----------------------------
def require_api_key(x_api_key: Optional[str] = None):
    expected = os.getenv("PREDICTFLOW_API_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="API not configured (missing PREDICTFLOW_API_KEY).")
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return True

def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=5, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn

def _init_tables():
    conn = _conn()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                role TEXT NOT NULL,           -- admin|operator|viewer
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """
        )
    finally:
        conn.close()

_init_tables()

# -----------------------------
# Models & Validators
# -----------------------------
ROLES = {"admin", "operator", "viewer"}
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")

class UserCreate(BaseModel):
    email: str = Field(..., max_length=255)
    display_name: str = Field(..., min_length=1, max_length=80)
    role: str = Field("viewer")

    @validator("email")
    def valid_email(cls, v: str) -> str:
        v = v.strip().lower()
        if not EMAIL_RE.match(v):
            raise ValueError("invalid email")
        return v

    @validator("role")
    def valid_role(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ROLES:
            raise ValueError("role must be one of: admin, operator, viewer")
        return v

class UserUpdate(BaseModel):
    display_name: Optional[str] = Field(None, min_length=1, max_length=80)
    role: Optional[str] = Field(None)

    @validator("role")
    def valid_role(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip().lower()
        if v not in ROLES:
            raise ValueError("role must be one of: admin, operator, viewer")
        return v

class UserOut(BaseModel):
    id: int
    email: str
    display_name: str
    role: str
    created_at: str

def _row_to_user(row: sqlite3.Row) -> UserOut:
    return UserOut(
        id=row["id"],
        email=row["email"],
        display_name=row["display_name"],
        role=row["role"],
        created_at=row["created_at"],
    )

# -----------------------------
# Endpoints
# -----------------------------
@router.get("/", response_model=List[UserOut])
def list_users(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    role: Optional[str] = Query(None, description="admin|operator|viewer"),
    _auth=Depends(require_api_key),
):
    conn = _conn()
    try:
        if role:
            r = role.strip().lower()
            if r not in ROLES:
                raise HTTPException(400, "Invalid role filter")
            cur = conn.execute(
                "SELECT * FROM users WHERE role = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (r, limit, offset),
            )
        else:
            cur = conn.execute(
                "SELECT * FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        return [_row_to_user(r) for r in cur.fetchall()]
    finally:
        conn.close()


@router.post("/", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user(payload: UserCreate, _auth=Depends(require_api_key)):
    conn = _conn()
    try:
        # ensure unique email
        exists = conn.execute("SELECT 1 FROM users WHERE email = ?", (payload.email,)).fetchone()
        if exists:
            raise HTTPException(409, "Email already exists")
        conn.execute(
            "INSERT INTO users(email, display_name, role) VALUES (?, ?, ?)",
            (payload.email, payload.display_name.strip(), payload.role),
        )
        row = conn.execute("SELECT * FROM users WHERE email = ?", (payload.email,)).fetchone()
        return _row_to_user(row)
    finally:
        conn.close()


@router.get("/{user_id}", response_model=UserOut)
def get_user(user_id: int, _auth=Depends(require_api_key)):
    conn = _conn()
    try:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if not row:
            raise HTTPException(404, "User not found")
        return _row_to_user(row)
    finally:
        conn.close()


@router.patch("/{user_id}", response_model=UserOut)
def update_user(user_id: int, payload: UserUpdate, _auth=Depends(require_api_key)):
    conn = _conn()
    try:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if not row:
            raise HTTPException(404, "User not found")

        fields, params = [], []
        if payload.display_name is not None:
            fields.append("display_name = ?")
            params.append(payload.display_name.strip())
        if payload.role is not None:
            fields.append("role = ?")
            params.append(payload.role)

        if fields:
            q = f"UPDATE users SET {', '.join(fields)} WHERE id = ?"
            params.append(user_id)
            conn.execute(q, tuple(params))

        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return _row_to_user(row)
    finally:
        conn.close()


@router.delete("/{user_id}", status_code=204)
def delete_user(user_id: int, _auth=Depends(require_api_key)):
    conn = _conn()
    try:
        cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        if cur.rowcount == 0:
            raise HTTPException(404, "User not found")
        return
    finally:
        conn.close()
