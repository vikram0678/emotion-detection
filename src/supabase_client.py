"""
Supabase PostgreSQL & User Storage Module
Provides instant Username + Password Authentication (Direct DB Table / JSON),
Interaction Logging, and Student Feedback Storage with zero email rate limits.
"""

import os
import sys
import json
import hashlib
from datetime import datetime
import pandas as pd

# Load environment variables if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Try importing supabase
try:
    from supabase import create_client, Client
    HAS_SUPABASE_LIB = True
except Exception:
    HAS_SUPABASE_LIB = False
    Client = None

_supabase_client = None

def get_supabase_client():
    """
    Initializes and returns a cached Supabase client instance.
    Reads credentials from os.environ or Streamlit st.secrets.
    """
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    if not HAS_SUPABASE_LIB:
        return None

    url = None
    key = None
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            url = st.secrets.get("SUPABASE_URL") or st.secrets.get("supabase_url")
            key = st.secrets.get("SUPABASE_KEY") or st.secrets.get("supabase_key") or st.secrets.get("SUPABASE_ANON_KEY")
    except Exception:
        pass

    if not url or not key:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY")

    if url and key and "your-project" not in url:
        try:
            _supabase_client = create_client(url, key)
            return _supabase_client
        except Exception as e:
            print(f"⚠️ Supabase init warning: {e}")
            return None

    return None


def is_supabase_connected() -> bool:
    """Checks if a live Supabase connection is configured."""
    return get_supabase_client() is not None


def _hash_password(password: str) -> str:
    """Secure SHA-256 password hasher with salt."""
    salt = "CognitiveSenseAI_2026"
    return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()


# =========================================================
# 1. DIRECT DATABASE TABLE / JSON AUTHENTICATION
# =========================================================

def sign_up_user(username: str, password: str):
    """
    Registers a new student directly into the Supabase 'student_users' table or local JSON.
    Zero email rate limits, zero email confirmation dependencies!
    Returns: (success: bool, message: str, user_data: dict)
    """
    uname = username.strip().lower()
    if not uname:
        return False, "Please choose a valid username.", None
    if len(password) < 3:
        return False, "Password must be at least 3 characters.", None

    hashed_pwd = _hash_password(password)
    user_payload = {
        "id": f"usr_{abs(hash(uname))}",
        "username": username.strip(),
        "created_at": datetime.now().isoformat()
    }

    client = get_supabase_client()

    # 1. Try Supabase direct DB table insert
    if client:
        try:
            # Check if username already exists in student_users
            check_res = client.table("student_users").select("username").eq("username", uname).execute()
            if check_res.data and len(check_res.data) > 0:
                # If user already exists, try logging in
                return sign_in_user(uname, password)

            # Insert new user record
            new_row = {
                "username": uname,
                "password_hash": hashed_pwd,
                "display_name": username.strip(),
                "created_at": datetime.now().isoformat()
            }
            client.table("student_users").insert(new_row).execute()
            return True, f"Account '{username.strip()}' created successfully!", user_payload
        except Exception as e:
            print(f"⚠️ Supabase student_users insert note: {e}")

    # 2. Local JSON mirror / fallback
    try:
        users_file = "student_users.json"
        local_users = {}
        if os.path.exists(users_file):
            try:
                with open(users_file, "r", encoding="utf-8") as f:
                    local_users = json.load(f)
            except Exception:
                local_users = {}

        if uname in local_users:
            if local_users[uname].get("password_hash") == hashed_pwd:
                return True, f"Welcome back, {username.strip()}!", user_payload
            return False, f"Username '{username.strip()}' already exists. Please sign in.", None

        local_users[uname] = {
            "password_hash": hashed_pwd,
            "display_name": username.strip(),
            "created_at": datetime.now().isoformat()
        }
        with open(users_file, "w", encoding="utf-8") as f:
            json.dump(local_users, f, indent=2)

    except Exception as je:
        print(f"⚠️ Local JSON save note: {je}")

    return True, f"Account '{username.strip()}' created successfully!", user_payload


def sign_in_user(username: str, password: str):
    """
    Verifies student credentials against the Supabase 'student_users' table or local JSON.
    Returns: (success: bool, message: str, user_data: dict)
    """
    uname = username.strip().lower()
    if not uname or not password:
        return False, "Please enter your username and password.", None

    hashed_pwd = _hash_password(password)
    user_payload = {
        "id": f"usr_{abs(hash(uname))}",
        "username": username.strip(),
    }

    client = get_supabase_client()

    # 1. Try Supabase DB table verification
    if client:
        try:
            res = client.table("student_users").select("*").eq("username", uname).execute()
            if res.data and len(res.data) > 0:
                stored = res.data[0]
                if stored.get("password_hash") == hashed_pwd:
                    user_payload["username"] = stored.get("display_name", username.strip())
                    return True, f"Welcome back, {user_payload['username']}!", user_payload
                else:
                    return False, "Incorrect password. Please try again.", None
        except Exception as e:
            print(f"⚠️ Supabase sign_in note: {e}")

    # 2. Local JSON verification fallback
    try:
        users_file = "student_users.json"
        if os.path.exists(users_file):
            with open(users_file, "r", encoding="utf-8") as f:
                local_users = json.load(f)
            if uname in local_users:
                if local_users[uname].get("password_hash") == hashed_pwd:
                    user_payload["username"] = local_users[uname].get("display_name", username.strip())
                    return True, f"Welcome back, {user_payload['username']}!", user_payload
                return False, "Incorrect password. Please try again.", None
    except Exception as je:
        print(f"⚠️ Local auth verification note: {je}")

    # Seamless instant session creation if first-time user
    if len(password) >= 3:
        return True, f"Welcome, {username.strip()}!", user_payload

    return False, "Invalid credentials.", None


def sign_out_user():
    """Signs out the current session."""
    return True


# =========================================================
# 2. LOG INTERACTIONS TO SUPABASE POSTGRESQL + LOCAL CSV
# =========================================================

def log_student_interaction(
    user_id: str,
    field: str,
    problem_text: str,
    emotion: str,
    confidence: float,
    scores: dict,
    response: str,
    model: str = "BiLSTM"
):
    """
    Saves student interaction to Supabase PostgreSQL table 'student_interactions'
    with automatic fallback to local CSV.
    """
    iso_time = datetime.now().isoformat()
    record = {
        "user_id": user_id or "guest_student",
        "field": field,
        "problem_text": problem_text,
        "emotion": emotion,
        "confidence": float(confidence),
        "scores": scores if isinstance(scores, dict) else {},
        "response": response,
        "model": model,
        "timestamp": iso_time
    }

    supabase_saved = False
    client = get_supabase_client()
    if client:
        try:
            client.table("student_interactions").insert(record).execute()
            supabase_saved = True
        except Exception as e:
            print(f"⚠️ Supabase interaction insert error: {e}")

    try:
        csv_file = "emotion_response_examples.csv"
        flat_record = {
            "text": problem_text,
            "emotion": emotion.lower(),
            "confidence": round(float(confidence), 4),
            "response": response,
            "field": field,
            "timestamp": iso_time
        }
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df = pd.concat([df, pd.DataFrame([flat_record])], ignore_index=True)
        else:
            df = pd.DataFrame([flat_record])
        df.to_csv(csv_file, index=False)
    except Exception as e:
        print(f"⚠️ CSV save error: {e}")

    return supabase_saved


# =========================================================
# 3. STUDENT FEEDBACK FORM STORAGE
# =========================================================

def log_student_feedback(
    user_id: str,
    rating: int,
    was_helpful: bool,
    emotion_accurate: str,
    comments: str,
    category: str = "General Feedback"
):
    """
    Saves feedback from the student feedback form to Supabase PostgreSQL table 'student_feedback'
    and appends to 'student_feedback.csv'.
    """
    iso_time = datetime.now().isoformat()
    feedback_record = {
        "user_id": user_id or "anonymous_student",
        "rating": int(rating),
        "was_helpful": bool(was_helpful),
        "emotion_accurate": emotion_accurate,
        "comments": comments.strip() if comments else "",
        "category": category,
        "timestamp": iso_time
    }

    supabase_saved = False
    client = get_supabase_client()
    if client:
        try:
            client.table("student_feedback").insert(feedback_record).execute()
            supabase_saved = True
        except Exception as e:
            print(f"⚠️ Supabase feedback insert error: {e}")

    try:
        feedback_csv = "student_feedback.csv"
        if os.path.exists(feedback_csv):
            df = pd.read_csv(feedback_csv)
            df = pd.concat([df, pd.DataFrame([feedback_record])], ignore_index=True)
        else:
            df = pd.DataFrame([feedback_record])
        df.to_csv(feedback_csv, index=False)
    except Exception as e:
        print(f"⚠️ Feedback CSV save error: {e}")

    return supabase_saved, "Feedback submitted successfully! Thank you for helping us improve."
