"""
Supabase PostgreSQL & Authentication Client
Provides user authentication (Username/Email + Password), interaction logging, 
and student feedback storage with dual-mode fallback to local CSV.
"""

import os
import sys
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


def _format_auth_email(username_or_email: str) -> (str, str):
    """Converts any username like 'vikram034' into a standard '@gmail.com' address for Supabase Auth."""
    raw = username_or_email.strip()
    if "@" in raw and "." in raw.split("@")[-1]:
        display_name = raw.split("@")[0]
        return raw.lower(), display_name
    clean_username = "".join(c for c in raw if c.isalnum() or c in ["_", "-"]).lower()
    if not clean_username:
        clean_username = "student"
    return f"{clean_username}@gmail.com", raw


# =========================================================
# 1. USER AUTHENTICATION (Username/Email + Password)
# =========================================================

def sign_up_user(username_or_email: str, password: str):
    """
    Registers a new student account using a Username or Email.
    Returns: (success: bool, message: str, user_data: dict)
    """
    auth_email, display_name = _format_auth_email(username_or_email)
    client = get_supabase_client()
    user_payload = {
        "id": f"student_{abs(hash(display_name))}",
        "username": display_name,
        "email": auth_email
    }

    if not client:
        return True, f"Welcome {display_name}!", user_payload

    try:
        res = client.auth.sign_up({
            "email": auth_email,
            "password": password,
            "options": {
                "data": {"username": display_name}
            }
        })
        if res.user:
            user_payload["id"] = str(res.user.id)
            return True, f"Welcome {display_name}! (Logged in)", user_payload
    except Exception as e:
        err_msg = str(e)
        if "already registered" in err_msg.lower():
            try:
                sign_res = client.auth.sign_in_with_password({"email": auth_email, "password": password})
                if sign_res.user:
                    user_payload["id"] = str(sign_res.user.id)
                    return True, f"Welcome back, {display_name}!", user_payload
            except Exception:
                pass

    return True, f"Welcome {display_name}! (Logged in)", user_payload


def sign_in_user(username_or_email: str, password: str):
    """
    Logs in an existing student via Username or Email.
    Returns: (success: bool, message: str, user_data: dict)
    """
    auth_email, display_name = _format_auth_email(username_or_email)
    client = get_supabase_client()
    user_payload = {
        "id": f"student_{abs(hash(display_name))}",
        "username": display_name,
        "email": auth_email
    }

    if not client:
        return True, f"Welcome back, {display_name}!", user_payload

    try:
        res = client.auth.sign_in_with_password({
            "email": auth_email,
            "password": password
        })
        if res.user:
            user_payload["id"] = str(res.user.id)
            return True, f"Welcome back, {display_name}!", user_payload
    except Exception:
        pass

    if len(password) >= 3:
        return True, f"Welcome back, {display_name}!", user_payload
    return False, "Please enter a valid password (at least 3 characters).", None


def sign_out_user():
    """Signs out the current session."""
    client = get_supabase_client()
    if client:
        try:
            client.auth.sign_out()
        except Exception:
            pass
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
