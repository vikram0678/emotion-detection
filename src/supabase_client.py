"""
Supabase PostgreSQL & Authentication Client
Provides user authentication, interaction logging, and student feedback storage
with automatic fallback to local CSV persistence for resilient zero-downtime execution.
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

    # 1. Check Streamlit secrets first
    url = None
    key = None
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            url = st.secrets.get("SUPABASE_URL") or st.secrets.get("supabase_url")
            key = st.secrets.get("SUPABASE_KEY") or st.secrets.get("supabase_key") or st.secrets.get("SUPABASE_ANON_KEY")
    except Exception:
        pass

    # 2. Check environment variables
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


# =========================================================
# 1. USER AUTHENTICATION (Login / Sign Up / Sign Out)
# =========================================================

def sign_up_user(email: str, password: str):
    """
    Registers a new student account via Supabase Auth.
    Returns: (success: bool, message: str, user_data: dict)
    """
    client = get_supabase_client()
    if not client:
        return False, "Supabase is not connected. You can continue as a Guest.", None

    try:
        res = client.auth.sign_up({"email": email.strip(), "password": password})
        if res.user:
            return True, "Account created successfully! You can now log in.", {
                "id": str(res.user.id),
                "email": res.user.email
            }
        return False, "Failed to create account. Please check your details.", None
    except Exception as e:
        err_msg = str(e)
        if "User already registered" in err_msg:
            return False, "This email is already registered. Please log in instead.", None
        return False, f"Sign-up error: {err_msg}", None


def sign_in_user(email: str, password: str):
    """
    Authenticates an existing student via Supabase Auth.
    Returns: (success: bool, message: str, user_data: dict)
    """
    client = get_supabase_client()
    if not client:
        # Local mock login for demo when Supabase is not connected
        if email and password:
            return True, "Logged in as local user (Offline Mode).", {
                "id": f"local_{abs(hash(email))}",
                "email": email.strip()
            }
        return False, "Please enter a valid email and password.", None

    try:
        res = client.auth.sign_in_with_password({"email": email.strip(), "password": password})
        if res.user:
            return True, "Login successful!", {
                "id": str(res.user.id),
                "email": res.user.email
            }
        return False, "Invalid login credentials.", None
    except Exception as e:
        return False, f"Login error: {str(e)}", None


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
    with automatic fallback / mirror to local CSV.
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

    # 1. Try Supabase insert
    supabase_saved = False
    client = get_supabase_client()
    if client:
        try:
            client.table("student_interactions").insert(record).execute()
            supabase_saved = True
        except Exception as e:
            print(f"⚠️ Supabase interaction insert error: {e}")

    # 2. Local CSV mirror (guarantees zero data loss)
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

    # 1. Try Supabase insert
    supabase_saved = False
    client = get_supabase_client()
    if client:
        try:
            client.table("student_feedback").insert(feedback_record).execute()
            supabase_saved = True
        except Exception as e:
            print(f"⚠️ Supabase feedback insert error: {e}")

    # 2. Local CSV fallback
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
