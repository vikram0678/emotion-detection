import sys
import os
import time
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

# Configure environment to suppress TensorFlow verbose logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Ensure UTF-8 output encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from mixed_emotion import get_mixed_emotions, EMOTION_RESPONSES
from gemini_helper import get_gemini_response
from supabase_client import (
    is_supabase_connected,
    sign_in_user,
    sign_up_user,
    sign_out_user,
    log_student_interaction,
    log_student_feedback,
)

# ============================================
# 1. PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="AI Learning Assistant - Emotion Aware",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for the exact dark theme styling in project screenshots
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background & Card Styling */
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 2px;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-bottom: 12px;
    }

    /* Response Card styling */
    .response-container {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 24px;
        margin-top: 20px;
    }
    .response-header-chip {
        background: #1E3A8A;
        color: #93C5FD;
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 16px;
        display: inline-block;
        width: 100%;
    }
    .strategy-pill {
        background: #1E293B;
        color: #38BDF8;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 0.92rem;
        font-weight: 500;
        display: inline-block;
        margin-top: 6px;
    }

    /* Red Primary Button matching screenshot */
    div.stButton > button[kind="primary"] {
        background-color: #EF4444;
        background-image: linear-gradient(90deg, #F87171, #EF4444);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 12px 24px;
        box-shadow: 0 4px 14px 0 rgba(239, 68, 68, 0.39);
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #DC2626;
        box-shadow: 0 6px 20px rgba(239, 68, 68, 0.5);
        transform: translateY(-1px);
    }

    /* Example Buttons styling */
    div.stButton > button:not([kind="primary"]) {
        background: #1E293B;
        color: #E2E8F0;
        border: 1px solid #334155;
        border-radius: 8px;
        font-size: 0.85rem;
        padding: 8px 14px;
    }
    div.stButton > button:not([kind="primary"]):hover {
        background: #334155;
        color: #FFFFFF;
        border-color: #64748B;
    }

    /* Big Mixed Emotion Header */
    .mixed-emotion-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 4px;
    }
    .primary-metric-sub {
        color: #10B981;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 12px;
    }

    /* Top Right Profile Card */
    .top-profile-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 10px 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    .top-profile-name {
        color: #38BDF8;
        font-weight: 600;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# 2. SESSION STATE INITIALIZATION
# ============================================
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None

if "user" not in st.session_state:
    st.session_state.user = None  # None indicates Guest Student mode


# ============================================
# 3. FAST CACHED MODEL LOADER
# ============================================
@st.cache_resource(show_spinner="⚡ Loading models...")
def load_models():
    """Loads BiLSTM model and BERT/MiniLM Transformer with instant caching."""
    bilstm_model = None
    bert_model = None
    status_msg = "Models loaded"

    # 1. PyTorch BiLSTM Predictor
    try:
        from bilstm_predictor import EmotionPredictor
        bilstm_model = EmotionPredictor(
            model_path="models/bltsm/bilstm_model.pt",
            vocab_path="models/bltsm/vocab.json",
        )
    except Exception as e:
        status_msg = f"BiLSTM: {e}"

    # 2. MiniLM ONNX INT8 Classifier
    try:
        from bert_model import BERTEmotionClassifier
        bert_model = BERTEmotionClassifier()
        if os.path.exists("models/bert_student_adaptive"):
            bert_model.load_model("models/bert_student_adaptive")
        elif os.path.exists("models/minilm_student_adaptive"):
            bert_model.load_model("models/minilm_student_adaptive")
        else:
            bert_model.load_model("models/bert_emotion_model_final")
    except Exception as e:
        status_msg = f"BERT: {e}"

    return bilstm_model, bert_model, status_msg


bilstm_model, bert_model, status_text = load_models()


# ============================================
# 4. HISTORY LOGGING HELPER
# ============================================
def add_to_history(field, problem, emotion, confidence, ai_response, bilstm_scores, bert_result=None):
    """Appends interactions to session state for BiLSTM and BERT predictions."""
    bilstm_mixed = get_mixed_emotions(bilstm_scores)
    bilstm_label = " + ".join([em[0] for em in bilstm_mixed]) if len(bilstm_mixed) > 1 else emotion

    st.session_state.emotion_history.append({
        "timestamp": datetime.now(),
        "field": field,
        "problem": problem,
        "emotion": bilstm_label,
        "confidence": confidence,
        "ai_response": ai_response,
        "all_scores": bilstm_scores,
        "model": "BiLSTM",
    })

    if bert_result:
        bert_mixed = get_mixed_emotions(bert_result["scores"])
        bert_label = " + ".join([em[0] for em in bert_mixed]) if len(bert_mixed) > 1 else bert_result["emotion"]

        st.session_state.emotion_history.append({
            "timestamp": datetime.now(),
            "field": field,
            "problem": problem,
            "emotion": bert_label,
            "confidence": bert_result["confidence"],
            "ai_response": ai_response,
            "all_scores": bert_result["scores"],
            "model": "BERT",
        })


# ============================================
# 5. SIDEBAR: SYSTEM DASHBOARD
# ============================================
examples_count = 0
if os.path.exists("emotion_response_examples.csv"):
    try:
        ex_df = pd.read_csv("emotion_response_examples.csv")
        examples_count = len(ex_df)
    except Exception:
        examples_count = 55

supabase_active = is_supabase_connected()

with st.sidebar:
    st.markdown("### 📊 System Dashboard")
    st.markdown(f"**Models:** ✅ {status_text}")
    st.markdown(f"**Database:** {'☁️ Supabase PostgreSQL' if supabase_active else '📁 Local CSV'}")
    st.markdown(f"**Total Interactions:** `{len(st.session_state.emotion_history)}`")
    st.markdown(f"**Saved Examples:** `{examples_count}`")

    if st.button("Clear History", use_container_width=True):
        st.session_state.emotion_history = []
        st.session_state.last_analysis = None
        st.session_state.input_text = ""
        st.rerun()

    if st.session_state.emotion_history:
        st.markdown("---")
        st.markdown("#### 🕒 Recent Sessions")
        recent = st.session_state.emotion_history[-3:]
        for item in reversed(recent):
            st.markdown(f"• **{item['field']}**: {item['emotion']} (`{item['confidence']:.1%}`)")


# ============================================
# 6. TOP HEADER WITH USER PROFILE (RIGHT TOP)
# ============================================
head_col1, head_col2 = st.columns([3.2, 1.3], gap="medium")

with head_col1:
    st.markdown("""
    <div class="hero-title">🤖 Emotion-Aware Learning Assistant</div>
    <div class="hero-subtitle">Get personalized pedagogical guidance based on your field and emotional state</div>
    """, unsafe_allow_html=True)

with head_col2:
    if st.session_state.user:
        u_name = st.session_state.user.get("username", "Student")
        st.markdown(f"""
        <div class="top-profile-card">
            <div>
                <span style="font-size: 1.1rem;">🎓</span> 
                <span class="top-profile-name">{u_name}</span>
                <div style="font-size: 0.75rem; color: #10B981;">● Logged In</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚪 Sign Out", key="top_signout_btn", use_container_width=True):
            sign_out_user()
            st.session_state.user = None
            st.rerun()
    else:
        st.markdown("""
        <div class="top-profile-card">
            <div>
                <span style="font-size: 1.1rem;">👤</span> 
                <span style="color: #94A3B8; font-size: 0.9rem; font-weight: 500;">Guest Student</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("🔑 Sign In / Sign Up", expanded=False):
            t_tab1, t_tab2 = st.tabs(["Log In", "Sign Up"])
            
            with t_tab1:
                uname = st.text_input("Username or Email", key="top_login_u")
                upwd = st.text_input("Password", type="password", key="top_login_p")
                if st.button("Sign In", key="top_login_btn", use_container_width=True):
                    if uname and upwd:
                        ok, msg, udata = sign_in_user(uname, upwd)
                        if ok:
                            st.session_state.user = udata
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
                    else:
                        st.warning("Please enter username and password.")

            with t_tab2:
                new_u = st.text_input("Choose Username", key="top_reg_u")
                new_p = st.text_input("Choose Password", type="password", key="top_reg_p")
                if st.button("Create Account", key="top_reg_btn", use_container_width=True):
                    if new_u and new_p:
                        ok, msg, udata = sign_up_user(new_u, new_p)
                        if ok:
                            st.session_state.user = udata
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
                    else:
                        st.warning("Please provide a username and password.")


# ============================================
# 7. MAIN NAVIGATION TABS
# ============================================
main_tab1, main_tab2, main_tab3 = st.tabs([
    "🤖 Learning Assistant",
    "📊 Analytics & Journey",
    "📝 Student Feedback"
])


# ============================================
# TAB 1: LEARNING ASSISTANT
# ============================================
with main_tab1:
    col1, col2 = st.columns([2.5, 1.3], gap="large")

    with col1:
        st.markdown("#### 📝 Tell us about your learning challenge")
        
        fields = [
            "Computer Science", "Mathematics", "Physics", "Chemistry", "Biology",
            "Engineering", "Business", "Literature", "History", "Psychology", "Other"
        ]
        
        field = st.selectbox(
            "What field are you studying?",
            fields,
            index=0,
            help="Select your academic field"
        )

        def set_example(text: str):
            st.session_state["input_text"] = text

        problem_text = st.text_area(
            f"Describe your {field} problem or challenge:",
            placeholder=f"e.g., 'I'm struggling with algorithms in {field}' or 'This concept is confusing'",
            height=120,
            key="input_text"
        )

        st.markdown("**Quick Examples:**")
        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            st.button(
                "😕 Confused about recursion",
                use_container_width=True,
                on_click=set_example,
                args=("I'm confused about recursion and how base cases return.",)
            )
        with ex2:
            st.button(
                "😤 Debugging is frustrating",
                use_container_width=True,
                on_click=set_example,
                args=("Debugging this bug is so frustrating, nothing works no matter what I try.",)
            )
        with ex3:
            st.button(
                "🧐 Curious about ML",
                use_container_width=True,
                on_click=set_example,
                args=("I'm curious about machine learning and how neural networks learn.",)
            )

        ex4, ex5, ex6 = st.columns(3)
        with ex4:
            st.button(
                "💪 Solved problems easily",
                use_container_width=True,
                on_click=set_example,
                args=("I solved all the practice problems easily and feel very confident about this chapter!",)
            )
        with ex5:
            st.button(
                "😐 Lecture is repetitive",
                use_container_width=True,
                on_click=set_example,
                args=("This review lecture is repetitive and boring, I already know all of this material.",)
            )
        with ex6:
            st.button(
                "🎭 Fascinating but tired",
                use_container_width=True,
                on_click=set_example,
                args=("Ohh! This concept seems fascinating but now I am tired and stuck on details.",)
            )

    with col2:
        st.markdown("#### ⚙️ Settings")
        use_ai = st.checkbox("Use AI Response (Gemini)", value=True)
        save_data = st.checkbox("Save to Database / CSV", value=True)
        show_details = st.checkbox("Show analysis details", value=True)

        st.markdown("---")
        st.markdown("#### 📊 Predict from Saved Data")
        use_csv_prediction = st.checkbox("Use CSV-based prediction", value=False)
        if use_csv_prediction and examples_count > 0:
            st.info(f"Using {examples_count} saved examples for prediction")

    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    # Primary Action Button
    if st.button("🔍 Get AI Learning Help", type="primary", use_container_width=True):
        active_text = problem_text.strip()
        if not active_text:
            st.warning("⚠️ Please describe your problem or select a quick example above.")
        else:
            with st.spinner("Analyzing emotion & generating personalized guidance..."):
                t0 = time.time()
                
                # 1. Inference BiLSTM
                bilstm_res = bilstm_model.predict(active_text) if bilstm_model else {
                    "emotion": "Confused", "confidence": 0.5,
                    "scores": {"Bored": 0.1, "Confident": 0.1, "Confused": 0.6, "Curious": 0.1, "Frustrated": 0.1},
                    "cleaned_text": active_text
                }
                
                # 2. Inference MiniLM BERT
                bert_res = bert_model.predict(active_text) if bert_model else None

                primary_emotion = bilstm_res["emotion"]
                primary_confidence = bilstm_res["confidence"]

                # 3. Response generation
                ai_resp = get_gemini_response(
                    field=field,
                    problem=active_text,
                    emotion=primary_emotion,
                    confidence=primary_confidence,
                    use_ai=use_ai,
                    persona="Socratic Mentor"
                )

                # 4. Database & CSV Persistence
                current_user_id = st.session_state.user["username"] if st.session_state.user else "guest_student"
                if save_data:
                    log_student_interaction(
                        user_id=current_user_id,
                        field=field,
                        problem_text=active_text,
                        emotion=primary_emotion,
                        confidence=primary_confidence,
                        scores=bilstm_res["scores"],
                        response=ai_resp,
                        model="BiLSTM"
                    )

                # 5. History logging
                add_to_history(
                    field, active_text, primary_emotion, primary_confidence,
                    ai_resp, bilstm_res["scores"], bert_res
                )

                latency = (time.time() - t0) * 1000

                st.session_state.last_analysis = {
                    "field": field,
                    "problem": active_text,
                    "bilstm_result": bilstm_res,
                    "bert_result": bert_res,
                    "ai_response": ai_resp,
                    "use_ai": use_ai,
                    "latency": latency,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

    # Results & Model Comparison Section
    if st.session_state.last_analysis is not None:
        res = st.session_state.last_analysis
        b_res = res["bilstm_result"]
        bert_res = res["bert_result"]
        p_emotion = b_res["emotion"]
        p_conf = b_res["confidence"]
        strategy_text = EMOTION_RESPONSES.get(p_emotion, {}).get("action", "Provide step-by-step clarity")

        st.markdown("---")
        st.markdown("### 🔬 Model Predictions Comparison")

        col_m1, col_m2 = st.columns(2, gap="large")

        # Left Column: BiLSTM
        with col_m1:
            st.markdown("##### BiLSTM Student Adaptive")
            st.caption("Mixed Emotions")
            
            b_mixed = get_mixed_emotions(b_res["scores"])
            if len(b_mixed) > 1:
                mixed_header = " + ".join([f"{EMOTION_RESPONSES.get(em[0], {}).get('emoji', '🎯')} {em[0]}" for em in b_mixed])
                st.markdown(f"<div class='mixed-emotion-title'>{mixed_header}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='primary-metric-sub'>↑ Primary: {b_mixed[0][1]:.1%}</div>", unsafe_allow_html=True)
            else:
                emoji = EMOTION_RESPONSES.get(p_emotion, {}).get("emoji", "🎯")
                st.markdown(f"<div class='mixed-emotion-title'>{emoji} {p_emotion}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='primary-metric-sub'>↑ Primary: {p_conf:.1%}</div>", unsafe_allow_html=True)

            for em_name, em_val in sorted(b_res["scores"].items(), key=lambda x: x[1], reverse=True):
                st.caption(f"{em_name}: {em_val:.1%}")
                st.progress(float(em_val))

        # Right Column: BERT
        with col_m2:
            st.markdown("##### BERT Transformer")
            st.caption("Mixed Emotions")

            if bert_res:
                t_mixed = get_mixed_emotions(bert_res["scores"])
                if len(t_mixed) > 1:
                    t_mixed_header = " + ".join([f"{EMOTION_RESPONSES.get(em[0], {}).get('emoji', '🎯')} {em[0]}" for em in t_mixed])
                    st.markdown(f"<div class='mixed-emotion-title'>{t_mixed_header}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='primary-metric-sub'>↑ Primary: {t_mixed[0][1]:.1%}</div>", unsafe_allow_html=True)
                else:
                    t_emoji = EMOTION_RESPONSES.get(bert_res["emotion"], {}).get("emoji", "🎯")
                    st.markdown(f"<div class='mixed-emotion-title'>{t_emoji} {bert_res['emotion']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='primary-metric-sub'>↑ Primary: {bert_res['confidence']:.1%}</div>", unsafe_allow_html=True)

                for em_name, em_val in sorted(bert_res["scores"].items(), key=lambda x: x[1], reverse=True):
                    st.caption(f"{em_name}: {em_val:.1%}")
                    st.progress(float(em_val))
            else:
                st.info("BERT model loaded on-demand.")

        # AI Learning Assistant Response Card
        st.markdown("---")
        st.markdown("### 🤖 AI Learning Assistant Response")
        st.markdown(f"""
        <div class="response-container">
            <div class="response-header-chip">
                💡 AI Response based on BiLSTM prediction: {p_emotion}
            </div>
            <div style="font-size: 1.02rem; line-height: 1.6; color: #F8FAFC; margin-bottom: 20px;">
                {res['ai_response']}
            </div>
            <div style="margin-top: 14px;">
                <h4 style="color: #FFFFFF; margin-bottom: 4px;">📖 Additional Support</h4>
                <div class="strategy-pill">Strategy: {strategy_text}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Analysis Details Expander
        if show_details:
            with st.expander("🔍 Analysis Details", expanded=False):
                st.markdown(f"**Original Problem:** {res['problem']}")
                st.markdown(f"**BiLSTM Processed:** {b_res['cleaned_text']}")
                st.markdown(f"**BiLSTM Confidence:** {b_res['confidence']:.3f}")
                st.markdown(f"**AI Model:** {'Gemini 2.5 Flash' if res['use_ai'] else 'Offline Fallback'}")
                st.markdown(f"**Inference Latency:** `{res['latency']:.1f} ms`")
                st.markdown(f"**Timestamp:** {res['timestamp']}")


# ============================================
# TAB 2: ANALYTICS & JOURNEY
# ============================================
with main_tab2:
    st.markdown("### 📊 Learning Analytics & Emotional Journey")
    
    if st.session_state.emotion_history:
        df_history = pd.DataFrame(st.session_state.emotion_history)

        sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Emotions Breakdown", "Study Fields", "Interaction Logs"])

        # Sub Tab 1: Emotions
        with sub_tab1:
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                emotion_counts = df_history["emotion"].value_counts()
                fig1 = px.pie(
                    values=emotion_counts.values,
                    names=emotion_counts.index,
                    title="Overall Emotion Distribution",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig1.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig1, use_container_width=True)

            with col_c2:
                df_copy = df_history.copy()
                df_copy["time"] = df_copy["timestamp"].apply(
                    lambda t: t.strftime("%H:%M:%S") if hasattr(t, "strftime") else str(t)[11:19]
                )
                fig2 = px.line(
                    df_copy,
                    x="time",
                    y="confidence",
                    color="emotion",
                    markers=True,
                    title="Confidence & Emotional Trajectory",
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig2.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(tickformat=".0%"),
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig2, use_container_width=True)

        # Sub Tab 2: Fields
        with sub_tab2:
            if "model" in df_history.columns and df_history["model"].nunique() > 1:
                field_emotion = df_history.groupby(["field", "emotion", "model"]).size().reset_index(name="count")
                fig3 = px.bar(
                    field_emotion,
                    x="field",
                    y="count",
                    color="emotion",
                    facet_col="model",
                    title="Emotions by Academic Field & Model",
                    color_discrete_sequence=px.colors.qualitative.Vivid
                )
            else:
                field_emotion = df_history.groupby(["field", "emotion"]).size().reset_index(name="count")
                fig3 = px.bar(
                    field_emotion,
                    x="field",
                    y="count",
                    color="emotion",
                    title="Emotions by Academic Field",
                    color_discrete_sequence=px.colors.qualitative.Vivid
                )

            fig3.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig3, use_container_width=True)

        # Sub Tab 3: Logs Table & Summary Metrics
        with sub_tab3:
            s1, s2, s3 = st.columns(3)
            s1.metric("Total Session Queries", len(df_history))
            s2.metric("Dominant Emotion", df_history["emotion"].mode()[0] if not df_history.empty else "N/A")
            s3.metric("Average Confidence", f"{df_history['confidence'].mean():.1%}")

            st.markdown("#### Detailed Interaction Records")
            cols_to_show = [c for c in ["timestamp", "field", "emotion", "confidence", "model", "problem"] if c in df_history.columns]
            st.dataframe(df_history[cols_to_show], use_container_width=True)
    else:
        st.info("💡 No interactions recorded yet in this session. Try asking a question or clicking a quick example in the Learning Assistant tab!")


# ============================================
# TAB 3: STUDENT FEEDBACK FORM
# ============================================
with main_tab3:
    st.markdown("### 📝 Student Feedback & Model Improvement")
    st.markdown("Your feedback helps our AI pedagogical model adapt and understand learning challenges better.")

    with st.form("student_feedback_form"):
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            feedback_rating = st.slider("⭐ How helpful was the AI explanation?", min_value=1, max_value=5, value=5)
            feedback_helpful = st.radio("Did this guidance help resolve your learning doubt?", ["Yes, completely", "Partially", "No, still stuck"], horizontal=True)

        with col_f2:
            feedback_emotion_acc = st.selectbox(
                "Did the model detect your emotional state correctly?",
                ["Yes, accurately detected", "Somewhat accurate", "No, I felt differently"]
            )
            feedback_category = st.selectbox(
                "Feedback Category",
                ["Pedagogical Guidance Quality", "Emotion Accuracy", "UI & Usability", "Concept Explanation", "Other"]
            )

        feedback_comments = st.text_area(
            "Additional Comments or Suggestions (Optional):",
            placeholder="Tell us what worked well or what we can improve in the explanation..."
        )

        submit_feedback = st.form_submit_button("🚀 Submit Feedback", type="primary")

        if submit_feedback:
            current_user = st.session_state.user["username"] if st.session_state.user else "guest_student"
            is_helpful_bool = True if feedback_helpful.startswith("Yes") else False
            
            saved_to_db, fb_msg = log_student_feedback(
                user_id=current_user,
                rating=feedback_rating,
                was_helpful=is_helpful_bool,
                emotion_accurate=feedback_emotion_acc,
                comments=feedback_comments,
                category=feedback_category
            )
            
            if saved_to_db:
                st.success("🎉 Thank you! Your feedback has been saved to the database.")
            else:
                st.success(f"🎉 {fb_msg}")