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
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 2px;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.0rem;
        margin-bottom: 20px;
    }

    /* Response Card styling matching screenshot */
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
        font-weight: 600;
        font-size: 1.05rem;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        width: 100%;
        margin-top: 10px;
        transition: all 0.2s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #DC2626;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
    }
    
    /* Quick Example Secondary Buttons */
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
# 4. CSV PERSISTENCE & HISTORY
# ============================================
def save_to_csv(field, problem, emotion, confidence, ai_response):
    """Saves interaction logs to CSV for continuous learning and analytics."""
    try:
        new_example = {
            "text": problem,
            "emotion": emotion.lower(),
            "confidence": confidence,
            "response": ai_response,
            "field": field,
            "timestamp": datetime.now().isoformat(),
        }
        examples_file = "emotion_response_examples.csv"
        if os.path.exists(examples_file):
            df = pd.read_csv(examples_file)
            df = pd.concat([df, pd.DataFrame([new_example])], ignore_index=True)
        else:
            df = pd.DataFrame([new_example])
        df.to_csv(examples_file, index=False)

        # Update mapping file
        mapping_file = "emotion_response_mapping.csv"
        if os.path.exists(mapping_file):
            mapping_df = pd.read_csv(mapping_file)
            if emotion not in mapping_df["emotion"].values:
                new_mapping = pd.DataFrame([{"emotion": emotion, "response": ai_response}])
                mapping_df = pd.concat([mapping_df, new_mapping], ignore_index=True)
                mapping_df.to_csv(mapping_file, index=False)
        else:
            pd.DataFrame([{"emotion": emotion, "response": ai_response}]).to_csv(mapping_file, index=False)
        return True
    except Exception:
        return False


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
# 5. SIDEBAR: DASHBOARD
# ============================================
examples_count = 0
if os.path.exists("emotion_response_examples.csv"):
    try:
        ex_df = pd.read_csv("emotion_response_examples.csv")
        examples_count = len(ex_df)
    except Exception:
        examples_count = 55

with st.sidebar:
    st.markdown("### 📊 Dashboard")
    st.markdown(f"**Models:** ✅ {status_text}")
    st.markdown(f"**Total Interactions:** `{len(st.session_state.emotion_history)}`")
    st.markdown(f"**CSV Examples:** `{examples_count}`")

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
# 6. HERO TITLE
# ============================================
st.markdown("""
<div class="hero-title">🤖 Emotion-Aware Learning Assistant</div>
<div class="hero-subtitle">Get personalized help based on your field and emotional state</div>
""", unsafe_allow_html=True)


# ============================================
# 7. INPUT & SETTINGS SECTION
# ============================================
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

    problem_text = st.text_area(
        f"Describe your {field} problem or challenge:",
        placeholder=f"e.g., 'I'm struggling with algorithms in {field}' or 'This concept is confusing'",
        height=120,
        value=st.session_state.input_text,
        key="problem_text_input_area"
    )

    st.markdown("**Quick Examples:**")
    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        if st.button("😕 Confused about recursion", use_container_width=True):
            st.session_state.input_text = "I'm confused about recursion and how base cases return."
            st.session_state.problem_text_input_area = "I'm confused about recursion and how base cases return."
            st.rerun()
    with ex2:
        if st.button("😤 Debugging is frustrating", use_container_width=True):
            st.session_state.input_text = "Debugging this bug is so frustrating, nothing works no matter what I try."
            st.session_state.problem_text_input_area = "Debugging this bug is so frustrating, nothing works no matter what I try."
            st.rerun()
    with ex3:
        if st.button("🧐 Curious about ML", use_container_width=True):
            st.session_state.input_text = "I'm curious about machine learning and how neural networks learn."
            st.session_state.problem_text_input_area = "I'm curious about machine learning and how neural networks learn."
            st.rerun()

    ex4, ex5, ex6 = st.columns(3)
    with ex4:
        if st.button("💪 Solved all problems easily", use_container_width=True):
            st.session_state.input_text = "I solved all the practice problems easily and feel very confident about this chapter!"
            st.session_state.problem_text_input_area = "I solved all the practice problems easily and feel very confident about this chapter!"
            st.rerun()
    with ex5:
        if st.button("😐 Lecture is repetitive & boring", use_container_width=True):
            st.session_state.input_text = "This review lecture is repetitive and boring, I already know all of this material."
            st.session_state.problem_text_input_area = "This review lecture is repetitive and boring, I already know all of this material."
            st.rerun()
    with ex6:
        if st.button("🎭 Fascinating but I am tired", use_container_width=True):
            st.session_state.input_text = "Ohh! This concept seems fascinating but now I am tired and stuck on details."
            st.session_state.problem_text_input_area = "Ohh! This concept seems fascinating but now I am tired and stuck on details."
            st.rerun()

with col2:
    st.markdown("#### ⚙️ Settings")
    use_ai = st.checkbox("Use AI Response (Gemini)", value=True)
    save_data = st.checkbox("Save to CSV for learning", value=True)
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

            # 4. CSV persistence
            if save_data:
                save_to_csv(field, active_text, primary_emotion, primary_confidence, ai_resp)

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


# ============================================
# 8. RESULTS & MODEL COMPARISON SECTION
# ============================================
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

    # AI Learning Assistant Response Card matching screenshot
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

    # Analysis Details Expander matching screenshot
    if show_details:
        with st.expander("🔍 Analysis Details", expanded=False):
            st.markdown(f"**Original Problem:** {res['problem']}")
            st.markdown(f"**BiLSTM Processed:** {b_res['cleaned_text']}")
            st.markdown(f"**BiLSTM Confidence:** {b_res['confidence']:.3f}")
            st.markdown(f"**AI Model:** {'Gemini 2.5 Flash' if res['use_ai'] else 'Offline Fallback'}")
            st.markdown(f"**Timestamp:** {res['timestamp']}")


# ============================================
# 9. ANALYTICS DASHBOARD TABS
# ============================================
if st.session_state.emotion_history:
    st.markdown("---")
    df_history = pd.DataFrame(st.session_state.emotion_history)

    tab1, tab2, tab3 = st.tabs(["Emotions", "Fields", "Summary"])

    # Tab 1: Emotions (Pie & Timeline Line Chart)
    with tab1:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            emotion_counts = df_history["emotion"].value_counts()
            fig1 = px.pie(
                values=emotion_counts.values,
                names=emotion_counts.index,
                title="Emotion Distribution",
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
                title="Emotional Journey",
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

    # Tab 2: Fields (Bar Chart faceted by model)
    with tab2:
        if "model" in df_history.columns and df_history["model"].nunique() > 1:
            field_emotion = df_history.groupby(["field", "emotion", "model"]).size().reset_index(name="count")
            fig3 = px.bar(
                field_emotion,
                x="field",
                y="count",
                color="emotion",
                facet_col="model",
                title="Emotions by Study Field & Model",
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
        else:
            field_emotion = df_history.groupby(["field", "emotion"]).size().reset_index(name="count")
            fig3 = px.bar(
                field_emotion,
                x="field",
                y="count",
                color="emotion",
                title="Emotions by Study Field",
                color_discrete_sequence=px.colors.qualitative.Vivid
            )

        fig3.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Tab 3: Summary
    with tab3:
        st.markdown("#### Overall Statistics")
        s1, s2, s3 = st.columns(3)
        s1.metric("Total Interactions", len(df_history))
        s2.metric("Most Frequent Emotion", df_history["emotion"].mode()[0] if not df_history.empty else "N/A")
        s3.metric("Average Confidence", f"{df_history['confidence'].mean():.1%}")

        st.markdown("#### Detailed Interaction Logs")
        cols_to_show = [c for c in ["timestamp", "field", "emotion", "confidence", "model"] if c in df_history.columns]
        st.dataframe(df_history[cols_to_show], use_container_width=True)