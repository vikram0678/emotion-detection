import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.express as px

from mixed_emotion import get_mixed_emotions, EMOTION_RESPONSES
from gemini_helper import get_gemini_response

st.set_page_config(page_title="AI Learning Assistant", page_icon="🤖", layout="wide")

# Initialize session state
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []


# ============================================
# CACHED MODEL LOADING
# ============================================
@st.cache_resource
def load_models():
    try:
        from bilstm_predictor import EmotionPredictor
        bilstm_model = EmotionPredictor(
            model_path="models/bltsm/bilstm_student_adaptive.keras",
            tokenizer_path="models/bltsm/tokenizer.pickle",
            label_encoder_path="models/bltsm/label_encoder.pickle",
        )

        bert_model = None
        try:
            from bert_model import BERTEmotionClassifier
            bert_model = BERTEmotionClassifier()
            bert_model.load_model("models/bert_student_adaptive")
        except Exception:
            pass

        return bilstm_model, bert_model, "✅ Models loaded"
    except Exception as e:
        return None, None, f"❌ Error: {e}"


# ============================================
# CSV PERSISTENCE
# ============================================
def save_to_csv(field, problem, emotion, confidence, ai_response):
    try:
        new_example = {
            "text": problem,
            "emotion": emotion.lower(),
            "confidence": confidence,
            "response": ai_response,
            "field": field,
            "timestamp": datetime.now().isoformat(),
        }

        if os.path.exists("emotion_response_examples.csv"):
            df = pd.read_csv("emotion_response_examples.csv")
            df = pd.concat([df, pd.DataFrame([new_example])], ignore_index=True)
        else:
            df = pd.DataFrame([new_example])
        df.to_csv("emotion_response_examples.csv", index=False)

        if os.path.exists("emotion_response_mapping.csv"):
            mapping_df = pd.read_csv("emotion_response_mapping.csv")
            if emotion not in mapping_df["emotion"].values:
                new_mapping = pd.DataFrame([{"emotion": emotion, "response": ai_response}])
                mapping_df = pd.concat([mapping_df, new_mapping], ignore_index=True)
                mapping_df.to_csv("emotion_response_mapping.csv", index=False)

        return True
    except Exception as e:
        st.error(f"Failed to save to CSV: {e}")
        return False


# ============================================
# SESSION HISTORY MANAGEMENT
# ============================================
def add_to_history(field, problem, emotion, confidence, ai_response, bilstm_scores, bert_result=None):
    mixed_emotions = get_mixed_emotions(bilstm_scores)
    emotion_label = " + ".join([em[0] for em in mixed_emotions]) if len(mixed_emotions) > 1 else emotion

    st.session_state.emotion_history.append({
        "timestamp": datetime.now(),
        "field": field,
        "problem": problem,
        "emotion": emotion_label,
        "confidence": confidence,
        "ai_response": ai_response,
        "all_scores": bilstm_scores,
        "model": "BiLSTM",
    })

    if bert_result:
        bert_mixed = get_mixed_emotions(bert_result["scores"])
        bert_emotion_label = " + ".join([em[0] for em in bert_mixed]) if len(bert_mixed) > 1 else bert_result["emotion"]

        st.session_state.emotion_history.append({
            "timestamp": datetime.now(),
            "field": field,
            "problem": problem,
            "emotion": bert_emotion_label,
            "confidence": bert_result["confidence"],
            "ai_response": ai_response,
            "all_scores": bert_result["scores"],
            "model": "BERT",
        })


def main():
    bilstm_model, bert_model, status = load_models()

    if bilstm_model is None:
        st.error(status)
        return

    # Load CSV examples once, reused in sidebar + settings panel
    if os.path.exists("emotion_response_examples.csv"):
        examples_df = pd.read_csv("emotion_response_examples.csv")
    else:
        examples_df = pd.DataFrame()

    # ============================================
    # SIDEBAR DASHBOARD
    # ============================================
    with st.sidebar:
        st.header("📊 Dashboard")
        st.write(f"Models: {status}")
        st.write(f"Total Interactions: {len(st.session_state.emotion_history)}")
        st.write(f"CSV Examples: {len(examples_df)}")

        if st.button("🗑️ Clear History"):
            st.session_state.emotion_history = []
            st.rerun()

        if st.session_state.emotion_history:
            st.subheader("Recent Sessions")
            recent = st.session_state.emotion_history[-3:]
            for item in reversed(recent):
                st.write(f"• {item['field']}: {item['emotion']} ({item['confidence']:.1%})")

    st.title("🎓 AI Learning Assistant")
    st.write("Tell us what you're working on, and we'll detect how you're feeling and offer guidance.")

    # ============================================
    # INPUT (col1) + SETTINGS (col2)
    # ============================================
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("📝 Tell us about your learning challenge")

        field = st.selectbox(
            "What field are you studying?",
            ["Computer Science", "General", "Mathematics", "Physics", "Chemistry", "Biology",
             "Engineering", "Business", "Literature", "History", "Psychology", "Other"],
            help="Select your area of study for personalized responses",
        )

        # problem = st.text_area(
        #     f"Describe your {field} problem or challenge:",
        #     placeholder=f"e.g., 'I'm struggling with algorithms in {field}' or 'This concept is confusing'",
        #     height=120,
        # )
        if "example_text" not in st.session_state:
            st.session_state.example_text = ""

        problem = st.text_area(
            f"Describe your {field} problem or challenge:",
            placeholder=f"e.g., 'I'm struggling with algorithms in {field}' or 'This concept is confusing'",
            height=120,
            value=st.session_state.example_text,
            key="problem_input",
        )

        st.write("**Quick Examples:**")
        ex1, ex2, ex3 = st.columns(3)

        with ex1:
            if st.button("I'm confused about recursion", use_container_width=True):
                st.session_state.example_text = "I'm confused about recursion"
                st.rerun()
        with ex2:
            if st.button("Debugging is frustrating", use_container_width=True):
                st.session_state.example_text = "Debugging is frustrating"
                st.rerun()
        with ex3:
            if st.button("I'm curious about machine learning", use_container_width=True):
                st.session_state.example_text = "I'm curious about machine learning"
                st.rerun()

    with col2:
        st.subheader("⚙️ Settings")
        use_ai = st.checkbox("Use AI Response (Gemini)", value=True)
        save_data = st.checkbox("Save to CSV for learning", value=True)
        show_details = st.checkbox("Show analysis details", value=False)

        # CSV Prediction Option
        st.markdown("---")
        st.write("**📊 Predict from Saved Data**")
        use_csv_prediction = st.checkbox("Use CSV-based prediction", value=False)

        if use_csv_prediction and len(examples_df) > 0:
            st.info(f"Using {len(examples_df)} saved examples for prediction")

    # ============================================
    # ANALYSIS BUTTON
    # ============================================
    if st.button("🔍 Get AI Learning Help", type="primary", use_container_width=True):
        if problem.strip():
            with st.spinner("Analyzing your learning state..."):
                bilstm_result = bilstm_model.predict(problem)
                bert_result = bert_model.predict(problem) if bert_model else None

                emotion_result = bilstm_result
                emotion = emotion_result["emotion"]
                confidence = emotion_result["confidence"]

                if use_ai:
                    ai_response = get_gemini_response(field, problem, emotion, confidence, use_ai=True)
                else:
                    ai_response = EMOTION_RESPONSES[emotion]["response"]

                if save_data:
                    save_success = save_to_csv(field, problem, emotion, confidence, ai_response)
                    if save_success:
                        st.success("💾 Interaction saved to improve future responses!")

                add_to_history(field, problem, emotion, confidence, ai_response,
                                bilstm_result["scores"], bert_result)

            # --- Model Predictions Comparison ---
            if show_details:
                st.subheader("🎭 Model Predictions Comparison")

                if bert_result:
                    dcol1, dcol2 = st.columns(2)
                else:
                    dcol1 = st.columns(1)[0]

                with dcol1:
                    st.write("**BiLSTM Student Adaptive**")
                    bilstm_mixed = get_mixed_emotions(bilstm_result["scores"])

                    if len(bilstm_mixed) > 1:
                        mixed_text = " + ".join(
                            [f"{EMOTION_RESPONSES[em[0]]['emoji']} {em[0]}" for em in bilstm_mixed]
                        )
                        st.metric("Mixed Emotions", mixed_text, f"Primary: {bilstm_mixed[0][1]:.1%}")
                    else:
                        bilstm_emoji = EMOTION_RESPONSES[bilstm_result["emotion"]]["emoji"]
                        st.metric("Emotion", f"{bilstm_emoji} {bilstm_result['emotion']}", f"{bilstm_result['confidence']:.1%}")

                    for emotion_name, score in sorted(bilstm_result["scores"].items(), key=lambda x: x[1], reverse=True):
                        st.progress(score, text=f"{emotion_name}: {score:.1%}")

                if bert_result:
                    with dcol2:
                        st.write("**BERT Transformer**")
                        bert_mixed = get_mixed_emotions(bert_result["scores"])

                        if len(bert_mixed) > 1:
                            mixed_text = " + ".join(
                                [f"{EMOTION_RESPONSES[em[0]]['emoji']} {em[0]}" for em in bert_mixed]
                            )
                            st.metric("Mixed Emotions", mixed_text, f"Primary: {bert_mixed[0][1]:.1%}")
                        else:
                            bert_emoji = EMOTION_RESPONSES[bert_result["emotion"]]["emoji"]
                            st.metric("Emotion", f"{bert_emoji} {bert_result['emotion']}", f"{bert_result['confidence']:.1%}")

                        for emotion_name, score in sorted(bert_result["scores"].items(), key=lambda x: x[1], reverse=True):
                            st.progress(score, text=f"{emotion_name}: {score:.1%}")
            else:
                emoji = EMOTION_RESPONSES[emotion]["emoji"]
                st.metric("Detected Emotion", f"{emoji} {emotion}", f"{confidence:.1%}")

        #     st.divider()
        #     st.write("**💬 Learning Guidance**")
        #     st.info(ai_response)
        # else:
        #     st.warning("Please describe your problem first.")
        
            st.divider()
            st.write("**💬 Learning Guidance**")
            st.info(f"AI Response based on BiLSTM prediction: {emotion}")
            st.write(ai_response)

            st.write("**📖 Additional Support**")
            st.info(f"**Strategy:** {EMOTION_RESPONSES[emotion]['action']}")

            with st.expander("🔍 Analysis Details"):
                st.write(f"**Original Problem:** {problem}")
                st.write(f"**BiLSTM Processed:** {bilstm_result['cleaned_text']}")
                st.write(f"**BiLSTM Confidence:** {bilstm_result['confidence']:.3f}")
                st.write(f"**AI Model:** {'Gemini 2.5 Flash' if use_ai else 'Template (AI disabled)'}")
                st.write(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.warning("Please describe your problem first.")

    # ============================================
    # ANALYTICS DASHBOARD (tabs)
    # ============================================
    # ============================================
    # ANALYTICS DASHBOARD (tabs, Plotly charts)
    # ============================================
    if st.session_state.emotion_history:
        st.markdown("---")
        st.header("📈 Learning Analytics")

        df = pd.DataFrame(st.session_state.emotion_history)

        tab1, tab2, tab3 = st.tabs(["Emotions", "Fields", "Summary"])

        with tab1:
            col1, col2 = st.columns(2)

            with col1:
                emotion_counts = df["emotion"].value_counts()
                fig1 = px.pie(values=emotion_counts.values, names=emotion_counts.index,
                              title="Emotion Distribution")
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                df_copy = df.copy()
                df_copy["time"] = df_copy["timestamp"].dt.strftime("%H:%M:%S")
                fig2 = px.line(df_copy, x="time", y="confidence", color="emotion",
                               title="Emotional Journey", markers=True)
                st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            if "model" in df.columns:
                field_emotion = df.groupby(["field", "emotion", "model"]).size().reset_index(name="count")
                fig3 = px.bar(field_emotion, x="field", y="count", color="emotion", facet_col="model",
                              title="Emotions by Study Field & Model")
            else:
                field_emotion = df.groupby(["field", "emotion"]).size().reset_index(name="count")
                fig3 = px.bar(field_emotion, x="field", y="count", color="emotion",
                              title="Emotions by Study Field")
            st.plotly_chart(fig3, use_container_width=True)

        with tab3:
            st.subheader("Overall Statistics")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Interactions", len(df))
            c2.metric("Most Common Emotion", df["emotion"].mode()[0] if not df.empty else "N/A")
            c3.metric("Avg Confidence", f"{df['confidence'].mean():.1%}")

            st.subheader("Raw History Table")
            st.dataframe(df[["timestamp", "field", "emotion", "confidence", "model"]], use_container_width=True)
        
if __name__ == "__main__":
    main()