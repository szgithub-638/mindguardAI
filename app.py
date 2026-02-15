import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from fpdf import FPDF
import io

# -------------------------------
# Load Emotion Model (Local AI)
# -------------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )

classifier = load_model()

# -------------------------------
# Coping tips dictionary
# -------------------------------
coping_tips = {
    "sadness": [
        "Take a 5-minute break and breathe deeply.",
        "Talk to a friend or a trusted adult.",
        "Write down your thoughts in a journal."
    ],
    "fear": [
        "Practice grounding techniques (5 senses exercise).",
        "List what is in your control.",
        "Take slow, deep breaths."
    ],
    "anger": [
        "Take a short walk to release tension.",
        "Try writing your feelings in a notebook.",
        "Count to 10 before reacting."
    ],
    "joy": [
        "Share your happiness with someone you care about.",
        "Keep a gratitude journal.",
        "Celebrate small wins!"
    ],
    "surprise": [
        "Take a moment to process the situation.",
        "Reflect on what this means for you.",
        "Share your thoughts with someone you trust."
    ],
    "neutral": [
        "Reflect on your day and plan one positive activity.",
        "Take a short break to reset your focus."
    ]
}

# -------------------------------
# Crisis keywords
# -------------------------------
crisis_keywords = ["suicide", "hopeless", "unsafe", "end my life", "self-harm"]

# -------------------------------
# Initialize Session State
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "entries" not in st.session_state:
    st.session_state.entries = []

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="MindGuard AI", page_icon="💜", layout="wide")
st.markdown("# 💜 MindGuard AI")
st.markdown("### AI-Powered Teen Emotional Risk Detection & Support System")
st.markdown("---")

st.markdown("## 📝 Daily Reflection")
user_input = st.text_area(
    "Describe how you're feeling today:",
    height=150,
    placeholder="Example: I feel overwhelmed with school and haven't been sleeping well..."
)

# -------------------------------
# Analyze Button
# -------------------------------
if st.button("Analyze"):

    if user_input.strip() == "":
        st.warning("Please enter how you're feeling.")
    else:
        # Check crisis keywords first
        crisis_flag = any(word in user_input.lower() for word in crisis_keywords)

        results = classifier(user_input)
        emotion_scores = results[0]
        top_emotion = max(emotion_scores, key=lambda x: x["score"])
        emotion = top_emotion["label"].lower()
        confidence = top_emotion["score"]

        # Calculate risk
        if crisis_flag:
            risk_score = 10
        elif emotion in ["sadness", "fear", "anger"]:
            risk_score = int(confidence * 10)
        else:
            risk_score = int((1 - confidence) * 5)

        # -------------------------------
        # Display Analysis
        # -------------------------------
        st.markdown("## 🧠 Emotional Analysis")
        st.write(f"**Primary Emotion:** {emotion.capitalize()}")
        st.write(f"**Confidence:** {round(confidence * 100, 2)}%")
        st.markdown(f"### 🔢 Risk Level: {risk_score}/10")

        # Color-coded message
        if crisis_flag or risk_score >= 8:
            st.error("⚠ Emotional distress detected. Consider reaching out to a trusted adult or counselor.")
        elif risk_score >= 5:
            st.warning("⚠ Moderate risk. Practice coping strategies or talk to someone you trust.")
        else:
            st.success("💚 Emotional state appears stable.")

        # -------------------------------
        # Coping Tips
        # -------------------------------
        tips = coping_tips.get(emotion, ["Take a break and focus on self-care."])
        st.markdown("## 🛠 Suggested Coping Tips")
        for tip in tips:
            st.write(f"- {tip}")

        # Save to session
        st.session_state.history.append(risk_score)
        st.session_state.entries.append(user_input)

# -------------------------------
# Trend Graph + History Table
# -------------------------------
if len(st.session_state.history) > 0:
    st.markdown("## 📊 Emotional Trend Over Time")

    df = pd.DataFrame({
        "Entry": st.session_state.entries,
        "Risk Level": st.session_state.history
    })

    # Graph
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df["Risk Level"], marker='o', color='purple')
    ax.set_ylabel("Risk Level")
    ax.set_xlabel("Entries")
    ax.set_ylim(0, 10)
    ax.grid(True)
    st.pyplot(fig)

    # Table
    st.markdown("### 📝 Reflection History")
    st.dataframe(df)

    # -------------------------------
    # Export PDF Button
    # -------------------------------
    if st.button("Export PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "💜 MindGuard AI - Reflection Report", ln=True, align="C")
        pdf.ln(10)

        pdf.set_font("Arial", '', 12)
        for i, (entry, risk) in enumerate(zip(st.session_state.entries, st.session_state.history), start=1):
            pdf.multi_cell(0, 8, f"{i}. {entry}\nRisk Level: {risk}/10\n")
            pdf.ln(2)

        # Save matplotlib figure to image buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        pdf.image(buf, x=10, y=None, w=pdf.w - 20)
        buf.close()

        pdf_output = "MindGuard_Report.pdf"
        pdf.output(pdf_output)

        st.success(f"PDF report saved as {pdf_output}! ✅")
