import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import numpy as np

st.header("PROJECT BY OYSCATECH")
st.title("🧠 Smart Essay Scoring System (NLP + ML)")


essay = st.text_area("✍️ Enter your essay:", height=250)

# -----------------------------
# SAMPLE TRAINING DATA (SIMULATION)
# -----------------------------
train_essays = [
    "This is a very good essay with proper grammar and structure.",
    "Bad essay no structure poor grammar",
    "Excellent work with strong arguments and clear meaning.",
    "Average writing some mistakes present.",
    "Poorly written essay lacking clarity and coherence."
]

train_scores = [9, 3, 10, 6, 2]

# -----------------------------
# TRAIN MODEL
# -----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(train_essays)

model = LinearRegression()
model.fit(X, train_scores)

# -----------------------------
# NLP FEATURES
# -----------------------------
def grammar_check(text):
    sentences = re.split(r'[.!?]', text)
    good_sentences = 0

    for s in sentences:
        if len(s.strip().split()) > 3:
            good_sentences += 1

    return good_sentences


def coherence_check(text):
    sentences = re.split(r'[.!?]', text)
    return len(sentences)


def vocabulary_score(text):
    words = text.split()
    return len(set(words))


# -----------------------------
# BUTTON
# -----------------------------
if st.button("🚀 Analyze Essay"):
    if essay.strip() == "":
        st.warning("Please enter an essay.")
    else:
        # NLP FEATURES
        grammar = grammar_check(essay)
        coherence = coherence_check(essay)
        vocab = vocabulary_score(essay)

        # ML PREDICTION
        input_vector = vectorizer.transform([essay])
        prediction = model.predict(input_vector)[0]

        final_score = min(max(int(prediction), 0), 10)

        # -----------------------------
        # OUTPUT
        # -----------------------------
        st.subheader("📊 Results")

        st.write(f"🤖 AI Score: {final_score}/10")
        st.write(f"📝 Grammar Score: {grammar}")
        st.write(f"📄 Coherence Score: {coherence}")
        st.write(f"📚 Vocabulary Score: {vocab}")

        # -----------------------------
        # EVALUATION (simple)
        # -----------------------------
        st.subheader("⚖️ Evaluation")

        expected_score = 8  # pretend human score
        error = abs(expected_score - final_score)

        st.write(f"👨‍🏫 Human Score (example): {expected_score}")
        st.write(f"📉 Error: {error}")

        if error <= 2:
            st.success("✅ Good prediction accuracy")
        else:
            st.warning("⚠️ Needs improvement")
