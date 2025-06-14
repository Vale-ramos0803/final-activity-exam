import os, pathlib

# ──────────────────────────────────────────────────────────────────────────────
# 1) Redirect HOME so that pathlib.Path.home() → /mnt/data (writable in Spaces)
os.environ["HOME"] = "/mnt/data"
pathlib.Path.home = lambda *a, **k: pathlib.Path("/mnt/data")

# 2) Make the .streamlit config directory there
home = pathlib.Path.home()
config_dir = home / ".streamlit"
config_dir.mkdir(parents=True, exist_ok=True)

# 3) Disable usage stats (write config.toml into that dir)
with open(config_dir / "config.toml", "w") as f:
    f.write("[browser]\n")
    f.write("gatherUsageStats = false\n")
# ──────────────────────────────────────────────────────────────────────────────

# Now it’s safe to import everything else
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud
from datasets import load_dataset
from huggingface_hub import hf_hub_download




# Page configuration and title
st.set_page_config(page_title="Hate Speech Detection App", layout="wide")
st.title("🚨 Hate Speech Detection App")
st.write("Use the sidebar to navigate between pages.")

# Load model and tokenizer
@st.cache_resource
def load_model():
    # This downloads & caches the model file at runtime
    path = hf_hub_download(
        repo_id="ValeriaRamos8/final-activity-model",
        filename="Final_Activity.h5"
    )
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_tokenizer():
    tok_path = hf_hub_download(
        repo_id="ValeriaRamos8/final-activity-model",
        filename="tokenizer.pkl"
    )
    with open(tok_path, "rb") as f:
        return pickle.load(f)


# Load dataset from Hugging Face Datasets
@st.cache_data
def load_data():
    ds = load_dataset("tweet_eval", "hate")
    df_train = ds["train"].to_pandas().rename(columns={"text": "tweet", "label": "class"})
    df_test  = ds["test"].to_pandas().rename(columns={"text": "tweet", "label": "class"})
    return pd.concat([df_train, df_test], ignore_index=True)

df = load_data()

# Sidebar for navigation
page = st.sidebar.selectbox(
    "Choose a page", [
        "Inference Interface", 
        "Dataset Visualization", 
        "Hyperparameter Tuning", 
        "Model Analysis and Justification"
    ]
)

# Page 1: Inference Interface
if page == "Inference Interface":
    st.header("🧠 Inference Interface")
    user_input = st.text_area("Enter text:")
    if st.button("Predict") and user_input:
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=26)
        pred = model.predict(padded)[0]
        labels = ["Hate Speech", "Offensive", "Neither"]
        st.subheader(f"Prediction: {labels[np.argmax(pred)]}")
        st.subheader("Confidence Scores:")
        for i, lbl in enumerate(labels):
            st.write(f"- {lbl}: {pred[i]:.4f}")

# Page 2: Dataset Visualization
elif page == "Dataset Visualization":
    st.header("📊 Dataset Visualization")
    # Class distribution
    st.subheader("Class Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="class", data=df, ax=ax1)
    ax1.set_xticklabels(["Hate Speech", "Offensive", "Neither"])
    st.pyplot(fig1)
    # Token length histogram
    st.subheader("Token Length Histogram")
    df["length"] = df["tweet"].apply(lambda x: len(str(x).split()))
    fig2, ax2 = plt.subplots()
    ax2.hist(df["length"], bins=30)
    ax2.set_title("Tweet Length Distribution")
    st.pyplot(fig2)
    # Word Cloud
    st.subheader("Word Cloud")
    text_all = " ".join(df["tweet"].astype(str))
    wc = WordCloud(width=800, height=400, background_color='white').generate(text_all)
    fig3, ax3 = plt.subplots()
    ax3.imshow(wc, interpolation='bilinear')
    ax3.axis('off')
    st.pyplot(fig3)

# Page 3: Hyperparameter Tuning
elif page == "Hyperparameter Tuning":
    st.header("⚙️ Hyperparameter Tuning")
    st.markdown("""
**Tuned parameters:** learning rate, dropout rate, LSTM units, batch size  
**Best configuration:**  
- Learning rate: 0.001  
- Dropout: 0.3  
- Units: 64  
- Batch size: 64  
""")
    st.image("https://raw.githubusercontent.com/username/hyperparam-visuals/main/optuna_plot.png", caption="Hyperparameter tuning results")

# Page 4: Model Analysis and Justification
elif page == "Model Analysis and Justification":
    st.header("📈 Model Analysis and Justification")
    labels = ["Hate Speech", "Offensive", "Neither"]
    # Prepare data
    texts = df["tweet"].astype(str).tolist()
    y_true = df["class"].values
    seqs = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=26)
    preds = model.predict(padded, batch_size=64)
    y_pred = np.argmax(preds, axis=1)
    # Classification report
    st.subheader("Classification Report")
    report = classification_report(y_true, y_pred, target_names=labels)
    st.text(report)
    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    st.pyplot(fig_cm)
    # Error analysis
    st.subheader("Error Analysis")
    df_err = df[y_true != y_pred]
    for idx, row in df_err.sample(5, random_state=42).iterrows():
        st.write(f"**True:** {labels[int(row['class'])]} vs **Pred:** {labels[y_pred[idx]]}")
        st.write(f"> {row['tweet']}\n")
    st.subheader("Suggestions for Improvement")
    st.write("- Add more annotated data\n- Use transformer-based models\n- Implement data augmentation\n- Ensemble multiple models")