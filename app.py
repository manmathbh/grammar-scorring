# Grammar_Scoring/app.py

# import gradio as gr
# import numpy as np
# import joblib
# import librosa

# # Check for silence
# def is_silent(audio, threshold=0.001):
#     return np.max(np.abs(audio)) < threshold

# # Feature extraction
# def extract_features(audio, sr):
#     features = {}
#     mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
#     mfccs_mean = mfccs.mean(axis=1)
#     for i, val in enumerate(mfccs_mean):
#         features[f"feature_{i}"] = val
#     return features

# # Main prediction function
# def predict_score(audio_path):
#     try:
#         print("📁 Filepath received:", audio_path)

#         if not audio_path or not isinstance(audio_path, str):
#             return "❌ Invalid audio file path"

#         # Load using librosa
#         audio, orig_sr = librosa.load(audio_path, sr=None)

#         if is_silent(audio):
#             return 0.0

#         # Resample
#         target_sr = 16000
#         if orig_sr != target_sr:
#             audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

#         # Feature extraction
#         feats = extract_features(audio, target_sr)
#         X = np.array([feats[col] for col in selected_features]).reshape(1, -1)
#         pred = model.predict(X)[0]
#         return round(pred, 2)

#     except Exception as e:
#         return f"❌ Error: {e}"

# # Load model and selected features
# model = joblib.load("grammar_score_model.pkl")
# selected_features = joblib.load("selected_features.pkl")

# # Gradio UI
# demo = gr.Interface(
#     fn=predict_score,
#     inputs=gr.Audio(sources="microphone", type="filepath", label="🎤 Record your voice (45–60s)"),
#     outputs=gr.Textbox(label="Predicted Grammar Score (0–5)"),
#     title="🗣️ Grammar Scoring Engine",
#     description="Speak for 45–60 seconds. The model will predict your grammar score based on MFCC features."
# )

# # Launch
# demo.launch(share=True)
#----------------------------------------------------------

import gradio as gr
import numpy as np
import joblib
import librosa

# Load model and selected features
model = joblib.load("grammar_score_model.pkl")
selected_features = joblib.load("selected_features.pkl")

# Check for silence
def is_silent(audio, threshold=0.001):
    return np.max(np.abs(audio)) < threshold

# Feature extraction
def extract_features(audio, sr):
    features = {}
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = mfccs.mean(axis=1)
    for i, val in enumerate(mfccs_mean):
        features[f"feature_{i}"] = val
    return features

# Main prediction function
def predict_score(audio_path):
    try:
        if not audio_path or not isinstance(audio_path, str):
            return "❌ Invalid audio file"

        audio, orig_sr = librosa.load(audio_path, sr=None)

        if is_silent(audio):
            return 0.0

        # Resample
        target_sr = 16000
        if orig_sr != target_sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

        # Feature extraction
        feats = extract_features(audio, target_sr)
        X = np.array([feats[col] for col in selected_features]).reshape(1, -1)
        pred = model.predict(X)[0]
        return round(pred, 2)

    except Exception as e:
        return f"❌ Error: {e}"

# 🎨 New Gradio Blocks UI
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue")) as demo:
    gr.Markdown("## 🗣️ Grammar Scoring Engine")
    gr.Markdown("Record your voice for 45–60 seconds and get your **grammar score (0–5)** based on audio features extracted using MFCC.")

    with gr.Row():
        audio_input = gr.Audio(sources="microphone", type="filepath", label="🎙️ Speak Now", interactive=True)
        score_output = gr.Textbox(label="📊 Predicted Grammar Score")

    submit_btn = gr.Button("🔍 Predict")

    submit_btn.click(fn=predict_score, inputs=audio_input, outputs=score_output)

    gr.Markdown("💡 Try to speak naturally for better results.")

# Launch app
demo.launch(share=True)
  