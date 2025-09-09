import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import os
import streamlit as st

# Paths and constants
AGE_MODEL_PATH = "age_model.keras"
EMOTION_MODEL_PATH = "emotion_model.h5"
SAMPLE_RATE = 22050
MFCC_FEATURES = 40
AGE_GROUPS = ['teens', 'twenties', 'thirties', 'fourties', 'fifties', 'sixties', 'seventies', 'eighties']
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']  # 7 classes to match model

# Function to extract MFCC features (shared for both models)
def extract_mfcc(file_path, n_mfcc=40, max_len=174):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        if mfcc.shape[1] > max_len:
            mfcc = mfcc[:, :max_len]
        else:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Gender detection using PYIN
def is_male_voice(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y=audio, 
            sr=sr, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7')
        )
        valid_pitches = f0[voiced_flag]
        if len(valid_pitches) == 0:
            print(f"No voiced segments detected in {file_path}. Assuming male.")
            return True
        avg_pitch = np.mean(valid_pitches)
        print(f"Average fundamental pitch for {file_path}: {avg_pitch:.2f} Hz")
        return avg_pitch < 200  # Adjusted threshold
    except Exception as e:
        print(f"Error in gender detection for {file_path}: {e}")
        return True

# Predict age for a single file
def predict_age(file_path, model, le):
    if not is_male_voice(file_path):
        print(f"Non-male voice detected for {file_path}. Skipping age prediction.")
        return "Upload male voice", None
    print(f"Given input is a male voice.")
    mfcc = extract_mfcc(file_path)
    if mfcc is None:
        return "Error processing audio", None
    mfcc = np.expand_dims(mfcc, axis=0)
    prediction = model.predict(mfcc)
    age_idx = np.argmax(prediction)
    age_group = le.inverse_transform([age_idx])[0]
    is_senior = age_group in ['sixties', 'seventies', 'eighties']
    return f"Predicted age group: {age_group}" + (" (Senior Citizen)" if is_senior else ""), is_senior

# Predict emotion for a single file
def predict_emotion(file_path, model, le):
    mfcc = extract_mfcc(file_path)
    if mfcc is None:
        return "Error processing audio"
    mfcc = np.expand_dims(mfcc, axis=0)
    prediction = model.predict(mfcc)
    emotion_idx = np.argmax(prediction)
    try:
        emotion = le.inverse_transform([emotion_idx])[0]
        return f"Predicted emotion: {emotion}"
    except ValueError as e:
        print(f"Error in emotion prediction: {e}")
        return f"Error: Predicted index {emotion_idx} not in {EMOTION_CLASSES}"

# Streamlit app
def main():
    st.title("Voice Age and Emotion Predictor")
    st.write("Upload a male voice audio file (.mp3) to predict age group. Emotion will be predicted only for senior citizens (age > 60).")

    # Load age model
    age_model = None
    age_le = None
    if not os.path.exists(AGE_MODEL_PATH):
        st.error(f"No age model found at {AGE_MODEL_PATH}. Please ensure the model file exists.")
        return
    try:
        st.write(f"Loading age model from {AGE_MODEL_PATH}")
        age_model = tf.keras.models.load_model(AGE_MODEL_PATH)
        age_le = LabelEncoder().fit(AGE_GROUPS)
    except Exception as e:
        st.error(f"Error loading age model: {e}")
        return

    # Load emotion model
    emotion_model = None
    emotion_le = None
    if not os.path.exists(EMOTION_MODEL_PATH):
        st.error(f"No emotion model found at {EMOTION_MODEL_PATH}. Please ensure the model file exists.")
        return
    try:
        st.write(f"Loading emotion model from {EMOTION_MODEL_PATH}")
        emotion_model = tf.keras.models.load_model(EMOTION_MODEL_PATH)
        emotion_le = LabelEncoder().fit(EMOTION_CLASSES)
    except Exception as e:
        st.error(f"Error loading emotion model: {e}")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an MP3 file", type=["mp3"])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_file_path = os.path.join("tmp", uploaded_file.name)
        os.makedirs("tmp", exist_ok=True)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.write(f"Selected file: {uploaded_file.name}")
        
        # Predict button
        if st.button("Predict Age and Emotion"):
            if os.path.exists(temp_file_path):
                # Predict age
                age_result, is_senior = predict_age(temp_file_path, age_model, age_le)
                st.write(f"Age Result: {age_result}")
                
                # Predict emotion only for senior citizens
                if is_senior:
                    emotion_result = predict_emotion(temp_file_path, emotion_model, emotion_le)
                    st.write(f"Emotion Result: {emotion_result}")
                
                # Clean up temporary file
                os.remove(temp_file_path)
            else:
                st.error(f"File not found: {temp_file_path}")

if __name__ == "__main__":
    main()