from pathlib import Path 

import joblib
import librosa
import numpy as np 
import pandas as pd

#Load the project folder
PROJECT_DIR = Path(__file__).resolve().parent


# Locate the project folder
DATASET_DIR = PROJECT_DIR  /"dataset"/ "Audio_Speech_Actors_01-24"
MODEL_PATH =  PROJECT_DIR / "emotion_model.joblib"

#Ask the user to enter an audio file path
audio_path_text = input(
    "Enter the full path to a WAV audio file"
).strip().strip('"')

#convert the text into file path
audio_path = Path(audio_path_text)

#Check whether the file exists
if not audio_path.is_file():
    raise FileNotFoundError(
        f"Audio file not found: {audio_path}"
    )

#Chack that the file is WAV audio file 
if audio_path.suffix.lower() != ".wav":
    raise ValueError("Please select a WAV audio file.")
print("Selected audio file:", audio_path.name)


#Select one first recording only if audio files were found


#Load the trained model
model = joblib.load(MODEL_PATH)

#Load the audio at the same sample rate used in training
audio, sample_rate = librosa.load(
    audio_path,
    sr=22050,
    res_type="kaiser_fast"
)

#Extract the 40 MFCC features
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=40
)

#Average the feature across time
mfcc_features = np.mean(mfcc, axis=1)

#Give the feature the same column names used during training
feature_names = [f"mfcc_{i+1}" for i in range(40)]
input_data = pd.DataFrame(
    [mfcc_features],
    columns=feature_names
)

#Predict the emotion
predicted_emotion = model.predict(input_data)[0]

#Display the result
print("Audio_file:", audio_path.name)
print("Predicted emotion", predicted_emotion)
