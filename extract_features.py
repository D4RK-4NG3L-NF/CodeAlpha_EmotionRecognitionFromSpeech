
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

# Locate the project and dataset folders
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset" / "Audio_Speech_Actors_01-24"

# Find all audio files
audio_files = sorted(DATASET_DIR.rglob("*.wav"))

# Convert emotion codes into emotion names
emotion_labels = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised"
}

features_list = []

# Process every audio file
for index, audio_path in enumerate(audio_files, start=1):
    # Load the audio
    audio, sample_rate = librosa.load(
        audio_path,
        sr=22050,
        res_type="kaiser_fast"
    )

    # Extract 40 MFCC features
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    # Average the features across time
    mfcc_features = np.mean(mfcc, axis=1)

    # Read the emotion code from the filename
    emotion_code = audio_path.stem.split("-")[2]
    emotion = emotion_labels[emotion_code]

    # Store features and emotion
    row = mfcc_features.tolist()
    row.append(emotion)
    features_list.append(row)

    # Show progress every 100 files
    if index % 100 == 0:
        print(f"Processed {index}/{len(audio_files)} files")

# Create column names for the CSV
columns = [f"mfcc_{i+1}" for i in range(40)]
columns.append("emotion")

# Convert the results into a table
features_df = pd.DataFrame(features_list, columns=columns)

# Save the extracted features
output_path = PROJECT_DIR / "features.csv"
features_df.to_csv(output_path, index=False)

print("\nFeature extraction complete!")
print("Total audio files processed:", len(features_df))
print("Features and labels shape:", features_df.shape)
print("Saved to:", output_path)
print("\nFirst five rows:")
print(features_df.head())

print("\nNumber of samples per emotion:")
print(features_df["emotion"].value_counts())
