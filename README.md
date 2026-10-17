
# Emotion Recognition from Speech

## Project Overview

This project uses machine learning to predict emotion categories from speech audio recordings. It extracts Mel-Frequency Cepstral Coefficients (MFCCs) from audio and uses a Random Forest classifier to predict an emotion.

This is a beginner AI/ML portfolio project developed as part of my machine-learning learning journey.

## Objectives

- Process speech recordings with Python.
- Extract numerical audio features using Librosa.
- Train a machine-learning classifier with Scikit-learn.
- Evaluate predictions using accuracy, a confusion matrix, and a classification report.
- Save and reuse the trained model for audio predictions.

## Dataset

This project uses the RAVDESS speech audio dataset.

The extracted dataset used in this project contains 1,440 WAV recordings across eight emotion categories:

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

The dataset is not included in this repository. Obtain the dataset separately and follow its access and usage terms.

Expected dataset folder:

`dataset/Audio_Speech_Actors_01-24/`

## Technologies and Libraries

- Python
- Librosa — audio loading and MFCC feature extraction
- NumPy — numerical calculations
- Pandas — data processing and CSV creation
- Scikit-learn — Random Forest training and evaluation
- Joblib — saving and loading the trained model
- VS Code — development environment

## How It Works

1. Load speech recordings from the dataset.
2. Extract 40 MFCC coefficients per recording.
3. Average the coefficients across time to create a 40-value feature vector.
4. Save the feature vectors and emotion labels to `features.csv`.
5. Separate the input features (X) from the target labels (y).
6. Split the data into training and testing sets using an 80/20 split.
7. Train a Random Forest classifier with 100 decision trees.
8. Evaluate the model using test accuracy, a confusion matrix, and a classification report.
9. Save the trained model as `emotion_model.joblib`.
10. Load the saved model to predict an emotion from a selected WAV recording.

## Project Structure

```text
Project_2/
├── dataset/                 # Dataset files (not included in GitHub)
├── .venv/                   # Local Python virtual environment (not included)
├── extract_features.py      # Extract MFCC features
├── features.csv             # Extracted features and labels
├── train_model.py           # Train and evaluate the model
├── emotion_model.joblib     # Saved trained model
├── predict_emotion.py       # Predict an emotion from a WAV file
├── requirements.txt         # Required Python libraries
├── .gitignore               # Files excluded from Git
└── README.md                # Project documentation
```

## Setup Instructions

### 1. Install Python

Python 3.11 is recommended for the environment used during development.

### 2. Create a virtual environment

From the project directory, run:

```powershell
py -3.11 -m venv .venv
```

### 3. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 5. Prepare the dataset

Obtain and extract the dataset into the expected folder structure:

`dataset/Audio_Speech_Actors_01-24/Actor_01/`

The remaining actor folders should also be inside `Audio_Speech_Actors_01-24`.

### 6. Extract features

```powershell
python extract_features.py
```

This creates `features.csv`.

### 7. Train and evaluate the model

```powershell
python train_model.py
```

This trains the Random Forest classifier, displays evaluation results, and saves `emotion_model.joblib`.

### 8. Predict an emotion

```powershell
python predict_emotion.py
```

When prompted, enter the full path to a WAV recording. The program extracts its features and displays the predicted emotion.

## Model Configuration and Results

- Algorithm: Random Forest Classifier
- Number of decision trees: 100
- Feature representation: 40 averaged MFCC values per recording
- Dataset size: 1,440 recordings
- Training split: 80% (1,152 recordings)
- Testing split: 20% (288 recordings)
- Baseline test accuracy: 63.19%

The model correctly classified 182 of the 288 recordings in the test set.

## Limitations and Future Improvements

- Improve the model's accuracy through feature engineering and model tuning.
- Evaluate performance on speakers not represented in the training set.
- Investigate which emotion classes are most frequently confused.
- Explore additional audio features and alternative classifiers.
- Improve the prediction interface to make audio selection easier.

The reported accuracy is a baseline result from a random stratified split. Because the split was performed by recording rather than by speaker, it does not establish how well the model generalizes to completely unseen speakers.

## Disclaimer

This model predicts one of eight dataset emotion categories from speech audio. Its predictions are estimates, not definitive judgments about a person's actual feelings.

## Author

Developed as a beginner AI/ML project to gain practical experience with audio processing, feature extraction, supervised learning, model evaluation, and model reuse.
