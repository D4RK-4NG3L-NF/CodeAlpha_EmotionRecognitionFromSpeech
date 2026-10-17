from pathlib import Path
import pandas as pd
#Locate the project 
PROJECT_DIR = Path(__file__).resolve().parent

#Locate the extracted features CSV file
FEATURES_FILE = PROJECT_DIR / "features.csv"

#Load CSV finle into a table
data = pd.read_csv(FEATURES_FILE)

#Display basic information 
print("Dataset Loaded Successfully!")
print("Dataset shape:", data.shape)
print("\nFirst five rows:")
print(data.head())

#Seperate the input features from the emotion labels
X = data.drop(columns=["emotion"])
y = data["emotion"]

#Display the shapes of the input and labels
print("\nInput features shape:", X.shape)
print("Emotion labels shape:", y.shape)

from  sklearn.model_selection  import train_test_split

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#Display the sizes of the split datasets
print("\nTraining features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Testing labels shape:", y_test.shape)

from sklearn.ensemble import RandomForestClassifier

#Create a Random Forest Classifier model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)
#Train the model using the training data
model.fit(X_train, y_train)

print("\nModel training completed succcessfuly!")


from sklearn.metrics import accuracy_score

#Predict emotions for the test recordings
y_pred = model.predict(X_test)

#Compare prediction with the correct emotions lables
accuracy = accuracy_score(y_test, y_pred)

#Display the model accoracy
print("\nModel accuracy:",  round(accuracy * 100, 2), "%")


from sklearn.metrics import confusion_matrix

#Define the order in which emotion will appear
emotion_names = [
    "Neutral", "Calm", "Happy", "Sad",
    "Angry", "Fearful", "Disgust",  "Surprised"
]
#Compare actual emotions with predicted emotions
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=emotion_names
)
#Display the confusion matrix
print("\nEmotion labels order:")
print(emotion_names)

print("\nConfusion_Matrix:")
print(cm)

from sklearn.metrics import classification_report

#Calculate performance assessments for each emotion
report = classification_report(
    y_test,
    y_pred,
    labels=emotion_names,
    zero_division=0
)
#Display the classification report
print("\nClassification report:")
print(report)


import joblib

#Set the path where the trained model will be saved
MODEL_PATH = PROJECT_DIR / "emotion_model.joblib"

#Save the trained model to a file
joblib.dump(model, MODEL_PATH)

print("\nModel saved succcesfully!")
print("Model location:", MODEL_PATH)
