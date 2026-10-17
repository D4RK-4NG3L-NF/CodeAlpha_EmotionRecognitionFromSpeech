from pathlib import Path
#Pojects dataset folder's
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
#Find all WAV audio file 
audio_file = list(DATASET_DIR.rglob("*.wav"))
print("RAVDESS Dataset Check")
print("----------------------------")
print(f"Dataset folder: {DATASET_DIR}")
print(f"Audio file found: {len(audio_file)}")
if audio_file:
    print("\nFirst 5 audio_files:")
    for file in audio_file[:5]:
        print(file.relative_to(DATASET_DIR))
else:
    print("\nNo WAV file found. We need to check the Extract folder location.")
