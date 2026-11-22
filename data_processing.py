import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import librosa
import librosa.display

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
genre_to_index = {genre: idx for idx, genre in enumerate(genres)}

audio_path = Path("genres_original")
csv_path = Path("features_3_sec.csv")
output_dir = Path("processed_data")
output_dir.mkdir(parents=True, exist_ok=True)

audio_duration = 30.0
clip_duration = 3.0
num_clips = 10 # Number of clips to extract from each audio file

np.random.seed(30) # For reproducibility
torch.manual_seed(30)

def find_invalid_audio_files(audio_path: Path, genres: list[str]) -> set[str]:
    "Checks every <genre>.<number>.wav audio file for 30s duration and loadability. Returns a set of 5-digit invalid audio file numbers from any genre so they can be dropped from all the genres for consistency."

    invalid_files = set()

    for genre in genres:
        genre_path = audio_path/genre

        for i in range(100):
            index_str = f"{i:05d}"
            file_path = genre_path/f"{genre}.{index_str}.wav"

            if not file_path.exists():
                invalid_files.add(index_str)
                continue

            try:
                y, sr = librosa.load(str(file_path), sr=None) # amplitude array and sampling rate
                duration = librosa.get_duration(y=y, sr=sr)
                if len(y) == 0 or duration < (audio_duration - 0.1):  # Allow a small tolerance for duration
                    invalid_files.add(index_str)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                invalid_files.add(index_str)
    
    return invalid_files

bad_audio = find_invalid_audio_files(audio_path, genres)
valid_audio = [f"{i:05d}" for i in range(100) if f"{i:05d}" not in bad_audio] # all the valid audio file numbers (5-digit strings) across all genres


def clean_features_csv(csv_path: Path, valid_audio: list[str], genres: list[str], out_path: Path) -> pd.DataFrame:
    "Cleans the features CSV file used in baseline model by dropping rows corresponding to invalid audio files."

    df = pd.read_csv(csv_path)
    before_rows = len(df)

    valid_filenames = {f"{genre}.{idx}.{clip_idx}.wav" for genre in genres for idx in valid_audio for clip_idx in range(10)}

    cleaned_df = df[df['filename'].isin(valid_filenames)].copy() # Keep only rows with valid filenames
    cleaned_df["track_id"] = cleaned_df['filename'].str.rsplit(".", n=2).str[0]# Extract track_id from filename
    cleaned_df.to_csv(out_path, index=False)

    print(f"CSV rows before cleaning: {before_rows} -> after cleaning: {len(cleaned_df)}")
    print(cleaned_df["label"].value_counts().sort_index())
    
    return cleaned_df

# cleaned_csv = clean_features_csv(csv_path, valid_audio, genres, output_dir/"features_3_sec_cleaned.csv")

## -- Check for exact feature duplicates in the cleaned CSV file --
# feature_cols = [c for c in cleaned_csv.columns if c not in ["filename", "length", "label", "track_id"]]

# # Exact feature match across different clips
# exact_dupes = cleaned_csv[cleaned_csv.duplicated(subset=feature_cols, keep=False)]
# print(f"Exact feature-identical rows: {len(exact_dupes)}")
# if len(exact_dupes) > 0:
#     print(exact_dupes[["filename", "track_id", "label"]].head(10))


def build_dataset(audio_path: Path, genres: list[str], valid_audio: list[str]):
    "Slices each valid 30s audio file into 10 non-overlapping 3s clips. It then computes the log-Mel Spectogram per clip."

    specs = []
    labels = []
    track_ids = []
    clips_name = []


    for genre in genres:
        genre_ind = genre_to_index[genre]

        for index_str in valid_audio:
            track_id = f"{genre}.{index_str}"
            file_path = audio_path/genre/f"{track_id}.wav"

            y, sr = librosa.load(str(file_path), mono=True) # Load audio file as mono
            total_samples = int(sr*audio_duration)
            if len(y) < total_samples:  # Pad the audio if it's shorter than 30 seconds
                y = np.pad(y, (0, total_samples - len(y)))  # Pad with zeros to make it 30 seconds long
            else:
                y = y[:total_samples]  # Ensure we only take the first 30 seconds

            for clip_idx in range(num_clips):
                start_sample = int(clip_duration * sr) * clip_idx
                end_sample = int(clip_duration * sr) * (clip_idx + 1)

                clip_y = y[start_sample:end_sample]

                mel = librosa.feature.melspectrogram(y=clip_y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
                
                log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

                # Standardize the spectogram for CNN model
                mean = log_mel.mean()
                std = log_mel.std() + 1e-9  # Add a small value to avoid division by zero
                log_mel_norm = (log_mel - mean) / std

                specs.append(log_mel_norm[np.newaxis, :, :])  # Add channel dimension
                labels.append(genre_ind)
                track_ids.append(track_id)
                clips_name.append(f"{track_id}.{clip_idx}.wav")
    
    dataset = np.stack(specs, axis=0) # Shape: (num_samples, 1, n_mels, time_frames)
    y_arr = np.array(labels, dtype=np.int64) # Label array
    track_arr = np.array(track_ids) # used to split the dataset into train/val/test sets based on track_id and not split the 3s clips up

    return dataset, y_arr, track_arr, clips_name

X_mel, mel_label, track_arr, clips_name = build_dataset(audio_path, genres, valid_audio)

def save_spectograms(X_mel: np.ndarray, clips_name: list[str], output_dir: Path):
    "Saves each log-Mel spectrogram as a PNG image for visual inspection."

    spec_dir = output_dir/"spectrograms"
    spec_dir.mkdir(parents=True, exist_ok=True)

    for i, clip_name in enumerate(clips_name):
        plt.figure(figsize=(4, 3))

        librosa.display.specshow(X_mel[i][0], sr=22050, hop_length=512, x_axis='time', y_axis='mel')

        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Log-Mel Spectrogram: {clip_name}")
        plt.tight_layout()
        plt.savefig(spec_dir/f"{clip_name}.png")
        plt.close()

# save_spectograms(X_mel, clips_name, output_dir)

# Split training, validation and testing data 70/15/15; group shuffle split by track_id so all 10 clips of a given song stay together in a single split.
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=30) 
train_val_idx, test_idx = next(gss_test.split(X_mel, mel_label, groups=track_arr))

X_test, label_test, track_test = X_mel[test_idx], mel_label[test_idx], track_arr[test_idx]

# Split remaining 85% into 70/15 so 0.15/0.85 ≈ 0.1765 of train_val
gss_val = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=30)
train_idx, val_idx = next(gss_val.split(X_mel[train_val_idx], mel_label[train_val_idx], groups=track_arr[train_val_idx]))

X_train, label_train = X_mel[train_val_idx][train_idx], mel_label[train_val_idx][train_idx]
X_val, label_val = X_mel[train_val_idx][val_idx], mel_label[train_val_idx][val_idx]

# Double check no song appears in more than one split
assert set(track_arr[train_val_idx][train_idx]).isdisjoint(track_arr[train_val_idx][val_idx])
assert set(track_arr[train_val_idx][train_idx]).isdisjoint(track_arr[test_idx])
assert set(track_arr[train_val_idx][val_idx]).isdisjoint(track_arr[test_idx])

print(f"Clips -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Save all splits to compressed file for reloading
np.savez_compressed(
    output_dir / "mel_splits.npz",
    X_train=X_train, train_label=label_train,
    X_val=X_val, val_label=label_val,
    X_test=X_test, test_label=label_test,
    genres=np.array(genres),
)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(label_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(label_val, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(label_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Check genre stratification in each split
data = np.load(Path("processed_data/mel_splits.npz"))
genres = data["genres"]

for split_name, key in [("Train", "train_label"), ("Val", "val_label"), ("Test", "test_label")]:
    label_split = data[key]
    counts = np.bincount(y_split, minlength=len(genres))
    print(f"\n{split_name} (total clips: {len(y_split)}):")
    for g, c in zip(genres, counts):
        print(f"  {g:10s}: {c}")