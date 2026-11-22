from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

data = np.load(Path("processed_data/mel_splits.npz"))
track_train = set(data["track_train"])
track_val   = set(data["track_val"])
track_test  = set(data["track_test"])

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
genre_to_index = {genre: idx for idx, genre in enumerate(genres)}

cleaned_csv_path = Path("processed_data/features_3_sec_cleaned.csv")
df = pd.read_csv(cleaned_csv_path)

# Double checking labels exist
inferred_genre = df["filename"].str.split(".", n=1).str[0]
df["label"] = df["label"].fillna(inferred_genre)

# Canonical representation of labels
df["idx_rep_label"] = df["label"].map(genre_to_index)
df["idx_rep_label"] = df["idx_rep_label"].astype(int)

train_df = df[df["track_id"].isin(track_train)]
val_df   = df[df["track_id"].isin(track_val)]
test_df  = df[df["track_id"].isin(track_test)]

assert set(train_df["track_id"]).isdisjoint(set(val_df["track_id"]))
assert set(train_df["track_id"]).isdisjoint(set(test_df["track_id"]))
assert set(val_df["track_id"]).isdisjoint(set(test_df["track_id"]))

# Verify the tracks are unqiue
print("Unique tracks in Train:", train_df["track_id"].nunique())
print("Unique tracks in Test: ", test_df["track_id"].nunique())

# Verify there are no overlapping tracks between train and test sets
overlap = set(train_df["track_id"]).intersection(set(test_df["track_id"]))
print("Overlapping tracks:", len(overlap))

ignore_cols = {"filename", "length", "label", "track_id", "idx_rep_label"} # Ignore non-numeric data
feature_cols = [c for c in df.columns if c not in ignore_cols]

# Find training, validation, and test sets for Random Forest with labels
X_train_rf, label_train_rf = train_df[feature_cols].to_numpy(), train_df["idx_rep_label"].to_numpy()
X_val_rf, label_val_rf     = val_df[feature_cols].to_numpy(), val_df["idx_rep_label"].to_numpy()
X_test_rf, label_test_rf   = test_df[feature_cols].to_numpy(), test_df["idx_rep_label"].to_numpy()

print(f"Random Forest Dataset Shapes:")
print(f"  Train: {X_train_rf.shape}")
print(f"  Val:   {X_val_rf.shape}")
print(f"  Test:  {X_test_rf.shape}")

# # Hyperparameter Tuning Loop on Validation Data
# depths = [5, 10, 15, 20, None]
# best_val_acc = 0.0
# best_depth = None
# best_rf = None

# print("--- Hyperparameter Tuning (max_depth) ---")
# for depth in depths:
#     rf_temp = RandomForestClassifier(n_estimators=200, max_depth=depths, random_state=30, n_jobs=-1)
#     rf_temp.fit(X_train_rf, label_train_rf)
    
#     train_acc = accuracy_score(label_train_rf, rf_temp.predict(X_train_rf))
#     val_acc = accuracy_score(label_val_rf, rf_temp.predict(X_val_rf))
    
#     print(f"max_depth={str(depth):4s} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         best_depth = depth
#         best_rf = rf_temp
    
# print(f"\nBest max_depth selected: {best_depth} (Val Acc: {best_val_acc:.4f})\n")

rf = RandomForestClassifier(n_estimators=400, max_depth=12, random_state=30, n_jobs=-1)
rf.fit(X_train_rf, label_train_rf)

train_acc = accuracy_score(label_train_rf, rf.predict(X_train_rf))
val_acc = accuracy_score(label_val_rf, rf.predict(X_val_rf))

print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

test_preds = rf.predict(X_test_rf)
print(f"Test Accuracy:       {accuracy_score(label_test_rf, test_preds):.4f}\n")
print(classification_report(label_test_rf, test_preds, target_names=genres, digits=3))
print("Confusion Matrix (rows=true, cols=pred):")
print(confusion_matrix(label_test_rf, test_preds))

fig, ax = plt.subplots(figsize=(10, 8))
cmp = ConfusionMatrixDisplay.from_predictions(label_test_rf, test_preds, display_labels=genres, ax=ax)

plt.title("Random Forest Baseline - Confusion Matrix")
plt.tight_layout()
plt.savefig("processed_data/rf_confusion_matrix.png", dpi=300)
plt.show()