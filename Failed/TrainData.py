import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load JSON file
with open("gesture_dataset.json", "r") as f:
    data = json.load(f)

X = []
y = []

# Loop through samples
for sample in data['samples']:
    label = sample['label']
    sequence = sample['sequence']  # list of frames
    # Flatten the sequence: frames x features -> 1D vector

    # Pad or truncate to frames_per_sample
    frames_per_sample = 30
    if len(sequence) < frames_per_sample:
        # Pad with zeros
        padding = [[0] * 63] * (frames_per_sample - len(sequence))
        sequence.extend(padding)
    elif len(sequence) > frames_per_sample:
        # Truncate extra frames
        sequence = sequence[:frames_per_sample]

    flat_sequence = np.array(sequence).flatten()
    X.append(flat_sequence)
    y.append(label)

X = np.array(X)
y = np.array(y)
print("Features shape:", X.shape)
print("Labels shape:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("Predictions:", y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

joblib.dump(knn, "knn_model_2.pkl")