import json,numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import joblib

with open("gesture_dataset_NN.json", "r") as f:
    data = json.load(f)



X = []
y = []

for sample in data["samples"]:
    label = sample["label"]
    for sequence in sample["sequence"]:
        X.append(np.array(sequence).flatten())
        y.append(label)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y_cat = to_categorical(y_encoded)

# Train / validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_cat, test_size=0.2, random_state=42
)

# Stop over fitting
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Pickup a echo with lower validation lose
checkpoint = ModelCheckpoint(
    "gesture_nn_best.h5",
    monitor="val_loss",        # or "val_accuracy"
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=8,
callbacks=[early_stop, checkpoint],
    verbose=1
)
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")