from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import json

import json
import numpy as np

def setSeq(sequence):

    frames_per_sample = 30
    if len(sequence) < frames_per_sample:
        # Pad with zeros
        padding = [[0] * 63] * (frames_per_sample - len(sequence))
        sequence.extend(padding)
    elif len(sequence) > frames_per_sample:
        # Truncate extra frames
        sequence = sequence[:frames_per_sample]

    for li in range(30):
        new_li = [ ]
        for i in range(0,63,3):
            new_li.append(sequence[li][i])
            new_li.append(sequence[li][i+1])
        sequence[li] = new_li
    return sequence
with open("gesture_dataset_NN.json", "r") as f:
    data = json.load(f)



X = []
y = []

for sample in data["samples"]:
    label = sample["label"]
    for sequence in sample["sequence"]:
        X.append(np.array(sequence).flatten())
        y.append(label)

X = np.array(X)
y = np.array(y)

print(X.shape)  # (num_samples, num_features)
print(y.shape)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure()
for label in np.unique(y):
    idx = y == label
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=label)

plt.legend()
plt.title("Motion Dataset Visualization (PCA)")
plt.show()