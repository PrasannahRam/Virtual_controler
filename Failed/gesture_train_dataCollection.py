import json
import os
import cv2

from Recorder import Recorder



def save_sample(sequence, label):
    FILE_NAME = "gesture_dataset.json"

    if not os.path.exists(FILE_NAME):
        data = {
            "meta": {
                "frames_per_sample": 30,
                "features": ["dx", "dy", "dz"]
            },
            "samples": []
        }

        with open(FILE_NAME, "w") as f:
            json.dump(data, f, indent=4)

    with open("gesture_dataset.json", "r") as f:
        data = json.load(f)

    sample = {
        "label": label,
        "sequence": sequence
    }

    data["samples"].append(sample)

    with open("gesture_dataset.json", "w") as f:
        json.dump(data, f, indent=4)






run = 30

sequence = []
while run > 0:
    recorder = Recorder(run)
    recorder.getLandmarks()
    flattenList = recorder.flatten()
    save_sample(flattenList,'open-palm')
    run -= 1

