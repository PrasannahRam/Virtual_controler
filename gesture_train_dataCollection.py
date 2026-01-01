import time
import json
import os
import cv2
import mediapipe as mp
import pyautogui

###########################################################################

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

screen_w, screen_h = pyautogui.size()


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


def get_coordinates(hanlLms):
    return [(hanlLms.landmark[i].x, hanlLms.landmark[i].y, hanlLms.landmark[i].z) for i in range(21)]


class Recorder:
    def __init__(self, cap,n):
        self.cap = cap
        self.lastLmc = [(0, 0, 0)] * 21
        self.devSet = []  ##[[(x,y,z),...] * 30]
        self.n = n

        # self.success, self.img = cap.read()
        # self.img = cv2.flip(self.img, 1)
        # self.imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

    def getLandmarks(self, frames: int = 30):
        print("get ready")
        time.sleep(1)
        print('Start')
        print('Scanning')
        print(' - ' * 30)
        for i in range(frames):
            print(' - ', end='')
            success, img = cap.read()
            img = cv2.flip(img, 1)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            cv2.putText(
                success,
                str(i),  # text to show
                (30, 50),  # position (x, y)
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (0, 255, 0),  # color (B, G, R)
                2,  # thickness
                cv2.LINE_AA
            )

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            cv2.imshow(f"Hand Detection {self.n}", img)
            if cv2.waitKey(1) & 0xFF == 27:
                return results



            if results.multi_hand_landmarks:
                handLms = get_coordinates(results.multi_hand_landmarks[0])
                dev = self.captureDev(handLms)
                self.devSet.append(dev)
                self.lastLmc = handLms
        cv2.destroyAllWindows()

    def captureDev(self, handLms):
        landmarkDeviation = [tuple(map(lambda j, k: j - k, self.lastLmc[n], handLms[n])) for n in range(21)]
        return landmarkDeviation

    def flatten(self):
        flattenList = []
        for frames in self.devSet:
            frame = []
            for fingers in frames:
                for divs in fingers:
                    frame.append(divs)
            flattenList.append(frame)
        return flattenList


run = 30

sequence = []
while run > 0:
    recorder = Recorder(cap,run)
    recorder.getLandmarks()
    flattenList = recorder.flatten()
    save_sample(flattenList,'open-palm')
    run -= 1

