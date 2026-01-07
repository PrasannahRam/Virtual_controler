import time
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import os ,json
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils


screen_w, screen_h = pyautogui.size()

def get_coordinates(hanlLms):
    return [(hanlLms.landmark[i].x, hanlLms.landmark[i].y) for i in range(21)]

def normalize_landmarks(handLms): # coordinated form
    wrist = handLms[0]
    return [(x - wrist[0], y - wrist[1]) for x, y in handLms]

def is_open_palm(hand_landmarks): # not coordinate form
    """
    Detect open palm based on Y-axis descending order
    Returns True if all 4 fingers are open
    """

    fingers = [
        [5, 6, 7, 8],     # Index
        [9, 10, 11, 12], # Middle
        [13, 14, 15, 16],# Ring
        [17, 18, 19, 20] # Pinky
    ]

    for finger in fingers:
        mcp = hand_landmarks.landmark[finger[0]].y
        pip = hand_landmarks.landmark[finger[1]].y
        dip = hand_landmarks.landmark[finger[2]].y
        tip = hand_landmarks.landmark[finger[3]].y

        # Must be descending upward (tip highest)
        if not (tip < dip < pip < mcp):
            return False

    return True     # #




def save_sample(sequence, label):
    FILE_NAME = "gesture_dataset_NN.json"

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

    with open(FILE_NAME, "r") as f:
        data = json.load(f)

    sample = {
        "label": label,
        "sequence": sequence
    }

    data["samples"].append(sample)

    with open(FILE_NAME, "w") as f:
        json.dump(data, f, indent=4)

class Recorder:
    def __init__(self,n,continuosCap = False):
        self.lastLmc = [(0, 0)] * 21
        self.data = []  ##[[(x,y,z),...] * 30frames * n times]

    def ispalmOpen(self,hand_landmarks):
        is_open_palm(hand_landmarks)

    def interpolate_motion(devSet, target_frames=30):
        """
        devSet: list of frames
          frame -> list of 21 landmarks
          landmark -> (dx, dy)

        returns: list with exactly target_frames frames
        """

        devSet = np.array(devSet)
        # shape: (orig_frames, 21, 2)

        orig_frames = devSet.shape[0]

        # Edge case: too small motion
        if orig_frames < 2:
            return None

        # Old and new time indices
        old_idx = np.arange(orig_frames)
        new_idx = np.linspace(0, orig_frames - 1, target_frames)

        normalized = np.zeros((target_frames, 21, 2))

        for lm in range(21):
            for axis in range(2):  # x and y
                normalized[:, lm, axis] = np.interp(
                    new_idx,
                    old_idx,
                    devSet[:, lm, axis]
                )

        return normalized.tolist()

    def capture(self):
        success, img = cap.read()
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        return results,img

    def captureDev(self, handLms):  # converts from position to deviation
        landmarkDeviation = [tuple(map(lambda j, k: j - k, self.lastLmc[n], handLms[n])) for n in range(21)]
        return landmarkDeviation

    def storeMotion(self):
        devSet = []

        while True:
            results,img = self.capture()

            if results.multi_hand_landmarks:
                if is_open_palm(results.multi_hand_landmarks[0]):
                    break
                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            else:
                continue

            cv2.imshow(f"Hand Detection ", img)
            if cv2.waitKey(1) & 0xFF == 27:
                    return results
            ##################################################################################################################

            handLms = get_coordinates(results.multi_hand_landmarks[0])
            handLms = normalize_landmarks(handLms)
            dev = self.captureDev(handLms)
            devSet.append(dev)
            self.lastLmc = handLms


        devSet = self.interpolate_motion(devSet, target_frames=15)
        if devSet:
            print(f'capture end,frames = {len(devSet)}')
            self.data.append(devSet)

    def scan(self):

        while len(self.data) < 50:
            results, _ = self.capture()
            if results.multi_hand_landmarks and not is_open_palm(results.multi_hand_landmarks[0]):
                print('detecting')
                self.storeMotion()


rec = Recorder(1)
rec.scan()
save_sample(rec.data,'zoom')
print()