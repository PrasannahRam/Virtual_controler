import time

import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils


screen_w, screen_h = pyautogui.size()

def get_coordinates(hanlLms):
    return [(hanlLms.landmark[i].x, hanlLms.landmark[i].y, hanlLms.landmark[i].z) for i in range(21)]

def normalize_landmarks(handLms):
    wrist = handLms[0]
    return [(x - wrist[0], y - wrist[1]) for x, y in handLms]

def is_open_palm(hand_landmarks):
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

    return True


class Recorder:
    def __init__(self,n,continuosCap = False):
        self.lastLmc = [(0, 0, 0)] * 21
        self.devSet = []  ##[[(x,y,z),...] * 30]
        self.n = n
        self.continuosCap = continuosCap
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
                img,
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
                handLms = normalize_landmarks(handLms)
                dev = self.captureDev(handLms)
                self.devSet.append(dev)
                self.lastLmc = handLms

        cv2.destroyWindow(f'Hand Detection {self.n}')
    def scan(self):
        while True:
            ##################################################################################################################
            success, img = cap.read()
            img = cv2.flip(img, 1)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            ##################################################################################################################
                handLms = get_coordinates(results.multi_hand_landmarks[0])
                handLms = normalize_landmarks(handLms)
                dev = self.captureDev(handLms)
                self.devSet.append(dev)
                self.lastLmc = handLms



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
