import time

import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def move_cursor(x_dir,ydir):
    x, y = pyautogui.position()

    screen_w, screen_h = pyautogui.size()
    mult  = 5000
    if x_dir<0.005 and ydir< 0.005:
        mult = 1000

    x,y = x+x_dir*mult, y+ydir*mult
    x = max(10, min(x, screen_w - 10))
    y = max(10, min(y, screen_h - 10))



    pyautogui.moveTo(x, y)


def drag_obj():
    pyautogui.mouseDown()
    time.sleep(0.3)
    last_index = [0, 0]
    while True:
        loc_x,loc_y = last_index
        results = scan()

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                ishandClosed = pose(handLms)['hand_closed']
                if not ishandClosed:
                    print('opened')
                    pyautogui.mouseUp()
                    return
                else:
                    print('closed')
                    screen_w, screen_h = pyautogui.size()
                    if last_index == [0, 0]:
                        last_index = [handLms.landmark[0].x, handLms.landmark[0].y]
                    else:
                        xdir, ydir = handLms.landmark[0].x - loc_x, handLms.landmark[0].y - loc_y
                        x, y = pyautogui.position()
                        x, y = x + xdir*5000, y + ydir*5000
                        print(x,y)
                        x = max(50, min(x, screen_w - 50))
                        y = max(50, min(y, screen_h - 50))
                        print(x,y)
                        print(screen_w, screen_h)
                        print('*************************')
                        try:
                            pyautogui.moveTo(x, y)
                        except pyautogui.FailSafeException:
                            print("Fail-safe triggered")
                        last_index = [handLms.landmark[0].x, handLms.landmark[0].y]


def pose(handLms):
    finger_tips = [ 12, 16, 20]  #Middle, Ring, Pinky
    palm_closed = True

    for tip in finger_tips:
        if handLms.landmark[tip].y < handLms.landmark[tip - 2].y:
            palm_closed = False
            break

    index_open = (handLms.landmark[8].y < handLms.landmark[7].y < handLms.landmark[6].y )
    if palm_closed and index_open:
        return {'indexing':True, 'landmark' : handLms.landmark[8], 'hand_closed' : False}
    elif (not index_open) and palm_closed:
        return {'indexing':False, 'landmark' : handLms.landmark[0], 'hand_closed' : True}

    else:
        return {'indexing':False, 'landmark' : [0,0,0], 'hand_closed' : False}




def scan():
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Detection", img)
    if cv2.waitKey(1) & 0xFF == 27:
        return results
    else:
        return results

last_index = [0,0]

while True:
    results = scan()

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            isindexing,index,ishandClosed = pose(handLms)['indexing'],pose(handLms)['landmark'], pose(handLms)['hand_closed']
            if isindexing:
                if last_index == [0,0]:
                    last_index = [index.x,index.y]
                else:
                    move_cursor(index.x - last_index[0], index.y - last_index[1])
                    last_index = [index.x,index.y]
            elif ishandClosed:
                drag_obj()
            else:
                last_index = [0,0]



cap.release()
cv2.destroyAllWindows()
