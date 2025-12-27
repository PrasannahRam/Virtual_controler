import time

import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)


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

def open_obj():
    while True:
        results = scan()

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                handPose = pose(handLms)

                if handPose['hand_open']:
                    print('open')
                    pyautogui.press('enter')

                    return
                elif handPose['hand_closed']:
                    print('didnt open')
                    time.sleep(0.5)
                    return



def pose(handLms):
    handPose =  {'indexing': False, 'landmark': [0,0,0], 'hand_closed': False ,'pinky_thum_touch':False,'hand_open':False}

    finger_tips = [ 12, 16, 20]  #Middle, Ring, Pinky
    palm_closed = True
    palm_open = True

    for tip in finger_tips:
        if handLms.landmark[tip].y < handLms.landmark[tip - 2].y:
            palm_closed = False
            break

    index_open = (handLms.landmark[8].y < handLms.landmark[7].y < handLms.landmark[6].y )

    for tip in finger_tips:
        if handLms.landmark[tip].y > handLms.landmark[tip - 2].y:
            palm_open = False
            break

    if palm_closed and index_open:
        handPose['indexing']=True
        handPose['landmark'] = handLms.landmark[8]
    elif (not index_open) and palm_closed:
        handPose['hand_closed'] = True
        handPose['landmark'] = handLms.landmark[0]

    elif not palm_closed:
        a,b,c,d = handLms.landmark[2].x,handLms.landmark[4].x,handLms.landmark[20].x,handLms.landmark[17].x
        pinky_thum_touch = (a<b<d or a>b>d) and (a<c<d or a>c>d)
        if pinky_thum_touch:
            handPose['pinky_thum_touch'] = True

    if palm_open:
        handPose['hand_open'] = True

    return handPose


last_index = [0,0]

while True:
    results = scan()

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            handPose = pose(handLms)
            isindexing,index,ishandClosed,ispinky_thum_touch = handPose['indexing'],handPose['landmark'], handPose['hand_closed'],handPose['pinky_thum_touch']

            ###################################################################################
            if isindexing:
                if last_index == [0,0]:
                    last_index = [index.x,index.y]
                else:
                    move_cursor(index.x - last_index[0], index.y - last_index[1])
                    last_index = [index.x,index.y]

            ###################################################################################
            elif ishandClosed:
                drag_obj()

            ###################################################################################
            elif ispinky_thum_touch:
                open_obj()


            else:
                last_index = [0,0]



cap.release()
cv2.destroyAllWindows()
