import pyautogui

screen_w, screen_h = pyautogui.size()


class ControlCursor:
    def __init__(self):
        self.lastPosition = [0,0]

    def moveCursor(self, position):
        if self.lastPosition == [0, 0]:
            self.lastPosition = [position.x, position.y]
        else:
            xDir, yDir = position.x - self.lastPosition[0], position.y - self.lastPosition[1]
            x, y = pyautogui.position()

            x, y = x + xDir * screen_w, y + yDir * screen_h
            x = max(10, min(x, screen_w - 10))
            y = max(10, min(y, screen_h - 10))

            pyautogui.moveTo(x, y, duration=0.05)
            self.lastPosition = [position.x, position.y]
