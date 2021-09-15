import cv2
import numpy as np
import time
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.HandDetector(minDetection=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#volume.GetMute()
volume.GetMasterVolumeLevel()
volumeRange = volume.GetVolumeRange()
minVolume = volumeRange[0]
maxVolume = volumeRange[1]
vol = minVolume

while True:
    success, frame = cap.read()
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw = False)

    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2)//2, (y1 + y2)//2

        cv2.circle(frame, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(frame, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(frame, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)

        length =  math.hypot(x2 - x1, y2 - y1)

        # Hand range 50-300, volume range -65-0
        vol = np.interp(length, [50, 300], [minVolume, maxVolume])
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)  

    cv2.rectangle(frame, (50, 150), (85, 400), (0,255, 0), 3)
    size = np.interp(vol, [minVolume, maxVolume], (400, 150))
    cv2.rectangle(frame, (50, int(size)), (85, 400), (0,255, 0), cv2.FILLED)  

    volPercentage = np.interp(vol, [minVolume, maxVolume], [0, 100])
    cv2.putText(frame, f"{int(volPercentage)}", (50, 430), cv2.FONT_ITALIC, 1, (0, 255, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f"FPS: {int(fps)}", (40, 50), cv2.FONT_ITALIC, 2, (0, 255, 0), 5)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()