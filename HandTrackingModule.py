import mediapipe as mp
import cv2
import time

class HandDetector():
    def __init__(self, mode = False, maxHands = 2, minDetection = 0.5, minTracking = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.minDetection = minDetection
        self.minTracking = minTracking
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.minDetection, self.minTracking)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw = True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        #print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
                for handLms in self.results.multi_hand_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
        return frame
    
    def findPosition(self, frame, handNo = 0, draw = True):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 4, (255, 0, 0), cv2.FILLED)
                    cv2.putText(frame, str(id), (cx + 5, cy + 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        return lmList

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0
    handDetector = HandDetector()

    while True:
        success, frame = cap.read()

        frame = handDetector.findHands(frame)
        lmList = handDetector.findPosition(frame)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(frame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()