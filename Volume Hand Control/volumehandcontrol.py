import numpy as np
import time
import cv2
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities,IAudioEndpointVolume
import math
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# volume control library usage
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minvol = volRange[0]
maxvol = volRange[1]
vol = 0
volbar = 400
volper=0

wCAM, hCAM = 640 ,480

cap = cv2.VideoCapture(0)
cap.set(3, wCAM)
cap.set(4, hCAM)
PTime=0


with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

  while True:
    success, img = cap.read()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            img,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
    lmList = []
    if results.multi_hand_landmarks:
      myHand = results.multi_hand_landmarks[0]
      for id, lm in enumerate(myHand.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        lmList.append([id, cx, cy])     

    if len(lmList) != 0:
        print (lmList[4],lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx,cy = (x1+x2)//2, (y1 + y2) // 2
        cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
        cv2.circle(img,(x2,y2),15,(255,0,255),cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
        cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)

        length = math.hypot(x2-x1,y2-y1)
        print(length)
        
        #hand range 50 - 300
        # volume Range -63 - 0

        vol= np.interp(length , (50,300) , [minvol, maxvol])
        print(int(length),vol)
        volume.SetMasterVolumeLevel(vol,None)
        volbar= np.interp(length , (50,300) , [400,150])
        volper= np.interp(length , (50,300) , [0,100])
        if length<50:
           cv2.circle(img,(cx,cy),15,(0,255,0),cv2.FILLED)
    # volume Bar
    cv2.rectangle(img,(50,150), (85,400),(0,255,0),3)  
    cv2.rectangle(img,(50,int(volbar)), (85,400),(0,255,0),cv2.FILLED)        
    cv2.putText(img,f':{int(volper)} %',(40,450),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)   

    cTime = time.time()
    fps = 1/(cTime-PTime)
    PTime=cTime

    cv2.putText(img,f'FPS:{int(fps)}',(40,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
    
    cv2.imshow("img", img)
    cv2.waitKey(1)