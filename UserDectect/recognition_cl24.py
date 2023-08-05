import cv2
import mediapipe as mp
import time
import numpy as np
import torch
from torch import nn

class N_Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin_1 = nn.Linear(21, 128)
        self.lin_2 = nn.Linear(128, 512)
        self.lin_3 = nn.Linear(512,128)
        self.lin_4 = nn.Linear(128, 7)
        self.activate = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input):
        x = self.lin_1(input)
        x = self.activate(x)
        x = self.lin_2(x)
        x = self.activate(x)
        x = self.lin_3(x)
        x = self.activate(x)
        x = self.lin_4(x)
        x = self.softmax(x)
        return x

class stream_Reco():
    output_Label = 0
    labels = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',
                6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',
                12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',
                18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z',}
    def __init__(self):
        self.now_p = 0
        self.past_p = 0
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
                            static_image_mode=True,
                            max_num_hands=1,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5,)
        self.mpDraw = mp.solutions.drawing_utils
        self.t1 = time.time()
        self.Time = 0.5
        self.win = "1"

    def winCreate(self, winName):
        self.win = winName
        cv2.namedWindow(winName)
        def nothing(x):
            pass
        cv2.createTrackbar('Time',winName,5,50,nothing)

    def getReco(self, img, results, model):
    #while True:
        self.Time = cv2.getTrackbarPos('Time',self.win)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                sz = []
                for id, lm in enumerate(handLms.landmark):
                    #print(id,lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x *w), int(lm.y*h)
                    test = lm.x * lm.y
                    sz.append(test)
                    cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
            sz = np.array(sz)
            tensor = torch.from_numpy(sz)
            tensor = tensor.to(torch.float32)
            tensor = tensor.unsqueeze(0)

            pred = model(tensor)
            self.now_p = self.labels[pred.argmax(1).item()]
            t2 = time.time()
            if(t2 - self.t1 >= self.Time/10):
                # print("---now",now_p)
                print("time",self.Time)
                if(self.now_p != self.past_p): 
                    self.output_Label = self.labels[pred.argmax(1).item()]   
                    #print("out",output_Label)
                self.past_p = self.labels[pred.argmax(1).item()]
                # print("---past",past_p)
                self.t1 = t2
            cv2.putText(img,'{} '.format(
                self.labels[pred.argmax(1).item()]),(300,450),
                cv2.FONT_HERSHEY_SIMPLEX,2,(255,120,0),3) # 预测字母
        else:
            self.now_p = "S"
            if(self.now_p != self.past_p):
                self.output_Label = "S" 
                #print("out_null",stream_Reco.output_Label)
            self.past_p = "S"
            self.t1 = time.time()
        return self.output_Label

    def display_output(self):
        print(self.output_Label)



