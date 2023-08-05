import recognition_cl as recoMod
import cv2
import mediapipe as mp
import time
import numpy as np
import torch
from torch import nn
import roslibpy

class N_Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin_1 = nn.Linear(21, 128)
        self.lin_2 = nn.Linear(128, 512)
        self.lin_3 = nn.Linear(512,128)
        self.lin_4 = nn.Linear(128, 7)
        self.activate = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    # def __init__(self):
    #     super().__init__()
    #     self.lin_1 = nn.Linear(21, 128)
    #     self.lin_2 = nn.Linear(128, 512)
    #     self.lin_3 = nn.Linear(512,128)
    #     self.lin_4 = nn.Linear(128, 25)
    #     self.activate = nn.ReLU()
    #     self.softmax = nn.Softmax(dim=1)
        
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

streamReco = recoMod.stream_Reco()
cap = cv2.VideoCapture(0)
model = torch.load('./asl_model_lite-single.pt')
# model = torch.load('./asl_model_lite-single.pt')
RecoWin = "ASL Alphabet Recognization Module"

guideImg = cv2.imread(r'./Guide.jpg')
GuideWin = "Guide"

streamReco.winCreate(RecoWin, True)
streamReco.winCreate(GuideWin, False)
cv2.imshow(GuideWin, guideImg)

# 定义ROS应用层对端
client = roslibpy.Ros(host='192.168.16.130', port=9090)
client.run()

# 创建话题发布者
topic = roslibpy.Topic(client, '/keys', 'std_msgs/String')
topic.advertise()

while cap.isOpened():
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgRGB.flags.writeable = False
    results = streamReco.hands.process(imgRGB)
    outputLabel = streamReco.getReco(img, results, model)
    print("Readed output is:",outputLabel)
    cv2.imshow(RecoWin, img)
    # 发布手势识别结果到 ROS 话题
    message = roslibpy.Message({'data': outputLabel})
    topic.publish(message)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按下q键直接关闭
        break
cap.release()
cv2.destroyAllWindows()

# 断开与 ROS 的连接
client.terminate()