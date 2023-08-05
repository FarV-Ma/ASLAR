import cv2
import mediapipe as mp
import time
import numpy as np
import torch
from torch import nn

# 模型网络重构类
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

# 视频流目标检测-目标识别类
class stream_Reco():
    # 输出变量
    output_Label = "null"
    # 识别结果标签转译字典 
    labels = {1:'A',2:'C',3:'D',4:'F',5:'L',6:'Y',} 
    # 内部类变量初始化函数
    def __init__(self):
        self.now_p = 0                      # 当前手势缓存变量
        self.past_p = 0                     # 前序手势缓存变量
        self.mpHands = mp.solutions.hands   # 初始化MP Hands解决方案
        # MP Hands手部目标检测与特征点识别参数初始化
        self.hands = self.mpHands.Hands(
                            static_image_mode=True,      # 逐帧读取静态图像
                            max_num_hands=1,             # 仅识别单手
                            min_detection_confidence=0.5,# 目标检测置信度
                            min_tracking_confidence=0.5,)# 特征点识别置信度
        self.mpDraw = mp.solutions.drawing_utils    # 初始化特征点可视化绘制器
        self.t1 = time.time()               # 初始化防误触比对时间戳1
        self.Time = 5                       # 防误触滤波默认时间间隔500ms
        self.win = "1"                      # 目标窗体变量
    
    # 可视化窗口创建函数
    def winCreate(self, winName, barCreate):
        self.win = winName          # 传入目标窗体
        cv2.namedWindow(winName)    # 创建窗口
        # 防卡死函数
        def nothing(x):    # 当输入图像流为空
            pass           # 跳出卡死状态，直达destroy
        # 根据传参判断是否创建防误触时间感度滑条
        if barCreate:
            # 创建一条名为Time, 目标窗体为winName
            #   默认值5（百ms），最大值50（百ms）的滑条
            cv2.createTrackbar('Time',winName,5,50,nothing)

    # 目标识别-预测模块
    def getReco(self, img, results, model):
        # 从目标窗体获取设定的时间感度值
        self.Time = cv2.getTrackbarPos('Time',self.win)
        # 当目标检测结果不为空时
        if results.multi_hand_landmarks:
            # 对手部特征点识别结果逐个读取
            for handLms in results.multi_hand_landmarks:
                sz = []     # 初始化读取用的空数组
                # 对每个特征点参数逐个读取
                for id, lm in enumerate(handLms.landmark):
                    # 从输入图像获得绝对坐标系数
                    h, w, c = img.shape        
                    # 同时将相对坐标值相乘
                    multiplyLandmark = lm.x * lm.y
                    # 并将其写入到读取用的数组
                    sz.append(multiplyLandmark)
                    # 将读取到的相对坐标映射到绝对坐标备用
                    cx, cy = int(lm.x *w), int(lm.y*h)
                    # 借助映射的绝对坐标绘制可视化特征点
                    cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)
                # 连接可视化特征点为可视化特征骨架
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
            # 将读取的特征点乘积集转换为numpy格式
            sz = np.array(sz)
            # 再将特征点乘积集张量化
            tensor = torch.from_numpy(sz)
            # 转换特征点乘积张量集数据格式
            tensor = tensor.to(torch.float32)
            # 张量维度降维
            tensor = tensor.unsqueeze(0)
            # 张量输入到模型以进行预测
            pred = model(tensor)
            # 当前手势缓存变量即转译后的当前识别结果
            self.now_p = self.labels[pred.argmax(1).item()]
            # 初始化防误触比对时间戳2
            t2 = time.time()
            # 若前后时间戳差满足时间感度阈值
            if(t2 - self.t1 >= self.Time/10):
                # 打印当前设定的时间感度
                print("time",self.Time)
                # 若前后识别结果改变
                if(self.now_p != self.past_p): 
                    # 将识别结果覆写为当前手势
                    self.output_Label = self.now_p 
                # 更新前序手势缓存变量
                self.past_p = self.now_p
                # 更新防误触比对时间戳
                self.t1 = t2
            # 识别结果可视化
            cv2.putText(img,'{} '.format(
                self.labels[pred.argmax(1).item()]),(300,450),
                cv2.FONT_HERSHEY_SIMPLEX,2,(255,120,0),3) 
        # 反之则目标检测结果为空
        else:
            self.now_p = "null"
            if(self.now_p != self.past_p):
                self.output_Label = "null" 
            self.past_p = self.now_p
            self.t1 = time.time()
        return self.output_Label

    def display_output(self):
        print(self.output_Label)



