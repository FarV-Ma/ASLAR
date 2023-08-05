# ASL Alphabet Reconition & Robot

A robot system with the controller based on target identification of sign language.

一种基于目标识别的手语控制机器人。

## File Define 各文件定义

### TrainSever：训练端源代码

> ASL_Alphabet_Train：完整训练集
> 
>  Model：导出的模型
> 
> > DatasetSpawner.ipynb：数据集预处理-封装器
>
> > DatasetSpawner-Lite.ipynb：轻量版数据集预处理-封装器
>
> > NetworkTrainer.ipynb：训练器
>
> > NetworkTrainer-Lite.ipynb：轻量版训练器
 

### UserDetect：识别遥控端源代码

> ros_hand-lite：识别端主程序组（轻量版）
> ros_hand：识别端主程序组（完整版）
> recognition_cl：识别-预测模块组

### RobotNUC：机载电脑端源代码

> 详情请见包内说明。


## How-to-use 各端环境布置与启动方法

This part is now only in Chinese. English version will be added later.

### 一、训练端

#### 1. 环境需求

环境库：Python3.7或以上、OpenCV4、MediaPipe0.9、Python CSV库、Pandas、matplotlib、numpy、Pytorch-CPU、TorchSummary、Jupyter；

容器：Anaconda（建议）

编辑器：VSCode（建议）

#### 2. 配置说明

建议通过Anaconda，新建环境容器，并通过conda install命令及pip3安装上述库。

#### 3. 启动说明

通过conda的active link，在VSCode中打开对应Jupyter Notebook后，点击“Run All”即可。

### 二、识别遥控端

#### 1. 环境需求

环境库：Python3.7或以上、OpenCV4、MediaPipe0.9、numpy、Pytorch-CPU、TorchSummary、Jupyter、roslibpy；

容器：Anaconda（建议）

编辑器：VSCode（建议）。若使用VSCode，请安装ROS Plugin。

#### 2. 配置说明

建议通过Anaconda，新建环境容器，并通过conda install命令及pip3安装上述库。

#### 3. 启动说明

打开ros_hand.py后，将ROS对端主机IP地址根据实际情况修改，等待机器人机载电脑端的roscore启动并订阅相关话题后，运行即可。注意：文件夹路径不能有中文。

### 三、机载电脑端

#### 1. 环境需求

系统：Ubuntu16 LTS

ROS：ROS1 Kinetic及其封装的系列库、mbot系列包、gazebo

#### 2. 配置说明

建议通过apt包管理器命令及pip3安装上述库。

#### 3. 启动说明

通过catkin_make对相关工作空间进行编译后，新建两个终端，分别打开以下两个launcher，以先后打开ros桥的websocket通信服务器与机器人仿真环境：

'''
roslaunch rosbridge_server rosbridge_websocket.launch
roslaunch mbot_gazebo mbot_laser_nav_gazebo.launch
'''
