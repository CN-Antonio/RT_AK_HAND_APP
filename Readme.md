# 

## 项目说明
手势分类模型，检测图片中的手势类型  
训练代码：[FireNet](https://github.com/Lebhoryi/FireNet-LightWeight-Network-for-Fire-Detection)
## 目录结构
```
hand_det # RTT项目文件夹
model    # 存放h5/tflite模型
code     # 存放代码
picture  # 存放示例图片
```
> jpg2array 将图片文件resize为64*64，转换为列表供数据输入  

> Train 训练模型-量化为int类型
## 训练说明
由于对11种图像进行分类，修改网络模型最后一层
```model.add(Dense(units=11, activation = 'softmax'))```  
> units=11
## 推理说明
> 代码修改自Cifar10推理项目  
> 保存11张（0~10）手势图片，组10为空白组即不存在手势。  

> 将每张图片进行推理，输出每个类型的置信度与图片属于的组别和推理的组别

## 输出结果
```
msh> hand_app
pred: 250 0 0 0 0 0 0 0 0 0 5
prediction: 0->0

pred: 0 255 0 0 0 0 0 0 0 0 0
prediction: 1->1

pred: 4 199 10 3 0 0 0 4 0 0 36
prediction: 2->1

pred: 0 0 0 238 0 0 0 0 0 0 18
prediction: 3->3

pred: 21 6 33 4 50 2 50 50 9 21 9
prediction: 4->4

pred: 0 0 0 0 0 255 0 0 0 0 0
prediction: 5->5

pred: 7 0 24 1 0 0 206 2 0 0 16
prediction: 6->6

pred: 0 0 0 12 0 0 0 243 0 0 0
prediction: 7->7

pred: 1 7 1 1 10 0 0 16 204 16 1
prediction: 8->8

pred: 0 0 0 2 0 0 0 0 0 127 127
prediction: 9->9

pred: 0 0 0 0 0 0 0 0 0 0 255
prediction: 10->10
```