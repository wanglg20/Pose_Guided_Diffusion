# 项目结构说明
```
project
│   README.md
|   requirements.txt: 需要的python库
|   h5_generate.ipynb:用于生成数据集的h5文件，可以加快数据集读取速度
|
└───output: 用于存储模型参数以及可视化结果
│   └───checkpoint： 训练检查点
|   |   └───primary: 初次训练的检查点
│   |   │ 
|   |   └───tuning： 采用若干训练技巧后的检查点
|   |   |
|   |   └───experiment： 缺失数据集测试的检查点
│   |
|   └─── weight: 最终模型参数
|   
└───tensorboard: 记录的训练过程
|
└───train:训练集
|
└───test: 测试集
|
└───val: 验证集
|
└───reference: 参考的论文
```

# 使用说明：
0. 配置环境：
```
    pip install -r requirements.txt
```
1. 训练模型：
```
    python train.py
```
2. 测试模型训练结果：
```
    python test.py
```
3. 可视化决策过程：参见CAM_visualization.ipynb
4. 查看训练过程：
```
    tensorboard --logdir tensorboard
```

# command
conda activate zero123
cd C:\Users\Administrator\Desktop\无尽深渊\大三暑假\3D-posed-diffusion\3d_posed_diffusion