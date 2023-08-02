# 项目结构说明
```
project
│   README.md
|   requirements.txt: 需要的python库
|   train.py:训练主体程序
|   download.py:数据集下载程序
|
└───output: 用于存储模型参数以及可视化结果
│   └───checkpoint： 训练检查点
│   |
|   └─── weight: 最终模型参数
|   
└───dataset: 数据集的txt部分
│   └───RealEstate10K
│       |
|       └───test: 测试集
|       └──train:训练集
|   
└───model: 包括Unet模型、epipolar attention的实现
|
└───utils: 包含diffusion实现以及一些常用工具
|
└───output: 存储模型参数以外的中间输出
|
└───debug: 包括一些和模型结构相关的额外说明信息

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
2. 结果可视化：参见visualization 文件夹
