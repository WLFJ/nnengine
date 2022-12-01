# NNEngine

BUAA 2022 硕士高等软工课程设计

## 项目结构

```
.
├── src
│   ├── core                     张量、模型等核心接口
│   │   ├── nn.py
│   │   └── tensor.py
│   ├── data                     数据
│   │   └── dataloader.py
│   └── utils                    训练、测试工具
│       ├── evaluator.py         评估器
│       ├── optimizer.py         优化器
│       └── trainer.py           训练器
├── test.py                            测试
└── usr                                自订模型、数据集
    └── mnist_dataset.py
```
