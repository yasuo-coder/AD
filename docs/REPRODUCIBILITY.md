# 复现与实验要求

## 代码和环境
记录上游/本仓库 Git SHA、dirty patch、Python/PyTorch/CUDA/图库版本、GPU 和依赖锁定信息。
当前环境未确认，不能先填写固定依赖或训练命令。

## 每次实验
记录 Experiment ID、完整配置/命令、seed、数据版本、split hash、标签映射、预处理、
checkpoint 选择指标、停止规则、作业 ID、起止时间、退出码、日志及 checkpoint 路径。
seed 与确定性设置需实际保存，不能把固定 seed 视为完全消除 GPU 非确定性。

## 验证顺序
1. FC 与标签检查。
2. 前向 shape、finite、loss、梯度与图边界检查。
3. 小样本过拟合和单折诊断。
4. 固定协议后的完整训练和消融。
只运行与改动相关的有效验证，不为目录占位写模型测试。

## 指标
报告 Accuracy、Balanced Accuracy、Macro-F1、每类 Precision/Recall/F1、混淆矩阵。
多类 AUC 写明 OvR/OvO、macro/weighted 和缺失类别处理，不能用硬标签代替预测概率。
以验证集选择 checkpoint 与超参；测试集固定后仅用于最终评价。
五折给出各折值与均值/标准差，多种子统计与折间统计分别标识。

## 作业恢复
使用服务器现有调度器、systemd 或 tmux 管理长时任务，记录恢复命令。
checkpoint 保存模型、优化器、调度器、epoch/step、随机状态和配置。
公开文档仅保留汇总；影像、标签明细、身份、密钥、权重留在受控存储。
