# 架构与候选方法
状态：概念规划，尚未导入或验证实现。

## 基线审查
核实 FC 与 ROI 语义输入、个体 GNN、subject readout、population graph 和预测头的实际接口。
记录原论文与上游 commit；不能把会话概述替代代码审查。

## 候选
| 模块 | 假设 | 必要验证 |
| --- | --- | --- |
| 语义引导自适应脑图 | FC、ROI 语义及个体特征共同调节边权 | 边权梯度、稀疏度、FC-only/semantic-only 消融 |
| 疾病感知语义读出 | disease prompt 为 ROI 权重提供语义条件 | 不使用样本真实标签选择 query；与平均池化比较 |
| 学习化 population graph | 可学习相似度改进个体间信息传播 | 防标签泄漏、批次依赖与 inductive/transductive 核验 |

三个方向不是已实现或已证实创新；核对 BrainPromptPlus 排除边界后才进入实施。
文本编码器候选为 encoder-only；具体模型、许可证、版本、冻结或后训练策略待审计。
额外候选单独记录研究假设、来源和消融；依据验证集选择最终三个。

## 实现约定
models/ 保存网络；training/ 保存训练与评估；configs/ 保存配置；data/ 保存加载与校验代码；
scripts/ 保存运行入口。均在 BrainPromptPro/ 下。
保留自定义数据路径及已有 --split 接口需先核对实际项目。
接口定义至少覆盖输入 shape、dtype、device、输出 logits、类序、图 batch、loss 与梯度。
