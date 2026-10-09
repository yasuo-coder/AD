# 项目背景
更新：2026-10-09

## 长期目标
在 BrainPrompt 开源方法基础上，研究语义提示与图学习如何改善 ADNI 多类别疾病分类。
评估 Accuracy、Macro-F1、Balanced Accuracy、每类 Recall、混淆矩阵及适用的多类别 AUC。

## 用户已明确的要求
- 数据由 ADNI_1、ADNI_2、ADNI_3、ADNI_4、ADNI_5 合并组成。
- 模型输入为 rs-fMRI 时间序列计算得到的 FC 矩阵，不直接输入扫描图像。
- 新代码全部放在 BrainPromptPro/。
- 关注语义引导自适应脑图、疾病感知读出、学习化 population graph；允许补充候选并通过证据选择最有效的三个。
- 不采用 BrainPromptPlus 提出的方法；具体排除范围需要论文和代码核验。
- 不同访视可作为独立样本，样本单位与评估划分另见数据协议。

## 待核验的历史背景
过去对话报告过全部预测 CN 的分类塌缩，且 Accuracy 等于多数类比例。
本仓库尚未取得原日志和 checkpoint，不能把这些报告当作本次复现实验结果。
当前类别定义、样本数、划分和已有优化实现均待代码与数据核验。

## 来源与界限
上游候选：https://github.com/AngusMonroe/BrainPrompt
来源目前为用户会话及仓库初始化检查；尚未阅读并导入上游论文和源码。
不预先宣称任何候选有效，也不预先锁定三项最终创新。
