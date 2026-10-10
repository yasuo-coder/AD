# BrainPrompt


This is the official PyTorch implementation of BrainPrompt from the paper 
*"	
BrainPrompt: Multi-Level Brain Prompt Enhancement for Neurological Condition Identification"* early accepted by 28th International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI'25).

<img alt="Model" src="figs/framework.png" title="Framework"/>

## Data
All Preprocessed data used in this paper are published in [this paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/44e3a3115ca26e5127851acd0cedd0d9-Paper-Datasets_and_Benchmarks.pdf). 
Data splits and configurations are stored in `./data/` and `./configs/`. If you want to process your own data, please check the dataloader script `./data/BrainNet.py`.

## Usage

Please check `baseline.sh` on how to run the project.

Local Table 2 baseline extensions (GCN, BrainNetCNN, BrainGNN, BNT,
ContrastPool, LG-GNN, and fTSPL): see [implementation differences, prerequisites,
and commands](docs/BASELINES.md), or use `run_table2_baseline.sh`.
These adaptations have not been validated against the paper's reported scores;
fTSPL is a paper-based reconstruction because its public repository has no code.

## Contact

If you have any questions, please feel free to reach out at `jiaxing003@e.ntu.edu.sg`.

## ADNI 标签策略更新（2026-10-04）

根据用户转述的作者邮件，MCI 统一并入 LMCI。五类顺序为 CN=0、AD=1、SMC=2、EMCI=3、LMCI=4；原始数值标签 5 仍归入 4。
新增 `adni_labels.py` 作为文本诊断规范化入口。预处理 manifest 保留 `original_diagnosis`，训练使用规范化后的 `diagnosis` 和 `label`，策略版本为 `mci_to_lmci_v1`。
ADNI_1、ADNI_2 的图、元数据嵌入、索引需一起重建；旧样本顺序的缓存和结果不能混用。新运行日志按策略版本隔离。
完整变更与验证见相邻 BrainPromptPro 项目的 `docs/MCI_LABEL_POLICY.md`；旧实验数据及索引已归档于 `BrainPromptPro/outputs/migrations/mci_to_lmci_v1/`。
