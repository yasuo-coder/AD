# 数据协议

当前七批独立研究见[ADNI_1–7审计](research/adni_1to7/DATA_AUDIT.md)、[方法与隔离](research/adni_1to7/METHOD_AUDIT.md)及计划004。1323FC/895人，固定新sample-level协议及独立subject-disjoint补充协议；私有成员和完整协议仅在服务器dataset_v1。以下998FC内容是历史五批协议，不能作为七批配置。

## 已核验的五批历史协议
依据服务器 `config.json`、`protocol.json`、data audit及冻结split；公开镜像在 `BrainPromptPro/configs/research_snapshot_20261007/`，不含划分明细。

- 数据根 `/data2/panyao_data/ADNI`，ADNI_1→5顺序拼接，各批subject/visit排序；一张AAL116 Pearson FC对应一次rs-fMRI acquisition，FC为116×116、float32，对称有限，保留负相关，对角为相关矩阵自相关。不是原始MRI输入；T1配对不额外算FC。
- 998 FC，685 biological subjects，216人多访视；CN626/AD41/SMC39/EMCI49/LMCI243。AD 4.108%。各批FC200/198/200/200/200，AD4/4/11/7/15。没有把目录当独立训练集。
- 原CSV影像行数不能当FC数；ADNI1–4的8/8/22/14条AD影像包含配对T1。两张缺少可核验映射的原始功能扫描排除，一张不在下载清单的替代扫描不补入；私有ID只留服务器。
- 标签使用下载CSV Group；CN=0/AD=1/SMC=2/EMCI=3/LMCI=4，MCI→LMCI（191条原MCI）。疾病prompt和输出同序。79原始来源冲突、合并后58冲突采用统一既定来源约定，非独立临床裁决。
- 严格检查同一扫描/Image ID、时序内容、FC内容；不同visit不自动去重。此队列未发现重复/无效FC。
- 固定seed42分层划分，train798 [500,33,31,39,195]，val/test各100 [63,4,4,5,24]。只由划分器内部打乱，三训练seed共用同一划分。
- split SHA256 `0a2245d4febd3f187291e78aec7a367066f18c43dc310ab0e426e97bde793d6e`；bundle SHA256 `a93f450ad99b5574bdbddf57993f2c4ea21a4253e2e40d51c79c918499f857c7`。
- Train–Val/Train–Test/Val–Test共享biological subjects=46/38/6。用户允许sample-level主协议，不能声称未见受试者泛化。ADNI1–4历史研究暴露已记录，不是新外部队列。
- age/sex来自下载metadata；site由历史ADNI_2/pipeline_logs/ADNI_metadata.csv按Image ID对齐，未独立核验真实采集site。source×site Cramér's V=.969、site×diagnosis=.340仅描述关联。Source_Dataset不进入分类特征/关系。
- 元数据编码、类别权重、M3参考库及V3 PCA仅训练集拟合。M3参考798 TRAIN样本、当前样本自环屏蔽；不传播query-query。所有疾病query同时使用，无真实标签选择prompt。

私有bundle：`/home/panyao/AD/BrainPromptPro/outputs/data/adni12345_download_mci_to_lmci_v1_20261007/cohort.pt`；划分：研究目录`splits/ADNI_Combined`。这些文件不上传；只保存哈希、分布和恢复规则。

七批跨协议边界：sample与subject两份独立划分之间存在样本TRAIN/VAL/TEST角色交叉。主模型选择仅用sample VAL，补充结果单独报告；协议内holdout不等于整个项目历史上完全未使用。交叉计数与成员审计仅保存在N/reports/cross_protocol_exposure_20261011.private.json；不以“test未评估”暗示全研究从未用过这些FC。
