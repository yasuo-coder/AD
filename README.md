# AD- / BrainPromptPro

BrainPromptPro：基于 ADNI 静息态 fMRI 功能连接与语义提示的阿尔茨海默病分类研究。

当前交付为项目目录及长期上下文管理体系。尚未导入 BrainPrompt 源码、数据或模型，不能直接启动训练。

## 快速恢复上下文
打开仓库根目录，让 Codex 读取 [AGENTS.md](AGENTS.md)，然后读取 [STATE.md](STATE.md)。
使用 [会话启动模板](docs/SESSION_START.md) 提交本次具体任务。

## 文档导航
| 文件 | 职责 |
| --- | --- |
| [CONTEXT.md](CONTEXT.md) | 研究背景、目标与约束 |
| [STATE.md](STATE.md) | 最新状态、阻碍、下一步 |
| [ROADMAP.md](ROADMAP.md) | 阶段目标与验收标准 |
| [DECISIONS.md](DECISIONS.md) | 决策、理由与证据 |
| [EXPERIMENTS.md](EXPERIMENTS.md) | 实验索引 |
| [数据协议](docs/DATA_PROTOCOL.md) | 数据、标签、划分、泄漏检查 |
| [架构](docs/ARCHITECTURE.md) | 候选方法及接口 |
| [复现要求](docs/REPRODUCIBILITY.md) | 环境、版本与实验记录 |
| [plans/](plans/) | 活跃执行计划 |
| [BrainPromptPro/](BrainPromptPro/) | 新开发代码目录 |

## 工作方式
每个里程碑：恢复上下文 → 执行计划 → 验证 → 更新文档 → 提交对应改动。
STATE 保持最新快照；决策与实验按 ID 追加；详细日志保存在实验服务器。
数据、个人信息、密钥和模型权重不提交本仓库。仓库目前为公开仓库。
