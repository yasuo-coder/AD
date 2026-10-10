# BrainPromptPro 实现
现已导入真实源码。保留`nets/`、`layers/`、`data/`、`cli.py`和独立`research_runner.py`；原占位models/training不代表实际入口，不嵌套同名包。
`llm-main.py --config ... --split 0`保留旧单折接口；`candidate_v3_initialized_runner`为修订population正式研究入口。当前冻结研究继续从原服务器工程执行；不要因接入在克隆再训练一次。
数据、prompt/LLM权重、checkpoint、split明细不在Git。configs为已有服务器路径模板；路径映射和固定协议见根docs/REPRODUCIBILITY.md。历史文档已移至docs/historical并明确版本范围。
公开合成验证：仓库根`python -B -m tests.run_integration_suite`。历史数据依赖测试不属于公共无数据测试集。
