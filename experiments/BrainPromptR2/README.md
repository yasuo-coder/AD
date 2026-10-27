# BrainPrompt R2

独立 R2 工程：只读历史诊断 → FIT 内重复验证 → 单参数先验校正 → 从头提示拆解。
沿用 895 人/1323 扫描与 MCI→EMCI。外折 TEST 仍封锁，不能根据历史 TEST 选型。
本次先发布方案与实际诊断，再启动经门槛允许的内部训练；外折评价等待 Kimi 审阅。
## Execution

The protocol is fixed to seed42 outer-fold0 FIT, 715 subjects / 1051 scans, with three repeats of three subject folds. See the root repository's `BrainPromptMaxFusion_R2_优化方案.md` and `configs/study.json`.

Private ADNI inputs and outputs are intentionally not distributed. On the authorized workstation:

```bash
/home/panyao/miniconda3/envs/brainprompt/bin/python -B -m r2.prepare
/home/panyao/miniconda3/envs/brainprompt/bin/python -B -m r2.seal
/home/panyao/miniconda3/envs/brainprompt/bin/python -B -m unittest discover -s tests -v
/home/panyao/miniconda3/envs/brainprompt/bin/python -B -m r2.verify_launch
```

Publish the plan and diagnostics first, record the real Git commit in `public/publication.json`, then launch `python -B -u -m r2.controller --gpus 0,4,5`. Each worker uses the allowlisted bubblewrap environment. `runs/fixed_fit_v1/status.json` records progress; `runs/fixed_fit_v1/public/` holds internal results. Recovery invokes the same controller and verifies manifests. There is no outer TEST evaluator.

The controller ends at `AWAIT_KIMI_REVIEW`, or explicitly records a failed fit / collapse gate. It never treats a running or incomplete study as finished.
