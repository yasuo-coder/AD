# BrainPrompt 来源与许可状态
上游：https://github.com/AngusMonroe/BrainPrompt 。原服务器checkout HEAD=`c85304a`（完整SHA见docs/provenance/source_import_manifest.json），原分支main含本地修改，未reset/stash/commit或移动。
此目录是必要源码与配置的导入，不是完整含数据上游镜像。保留原作者README和源码标注。官方nets/brainpromptg_net.py等核心与上游逐文件对照见baseline_import_comparison.json；loader/CLI/labels/其他baseline的原有修改保留。
在该上游commit根目录未发现LICENSE/COPYING文件，不能推断MIT/Apache或给其重新授权。额外baseline原有third_party/licenses中许可原文已保留；使用者须分别遵守上游许可/授权要求。本接入不替作者声明新的许可。
未导入官方样本索引、患者metadata prompt、任何.pt嵌入/权重、数据或图像。原可运行baseline仍在`/home/panyao/AD/BrainPrompt`，公开源码需要合法提供私有工件后方可运行。
