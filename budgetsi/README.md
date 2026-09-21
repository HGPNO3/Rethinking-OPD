# BudgetSI：当前学校版训练实现

本目录同步实际学校训练版本 **`6aa4b05`**（2026-09-21）。上游固定为 `ac26e38d6f1572eb027597b48a9f4e01f6915ef8`，来源为清华 THUNLP OPD / Rethinking-OPD。此目录是项目的社交任务接入层；`verl/` 和根目录上游启动脚本保持原样。

**师兄审阅入口：[当前配置与旧版差异](CURRENT_VERSION.md)**。机器可读参数见 [CURRENT_RECIPE.json](CURRENT_RECIPE.json)，逐文件来源校验见 [SOURCE_SNAPSHOT.json](SOURCE_SNAPSHOT.json)。

阅读顺序：

1. [CURRENT_VERSION.md](CURRENT_VERSION.md)：当前采用什么设置、与旧版有哪些差异、为什么选择这些设置。
2. [SCHOOL_ACTOR_ALIGNMENT.md](SCHOOL_ACTOR_ALIGNMENT.md)：上游 actor 的动态微批与 loss 加权细节。
3. [formal_run.py](formal_run.py) → [variant_bridge.py](variant_bridge.py) → [top16.py](top16.py)：在线采集、教师打分、actor 更新及断点恢复。
4. [social_protocol/runner.py](social_protocol/runner.py) 与 [school_data.py](school_data.py)：信息边界、教师上下文与训练场景调度。

本次交付包含训练代码、必需的公开场景初始化、配置快照及 CPU 检查。不包含社会效果评测脚本、评测结果、生成对话、模型权重、运行日志或凭据。PR #2 的早期参数审查继续保留在原 PR 中；这里的参数已经由实际学校版确定。

## 代码检查

在仓库根目录运行（只执行 CPU 单元检查，不调用模型/API）：

```sh
uv run --no-project --with torch==2.10.0 --with hydra-core==1.3.2 --with numpy --with peft==0.21.0 --with transformers==5.3.0 python -m unittest discover -s budgetsi/tests -p 'test_*.py'
PYTHONPATH=budgetsi/social_protocol uv run --no-project --with torch==2.10.0 --with hydra-core==1.3.2 --with numpy --with peft==0.21.0 --with transformers==5.3.0 python -m unittest discover -s budgetsi/social_protocol -p 'test_*.py'
```

具体本次检查范围见 [SYNC_VALIDATION.md](SYNC_VALIDATION.md)。CPU 测试使用固定上游函数体和小型张量，不等同于重新跑过完整 GPU 训练。

## 部署边界

`deployments/school_p0/*_formal.json` 是原部署配置快照，保留模型/运行时路径、文件哈希与本地端口供核对，**不是任意机器上直接可用的安装包**。`*_acceptance.json` 是工程验收配置，不是正式训练数据或已通过验收的证明。启动需要外部的 hash-bound approval 文件，仓库不分发该文件，也不自动启动任何任务。

若换机器，需要准备模型和兼容的训练/采集/教师运行时，核验模型、tokenizer、SOTOPIA 文件哈希，重新绑定当前 Git 提交和目标环境。`formal_gate.py` / `execution_migration.py` 仍有学校专用的 lease 路径与日志目标校验；不能仅替换 JSON 路径便宣称完成迁移。此次同步保留执行代码字节，未绕过这些门禁。教师/学生后台服务的学校调度控制器不在本次快照中。

`formal_config.json`、`variants/` 与非 school 的训练 manifest 是训练模块保留的旧配置/测试依赖，**当前版本请只以 CURRENT_RECIPE 和 school_p0 配置为准**。`adapter.py` / `register_loss.py` / `verify.py` 是早期接口与检查工具，学校正式更新使用 `variant_bridge.py` → `top16.py` → 上游 actor。
