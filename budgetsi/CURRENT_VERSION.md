# 当前学校版：参数与实现差异

本说明对应训练源码 `6aa4b05`，固定上游 `ac26e38d6f1572eb027597b48a9f4e01f6915ef8`。用途是让审阅者看清实际训练实现；没有在此发布评测，也没有把旧版与新版视为只改一个因素的复现实验。

## 两边分别是什么

- **旧 same-context 1000 节点版本**：项目自己组织更新批次和梯度累积，调用固定上游的 direct advantage 与 vanilla policy loss。它并非从零自写所有 OPD 数学，也没有走完整的上游 actor 更新入口。
- **当前学校版**：项目仍负责社交场景、采集、节点选择和教师评分；随后通过 `DataParallelPPOActor.update_policy` 使用上游的动态分批、loss 累积与优化器调用。Qwen3.5/LoRA 模型适配和进程管理仍属于项目实现，并非原封不动运行上游完整 Ray/FSDP 脚本。

两者的上游基点相同。关键差异是接入到哪一层，以及实际训练配方不同。

## 参数对照

| 项目 | 旧 same-context 1000 节点 | 当前学校版 | 选择依据 |
|---|---|---|---|
| 学生 / 教师 | Qwen3.5-4B / Qwen3.5-27B | 相同 | 保持模型对照 |
| 初始化 | 原始 4B | 原始 4B，独立轨迹 | 不是从旧 1000 节点继续训练 |
| OPD | sampled-token，K=0 | sampled-token，K=0，only_stu / student_p | 当前两个 context 对照的共同设置 |
| 学习率 | 1e-5 | 1e-5 | 明确保留项目选择；上游脚本默认 1e-6 |
| weight decay | 0 | 0.01 | 当前采用上游 optimizer 默认 |
| 学生采样温度 | 0.7 | 1.0 | 当前对齐上游温度；不是等价变换 |
| 教师评分温度 | 1.0 | 1.0 | 保持 raw 教师概率口径 |
| 训练场景 | 实际 80 个 | 固定 200 个非 test 场景，无 dev 划分 | 学校版明确扩展训练场景覆盖 |
| 预算 | 1000 个有效节点 | 每组目标 3000 个有效节点 | 节点数不等于 optimizer 更新次数 |
| 实际更新次数 | 17 | 随每批有效节点数而变 | 不能仅按节点数认为优化步数相同 |
| loss 累积 | 整个更新批所有有效回答 token 平均 | 动态微批内 token 平均，再按微批回答数占比加权 | 使用上游动态 actor 的实际实现 |
| 学生生成 | vLLM 路径 | HF 带种子的批量生成与同步副本 | 当前学校运行时的实现选择，仍是潜在差异 |

当前其余固定设置：LoRA rank32 / alpha64 / dropout0；AdamW；上下文上限40960；每次采集16段对话；生成微批上限8；正式调度 seed20360915。一个更新批使用这一批被接纳的 OPD 节点，最后一批受剩余节点配额约束；不是固定16个训练回答。逐项配置见 [CURRENT_RECIPE.json](CURRENT_RECIPE.json)。

旧版列值依据归档的 same-context K0 配置与 `opd_update.py` 的更新实现；当前列值直接对应本目录 school_p0 JSON、`school_actor_config` 与固定上游 actor。旧版代码/运行归档不包含在本次同步中，不能用 PR #2 更早的参数草案替代这次旧版配置。

## loss 加权，用直观例子解释

假设一个回答有10个有效 token，另一个有90个：

- 旧版把100个 token 放在一起算平均，因此两个回答分别贡献10%和90%。
- 当前版如果动态分成两个各含一个回答的微批，会先各自平均，再按每个微批占全部回答的1/2加权，分别贡献50%和50%。
- 如果当前版把两条放进同一个微批，就仍会得到10%和90%。因此不能概括成“新版永远每条回答等权”。实际权重受动态分组影响。

精确公式：`sum_g (微批 g 的回答数 / 整批回答数) × 微批 g 内有效 token 的平均 loss`。

为什么这样选：此次学校版的目标之一是复用上游 actor 的默认动态批处理，减少项目外围自行实现累积的差异。因此选择了该上游行为；这**不表示已证明它比旧版更好**，也不是服务器迁移必然要求这样改。温度、weight decay、训练场景与预算也同时变化，不能把质量变化单独归因于 loss 加权。

## 两个 context 组，只改变教师评分能看到什么

- `same_context`：教师评分使用原学生提示的 token 与学生实际生成的目标 token，不加入参考答案。
- `reference_context`：在同一学生消息副本的最后一个 user 内容附加明确标记的教师参考数据块，再由教师评分。

参考候选与选择记录仍用于审计，但 same-context 的教师评分提示不接收它；任何组的学生提示都不接收教师参考答案。严格 prompt/token/context 绑定校验保留在 `social_protocol/runner.py` 和训练入口。

200个训练场景按固定顺序循环，重访时轮换公开的有序角色组合并派生种子。两组初始化分配相同；更新后的 on-policy 对话和最终接纳节点可以不同。不是用历史公开答案回填训练目标。

## 学校执行工程

| 模块 | 职责 |
|---|---|
| `formal_run.py`, `run_state.py` | 采集→评分→更新；有效节点配额；adapter/optimizer/账本提交与恢复 |
| `variant_bridge.py`, `top16.py` | sampled-token K0 接入固定上游 reward、direct advantage、vanilla loss 和 actor 更新 |
| `parallel_collect.py`, `student_replica.py` | HF 批量生成、训练后同步生成副本、快照校验 |
| `score_replica.py`, `gpu_lease.py` | 冻结学生评分副本及 shared/exclusive 前向显存锁 |
| `persistent_teacher.py`, `remote_teacher.py` | 持续教师服务与校验过的概率通信 |
| `formal_gate.py`, `execution_migration.py` | 配置/代码/模型哈希绑定，恢复迁移检查 |
| `telemetry.py` | 训练数值事件与日志；学校额外的 W&B 连续同步进程不在本次范围 |

原学校 GPU 分工：0/1 教师生成，2/3 精确教师打分及冻结学生评分副本，4/5 两组主学生，6/7 学生生成副本。服务共享与锁用于调度，不改变训练数据预算。

### 两处必须披露的运行时修复

1. `identity_division.py` 在 hash 校验固定上游 forward 后，仅跳过温度为1时的 `logits.div_(temperature)`。保留完整张量形状、动态分组、loss 和更新调用；温度不为1或源码不符时拒绝使用。这是运行时替换 forward 的一个操作，不能只因为 `verl/` 文件未改就说运行行为完全未改。
2. `formal_run.py` 加载分词器后，从模型原始 `tokenizer.json` 恢复 backend，避免训练/采样的 Unicode 规则差异。严格逐 token 校验继续保留。

早先 response-only-head 候选数值验收失败，未部署；不能将它记作成功方案。本次同步的是上述实际采用的修复，不是该候选。显存优化不代表任意长度、任意机器均已验收。

## 可复核范围

[SOURCE_SNAPSHOT.json](SOURCE_SNAPSHOT.json) 记录每个复制文件的 SHA256；可与学校源码提交核对。上游框架字节保持固定版本，项目代码独立放在 `budgetsi/`。公开数据文件只有场景/角色初始化与来源信息，没有生成对话、评分或训练目标。

本次检查覆盖 CPU 数值和接线、数据/信息边界、恢复与进程工具测试；没有因同步而重新训练，也不宣称这个合并提交已在 GPU 上完成一次独立复现。部署限制见 [README.md](README.md)。
