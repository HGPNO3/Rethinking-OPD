# 参数审核：上游默认 → BudgetSI 已有实际配置

来源固定于 `../UPSTREAM.json`；两组实际科学配置见 `../configs/`。
这些 JSON 是完成运行的脱敏配置快照，不是新实验批准，也不是 verl 可直接读取的配置。

| 参数 | 上游主脚本默认 | BudgetSI 当前实现 | 审核含义 |
|---|---|---|---|
| 学生 / 教师 | R1-Distill-Qwen-1.5B / JustRL-DeepSeek-1.5B | Qwen3-4B / 14B；Qwen3.5-4B / 27B | 两组是不同模型家族，不是纯教师大小消融 |
| thinking | 原脚本未显式关闭 | 全部生成与评分非 thinking | 模板边界仍保留，不能生成后删思考 |
| 学习率 | 1e-6（原脚本硬编码） | 1e-5 | 本提交使 LR 真正可覆盖；不是声称原值不合理 |
| 更新 | 默认无 LoRA | BF16 LoRA r32/alpha64/dropout0 | Qwen3.5 仅文本骨干线性层；不能默认 all-linear |
| 优化器 | 依赖 verl 配置 | AdamW .9/.999，eps1e-8，wd0，clip1 | 核验最终优化器，不能只看 shell |
| rollout temperature | 1.0 | .7 | 行为分布与 raw-policy 概率分开处理 |
| teacher probability temperature | 1.0 | raw 1.0 | 与候选生成 .7 不同 |
| log_prob_top_k | 16 | 仅原始已采样 token 的精确概率 | 不是生成 top-k，也不是完整分布 KL |
| 每 prompt 样本 | 4 | 每行动单次生成；每唯一分支一次 partner 回复 | 对话数与训练节点数不可直接换算 |
| batch | 64 prompts × parallel size | 16 完整对话，入选节点数可变 | 整批有效 token 归一化，一次更新 |
| 长度 | prompt1024 / response7168 | 服务上下文40960，剩余上下文约束生成 | 不能未经验证把对话截到1024 |
| 教师输入 | 上游输入路径 | 专业提示 + 可见历史 + 入选参考 + 原始逐 token 前缀 | 候选生成与 OPD 信息边界不同 |
| 目标 | top-k 蒸馏默认配置 | 原始行动 sampled reverse-KL surrogate | 两者不自动数学等价 |
| 规模 | 1 epoch | 1000后继续到3000有效节点，保留optimizer | 不把新数据和更新次数视为单一因素消融 |
| 评价 | 数学验证任务；test_freq=-1 | 独立 SOTOPIA 七维，固定初始 partner | IG 不是社交充分性验证，也不是最终judge |

## 本提交实际改动

主脚本的模型、数据、长度、batch、GPU、学习率和 LoRA 配置改为可覆盖；默认 dry-run 输出真实命令参数，不启动 Ray/模型，不执行原先的全局 `ray stop --force`。
`OPD_DRY_RUN=0` 是上游直接执行入口，仅用于另行批准的基线运行。本次未执行。

`review_config.py` 从当前两组配置生成**单教师固定 prefix 基线提案**，显式要求 batch 与长度分配，避免把16对话伪装成16节点。输出只有最终 argv，不冒充 Hydra 完整解析配置。示例参数只是 CPU 检查用例：

```bash
python3 budgetsi/review_config.py --family qwen3 --train-file /data/train.parquet --validation-file /data/val.parquet --max-prompt-length 1024 --max-response-length 1024 --batch-size 16 --gpus 1
```

该提案保留上游损失/数学任务代码；不能作为已运行的社交 baseline。Qwen3.5 在旧版 verl 中的架构、LoRA目标层支持未验证。完整社交实现由第二次提交独立提供，避免默默改变研究方法。
