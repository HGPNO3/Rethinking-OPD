# 完整方法代码迁移与审查顺序

## 迁移的含义与边界

本提交将已运行的 BudgetSI 社交训练实现迁入 OPD fork，源文件逐字节保留，并提供共同代码 + 两家族 fresh/continuation 控制器组装工具。**这是完整训练路径的源码迁移，不是把该路径改写为 verl/FSDP 后端。** 上游数学单教师路径与社交路径各自独立，不能称其数学行为等价。

为了避免改变当前 pipeline，本提交不将原始行动监督改成候选 SFT，不将两教师选择换成固定教师，不把动态对话 flatten 后冒充在线交互，也不将正 IG 称为社会充分性验证。

## 先检查这条链

1. `method/adapter.py`：SOTOPIA 0.1.5 官方环境/行动模板；A/B各自可见历史，规则终止20轮/2 stale。
2. `method/runner.py::teacher_messages`：候选生成看可见历史及原始行动；表达保留意图，策略允许改变行动。没有 partner 私密资料或未来信息。
3. `runner.py::Pilot`：原始行动和两个合法不同候选各接一次新 partner 回复；相同行动去重，原对话继续原行动。
4. `runner.py::select`：候选自身 IG>0；表达还需更短，策略还需优于原行动；取(IG高、完整行动token低)前沿后最大IG/token，精确并列按种子选择。
5. `runner.py::opd_messages / validate_reference_record`：入选参考只给教师；参考不是已发生的事件。候选后的 partner 回复不进入OPD提示。目标始终是学生原始行动token。
6. `method/update_batch.py`、`update_batch_qwen35.py`：精确目标token概率、BF16 LoRA、AdamW，整批token归一化，一批一次更新，optimizer连续保存恢复。Qwen3.5单独处理架构与文本骨干目标层。
7. `variants/*/fresh/train.py`：先真实验收再从初始模型训练；`continuation/train.py` + `continuation.py`：保留1000节点历史、adapter/optimizer lineage到3000。历史被改动则拒绝恢复。
8. `protocol_gate.py`、`token_contract.py` 与测试：精确Git/config/manifest绑定、原始token边界、失败保留。Qwen3.5的格式错误隔离独立保留，不修复输出、不质量重采样。

## 与上游的重要差异，不能通过改参数抹平

- 上游主入口为`on_policy_distillation.sh` -> `verl/verl/trainer/main_ppo.py` -> `ppo/ray_trainer.py`，默认top-k=16；BudgetSI为官方社交环境逐行动分支选择后更新原始token。
- BudgetSI loss为 `-sum(exp(current_raw - behavior_tempered) * stopgrad(teacher_raw - old_hf_raw)) / total_tokens`。它是已选择样本上的一阶surrogate，不是完整trajectory KL，也不是full-vocabulary Forward-KL。
- 生成温度.7，而teacher/old/current评分使用raw概率；分母是实际行为概率。要移植到verl必须复核温度、importance correction、PPO clipping、epoch数、分批token归一化、EOS mask和optimizer步数，不能仅设置`log_prob_top_k=0`就宣布相同。
- 当前选师依据局部IG代理，社交充分性是研究目标/最终评价问题。没有新增独立质量gate，也没有完整续演到结局的候选验证。
- 16个对话产生可变数量入选行动。不能与上游64个prompt或每prompt4条response直接对齐。

## 本次验证与未验证

`CPU_VALIDATION.json`记录此次真正执行的测试；`SOURCE_MANIFEST.json`逐文件绑定原始已运行源码。来源哈希证明迁移忠实，不证明新机器/GPU环境已验收。
没有启动GPU、调用付费API或重跑实验；没有声称本fork复现过既有结果。上游完整Hydra配置与旧verl对Qwen3.5支持尚未验收，见参数说明。

## 下一步统一到verl时的验收条件

先让师兄确定基础损失/概率协议差异是否有意保留，再实现SOTOPIA交互rollout、每样本teacher-only提示、原始token打分和整批一次更新。以相同固定token轨迹比较teacher/old/current logprob、loss、梯度、一次参数更新和保存恢复。端到端GPU检查通过之前不能声称后端迁移完成。
