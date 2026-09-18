# 适配依据与参数审查

## 固定上游行为

主脚本 → Hydra → 学生rollout → 重算学生概率 → 教师对同一目标token评分 → token_reward_direct advantage → actor policy loss → minibatch更新。主脚本默认top-k16；top-k0才是sampled-token路径。教师打分温度与学生rollout温度是不同配置。`fsdp_workers.py:978`将学生评分温度设为rollout温度；`:237`按rollout.n与设备数换算minibatch。

## 需要保留的研究要求

两教师候选、正行动IG/token选择、非thinking、目标时刻信息边界、teacher-only参考候选、原始学生token含EOS监督、每批一次更新与optimizer续接。IG是训练选择代理，不证明社会充分性。

这些要求来自已确认项目方法。原代码只是实现参照，不是参数正确性的证明。

## 差异及处理

| 差异 | 上游 | 本项目要求/历史实现 | 项目侧处理与证据 |
|---|---|---|---|
| 概率目标 | 学生按rollout温度评分 | 行为.7，训练raw1，importance correction | 独立传入behavior与同快照HF raw分数；不能只把上游temperature改.7。数值反例验证两者不等价 |
| 教师条件 | 默认教师评分路径 | teacher-only reference与专业提示，student仅原始可见prefix | 接口只接收对原始目标token重新计算的teacher概率；参考内容不进入学生输入 |
| loss | 默认vanilla PPO | sampled raw-policy surrogate | 使用已有loss注册接口；不改框架，不把候选当SFT标签 |
| token平均 | microbatch均值按样本数加权 | 本批所有有效token等权 | 既有seq-mean-token-sum + advantage乘B/T，抵消框架microbatch样本权重；只在已验证的单rank、每批一次更新合同下使用 |
| 学习率 | 主脚本1e-6 | 历史1e-5 | 未证实哪个更好；不自动改默认，不把LoRA当十倍LR的理由 |
| weight decay | Hydra继承.01 | 历史0 | 真实优化器差异，明确交师兄审核；不悄悄覆盖 |
| LoRA | 默认rank0 | 历史r32/alpha64，3.5仅文本骨干 | 资源实现选择；必须验证参数清单/更新/保存恢复，不能直接抄all-linear |
| batch与n | 64 prompts、n4 | 16对话→可变入选节点 | 用真实入选节点B和token总数T构造一次更新，不把16对话当16训练样本 |
| 长度 | 1024+7168 | 历史服务context40960 | 服务容量不等于合理任务上限；需长度、截断与效果统计，未指定新的cap |
| 社交环境与评价 | 数学回调/验证 | SOTOPIA可见历史与partner、独立最终judge | 完整collector和回调接入须另外验收；不沿用数学reward衡量社交效果 |

## 归一化为什么可以在项目侧完成

令B为本次更新节点数，T为原始行动有效token总数，原advantage为A。项目构造`A'=A*B/T`。已有`seq-mean-token-sum`对每个microbatch先按序列平均；上游再乘microbatch样本数/B，得到该microbatch的`sum(token_loss)/T`。因此无须改动`dp_actor.py`。

成立条件：每个行动至少一个有效token；单rank无SP；整个B节点一次更新；静态microbatch=1或动态分组；所有microbatch更新前参数保持同一快照。任意多卡或多次epoch不能由这个证明直接推出，接口会拒绝未验收组合。

## 检验范围

数值测试验证上游函数体、原始surrogate、信息边界、mask、微批次分割与一次更新。它们不能证明学习率最优、社交表现改善或GPU端到端已可用。真实collector→评分→verl更新→新rollout与checkpoint链仍是后续验收事项。

上游定位：`core_algos.py:855,939,1058`；`dp_actor.py:789,816,860`；`fsdp_workers.py:237,978,2613,2720`。论文用于理解sampled/top-k区分：[§2.2](https://arxiv.org/html/2604.13016v2#S2.SS2)，以固定代码确定实际执行细节。
