# BudgetSI 审核入口

这份 fork 用两个提交分别审查配置适配和当前方法代码迁移。
固定上游见 [UPSTREAM.json](UPSTREAM.json)，先读 [参数对照](review/PARAMETERS.md)。

当前配置依据 2026-09-18 完成的两组3000节点运行；主规则：非thinking、原样goal、正行动IG/token选师、教师候选可见原行动、teacher-only参考条件原始token OPD。IG只是局部代理，不证明社交充分性。

配置快照移除了主机地址、私有路径、旧批准与checkpoint绑定。没有发布密钥、训练场景、原始对话或模型权重。本fork不自动获准新训练。

完整代码审查请读 [方法映射](review/METHOD_MAP.md)、[环境与组装](ENVIRONMENT.md) 和 [源文件清单](SOURCE_MANIFEST.json)。`method/` 是共同实现，`variants/` 保留两组不同的启动/续训工程行为。

**状态：源码迁移与CPU验收；尚未改写为verl后端，也没有在此fork完成GPU复现。** 本次没有变更原BudgetSI仓库的训练实现。不要将上游单教师命令当作完整社交方法入口。
