# BudgetSI × OPD：分开审核参数与方法

固定上游：`ac26e38d6f1572eb027597b48a9f4e01f6915ef8`。上游框架和默认启动脚本保持原样。

**只想看参数，请先看 [参数审查表](PARAMETER_REVIEW.md)，再看集中列值的 [parameters.review.json](parameters.review.json)。**

- 提交01：参数配置审查。只有参数表与说明，没有训练代码；未决值明确标注。
- 提交02：方法适配。数据格式、损失计算、信息边界及CPU测试；不是完整训练迁移。

参数表不是可运行配置，不能用于启动正式实验；没有擅自确定学习率、weight decay、LoRA或长度上限。

## 02 方法适配（需要审代码时再读）

先读[方法适配依据](DECISIONS.md)，再看 `adapter.py` 和 `register_loss.py`。这里的batch、聚合与概率设置是保持计算语义的接口约定，不是第一份提交中待选择的实验超参数。

## 项目接入接口

- `adapter.build_update`：输入已通过collector验证的入选节点及训练actor在**同一冻结快照**上重算的HF raw分数；输出原始student prefix/target、behavior分母、token mask及按B/T缩放的advantage。历史serving raw分数不能冒充HF重算。
- `adapter.actor_overrides(B)`：只给出保持上述更新语义所必需的接口设置。B是本批真实节点数，不是对话数。学习率、LoRA、模型与数据路径均未被自动填写。
- `register_loss.py`：使用上游`register_policy_loss`扩展。actor进程通过已有`VERL_USE_EXTERNAL_MODULES=budgetsi.register_loss`加载；repo根目录需可import。不会修改`verl/`中的文件。
- `PreparedUpdate.as_dataproto(actor_config=实际初始化后配置, world_size=1)`：先验证实际actor合同再构造上游数据对象，自己不调用训练。已批准driver才能调用`worker.update_actor`。

调用前必须由driver校验实际模型快照、同批概率缓存、已通过选择的原始A行动、精确tokenizer、门禁与checkpoint链。本接口中的snapshot字符串/概率回执只检查一致性，不能单凭它们证明模型权重或教师信息来源真实。

验收报告见[VALIDATION.json](VALIDATION.json)，重跑：

```bash
uv run --no-project --with torch==2.10.0 --with hydra-core==1.3.2 --with numpy python budgetsi/verify.py
```

当前有用交付是：配置/语义审查材料 + 经过CPU验证的项目侧更新适配。**完整collector→教师打分→verl服务→新rollout，以及真实模型保存恢复/GPU验收仍未完成。** 不能称为完整训练迁移已完成，也不能直接把示例节点用作实验结果。
