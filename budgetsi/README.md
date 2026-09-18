# BudgetSI × OPD：分开审核参数与方法

固定上游：`ac26e38d6f1572eb027597b48a9f4e01f6915ef8`。上游框架和默认启动脚本保持原样。

**只想看参数，请先看 [参数审查表](PARAMETER_REVIEW.md)，再看集中列值的 [parameters.review.json](parameters.review.json)。**

- 提交01：参数配置审查。只有参数表与说明，没有训练代码；未决值明确标注。
- 提交02：方法适配。数据格式、损失计算、信息边界及CPU测试；不是完整训练迁移。

参数表不是可运行配置，不能用于启动正式实验；没有擅自确定学习率、weight decay、LoRA或长度上限。
