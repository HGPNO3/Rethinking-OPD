# 2026-09-21 同步检查

检查对象：学校训练源提交 `6aa4b0568ce1dd5cd2e2eb232e440d2acd5220e9` 的97个复制文件，逐项 SHA256 见 `SOURCE_SNAPSHOT.json`。新增说明与入口不在原源码哈希清单中。

| 检查 | 结果 |
|---|---|
| `budgetsi/tests` | 110 次测试执行通过 |
| `budgetsi/social_protocol` | 17 次测试执行通过 |
| `budgetsi/test_*.py` | 43 次测试执行通过（包含继承/复用的测试，不视为43个独立科学验证） |
| 复制文件 SHA256 | 97/97 与源版本相同 |
| Python / JSON | 全部 Python 文件可解析、全部 JSON 可解析 |
| 两组正式配置的 protocol 源哈希 | 全部与复制源码匹配 |
| 上游 `verl/` 与 `on_policy_distillation.sh` | 与固定上游提交无差异 |
| diff 格式 | 原源码 `execution_migration.py:103` 有一处行尾空格，保留字节一致性；其余无 whitespace 错误 |
| 发布范围 | 无模型权重、生成对话、评测产物或运行日志；高置信度凭据模式扫描无匹配 |

前三组在 macOS、Python3.13 的 CPU 环境执行。测试包含小型 GPT2/Qwen3.5 随机模型、真实导入的固定 actor、loopback HTTP、离线 W&B 临时事件，以及用于错误传播验收的主动异常。没有加载4B/27B权重，没有模型 API 请求，没有 W&B 在线上传。

起初使用最小依赖环境时，运行时测试因缺少 peft、verl 导入路径和 W&B 依赖未通过；补全环境后上述三组全部通过，没有修改训练源码来跳过失败。

核心与协议命令见 README。更广的运行时检查命令为：

```sh
PYTHONPATH=verl uv run --no-project \
  --with torch==2.10.0 --with hydra-core==1.3.2 --with numpy \
  --with peft==0.21.0 --with transformers==5.3.0 \
  --with wandb --with ray --with tensordict --with einops \
  --with packaging --with codetiming --with pandas --with datasets \
  python -m unittest discover -s budgetsi -p 'test_*.py'
```

本机使用已有缓存（执行时附加 `--offline`）；这条命令是 CPU 检查依赖，不是学校 GPU 部署的完整锁定环境。本次同步未重新运行分布式 GPU、40k 上下文峰值显存、完整在线训练或社会效果评测。原执行源码一致不能替代新主机验收。
