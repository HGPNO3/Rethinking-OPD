# 环境与可执行检查

CPU审查只需要Python和aiohttp，不需要CUDA、模型或数据：

```bash
uv run --no-project --with aiohttp python budgetsi/check_migration.py
```

源码组装（不运行训练、不会覆盖目录）：

```bash
python3 budgetsi/prepare_runtime.py --family qwen3 --mode fresh --output /tmp/budgetsi-qwen3-review
python3 budgetsi/prepare_runtime.py --family qwen35 --mode continuation --output /tmp/budgetsi-qwen35-review
```

`fresh`对应1000节点初始训练控制器；`continuation`对应带不可变历史核验的3000节点控制器。组装后使用原来同目录import方式，避免改动prompt源码哈希。运行入口为`train.py --approval <external-packet> --output <new-run-dir>`与`serve_teacher.py --approval ...`，本次不执行。

原训练实现依赖三类独立环境：SOTOPIA 0.1.5 + aiohttp + transformers 的collector、PyTorch/transformers/PEFT的HF更新器、支持prompt logprob与LoRA serving的vLLM。两家族尤其Qwen3.5对模型架构支持有版本要求。当前来源没有完整可验证的pip lock，因此不编造版本锁；精确GPU环境重建仍需取原环境freeze并重新验收。上游verl安装方式见根README，不能把它的环境宣称为社交路径已验证环境。

组装出的config保留科学参数，但runtime留空，模型文件绑定与旧批准不继承。必须补齐：本机可执行文件/模型路径、模型与tokenizer哈希、与manifest匹配的私有inputs、clean Git快照和新外置批准。续训还要精确父运行收据与optimizer lineage。没有这些信息时控制器应停止，而不是使用猜测默认值。
