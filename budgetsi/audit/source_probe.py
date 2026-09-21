"""Load exact numerical functions for CPU probes without Ray/CUDA imports.

Decorators/type annotations are removed; function bodies remain unchanged.
This is a numerical unit harness, NOT an installed verl/FSDP runtime test.
"""

import ast
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
PIN = "ac26e38d6f1572eb027597b48a9f4e01f6915ef8"


def read_source(path, upstream=False):
    if upstream:
        return subprocess.check_output(["git", "show", f"{PIN}:{path}"], cwd=ROOT, text=True)
    return (ROOT / path).read_text()


def function(path, name, namespace, upstream=False, parent=None):
    tree = ast.parse(read_source(path, upstream))
    body = tree.body
    if parent:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == parent).body
    node = next(n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    node.decorator_list = []
    node.returns = None
    for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
        arg.annotation = None
    if node.args.vararg:
        node.args.vararg.annotation = None
    if node.args.kwarg:
        node.args.kwarg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(ROOT / path), "exec"), namespace)
    return namespace[name]


def numerical_functions(upstream=True):
    ns = {"torch": torch, "AlgoConfig": type("AlgoConfig", (), {})}
    path = "verl/verl/utils/torch_functional.py"
    function(path, "masked_sum", ns, upstream)
    function(path, "masked_mean", ns, upstream)
    ns["verl_F"] = SimpleNamespace(masked_mean=ns["masked_mean"])
    path = "verl/verl/trainer/ppo/core_algos.py"
    for name in [
        "agg_loss",
        "compute_token_reward_direct_advantage",
        "compute_policy_loss_vanilla",
    ]:
        function(path, name, ns, upstream)
    return ns
