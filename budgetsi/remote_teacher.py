"""Frozen teacher RPC over a localhost SSH tunnel; no public model endpoint.

Only compact FP32 scores/IDs cross the tunnel. The teacher uses exactly the
local reference-prefix forward and generation code, with per-request seeds.
"""

import argparse
import hashlib
import json
import traceback
import time
import uuid
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path

import torch
from budgetsi.gpu_lease import teacher_scoring_lease


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@torch.no_grad()
def compact_scores(model, prompt, target, ids, k):
    from budgetsi.top16 import response_logits

    q = response_logits(model, prompt, target).float().log_softmax(-1)
    if k:
        si = torch.tensor(ids, dtype=torch.long, device=q.device)
        if si.shape != (len(target), k):
            raise ValueError("Student support shape mismatch")
        tq, ti = q.topk(k, dim=-1)
        return dict(
            teacher=q.gather(-1, si).cpu().tolist(),
            teacher_ids=ti.cpu().tolist(),
            teacher_topk=tq.cpu().tolist(),
        )
    y = torch.tensor(target, device=q.device)
    return dict(teacher_sampled=q.gather(-1, y[:, None]).squeeze(-1).cpu().tolist())


class RemoteTeacher:
    def __init__(self, endpoint, binding):
        if not endpoint.startswith("http://127.0.0.1:"):
            raise ValueError("Teacher endpoint must be a localhost SSH tunnel")
        self.endpoint, self.binding = endpoint, binding
        self.timings = []
        self.info = self.call("status", {})

    def call(self, operation, payload):
        request = dict(operation=operation, payload=payload, binding=self.binding, request_id=uuid.uuid4().hex)
        started = time.monotonic()
        body = json.dumps(request).encode()
        req = urllib.request.Request(
            self.endpoint, data=body, headers={"Content-Type": "application/json"}
        )
        # No automatic retry of generation: preserve exactly one recorded result.
        with urllib.request.urlopen(req, timeout=21600) as response:
            result = json.load(response)
        if (
            result.get("request_sha256") != digest(request)
            or result.get("binding") != self.binding
        ):
            raise ValueError("Teacher response binding mismatch")
        self.timings.append({**result.get("timing", {}), "operation": operation, "rpc_seconds": time.monotonic()-started})
        return result["result"]

    def score_support(self, action, ids, k):
        result = self.call(
            "opd",
            dict(
                prompt=list(action.teacher_prompt_ids),
                target=list(action.target_ids),
                ids=ids.cpu().tolist() if k else [],
                k=k,
            ),
        )
        return {
            key: torch.tensor(
                value, dtype=torch.long if key == "teacher_ids" else torch.float32
            )
            for key, value in result.items()
        }

    def diagnose(self, action, summary):
        payload = dict(prompt=list(action.teacher_prompt_ids), target=list(action.target_ids),
                       summary={k: v.tolist() if isinstance(v, torch.Tensor) else v for k,v in summary.items()})
        return {k: torch.tensor(v) for k,v in self.call('diagnostics', payload).items()}

    def assert_frozen(self):
        status = self.call("status", {})
        if not status["frozen"] or status["assets"] != self.info["assets"]:
            raise ValueError("Teacher identity/frozen state changed")
        return status


class TeacherService:
    def __init__(self, model, tokenizer, config, binding, assets, out):
        from budgetsi.social_loop import Engine

        self.model, self.config, self.binding, self.assets = (
            model,
            config,
            binding,
            assets,
        )
        self.versions = {name: p._version for name, p in model.named_parameters()}
        if config.get("collection_parallel"):
            from budgetsi.parallel_collect import make_parallel_engine
            settings = config["collection_parallel"]
            self.engine = make_parallel_engine(None, model, {"teacher": tokenizer}, config["context"], out,
                                               batch_size=settings["batch_size"], wait_ms=settings["wait_ms"],
                                               temperature=config.get('temperature', .7))
        else:
            self.engine = Engine(None, model, {"teacher": tokenizer}, config["context"], out,
                                 temperature=config.get('temperature', .7))
        self.engine.accepting = True

    def frozen(self):
        return all(
            not p.requires_grad and p.grad is None and p._version == self.versions[n]
            for n, p in self.model.named_parameters()
        )

    def resolve_config(self, binding):
        if binding != self.binding:
            raise ValueError('Teacher contract mismatch')
        return self.config

    @teacher_scoring_lease
    def dispatch(self, request):
        config = self.resolve_config(request['binding'])
        if not self.frozen():
            raise ValueError("Teacher frozen state mismatch")
        operation, data = request["operation"], request["payload"]
        if operation == "status":
            result = dict(
                frozen=True,
                assets=self.assets,
                peak_allocated_gib=torch.cuda.max_memory_allocated(0) / 2**30
                if torch.cuda.is_available()
                else 0,
            )
        elif operation == "collect":
            if data["model"] != "teacher":
                raise ValueError("Teacher service cannot serve student or IG")
            result = self.engine.call({**data, "experiment_binding": digest(request["binding"])})
        elif operation == 'diagnostics':
            from budgetsi.diagnostics import compare
            from budgetsi.top16 import response_logits
            from contextlib import nullcontext
            settings = config.get('diagnostics')
            summary = data['summary']
            if not settings or not data['prompt'] or not data['target'] or len(data['prompt'])+len(data['target'])+1 > config['context']:
                raise ValueError('Diagnostic request outside bound configuration')
            summary = {k: torch.tensor(v, dtype=torch.long if k=='ids' else torch.float32)
                       if k!='vocab_size' else v for k,v in summary.items()}
            if summary['ids'].shape != (len(data['target']),settings['k']):
                raise ValueError('Diagnostic K/position mismatch')
            guard = self.engine.workers['teacher'].execution_lock if getattr(self.engine,'parallel',False) else nullcontext()
            with guard, torch.no_grad():
                result = {k:v.cpu().tolist() for k,v in compare(summary,response_logits(self.model,data['prompt'],data['target']),1.).items()}
        elif operation == "validate_raw":
            if not config.get('teacher_inference') or not data['prompt'] or not data['target']:
                raise ValueError('Raw validation requires phased teacher')
            if len(data['prompt'])+len(data['target'])+1 > config['context']:
                raise ValueError('Raw validation context overflow')
            with self.engine.workers['teacher'].execution_lock:
                result=compact_scores(self.model,data['prompt'],data['target'],[],0)
        elif operation == "opd":
            from budgetsi.variant_spec import from_config

            variant = from_config(config)
            if data["k"] != variant.k or not data["prompt"] or not data["target"]:
                raise ValueError("Teacher scoring contract mismatch")
            if len(data["prompt"]) + len(data["target"]) + 1 > config["context"]:
                raise ValueError("Teacher scoring context overflow")
            from contextlib import nullcontext
            guard = self.engine.workers["teacher"].execution_lock if getattr(self.engine, "parallel", False) else nullcontext()
            with guard:
                result = compact_scores(
                    self.model, data["prompt"], data["target"], data["ids"], data["k"]
                )
        else:
            raise ValueError("Unknown teacher operation")
        if not self.frozen():
            raise ValueError("Teacher modified during request")
        return dict(result=result, binding=request["binding"], request_sha256=digest(request))


class SharedTeacherService(TeacherService):
    """One frozen model/worker, explicit independently approved run bindings."""
    def __init__(self, model, tokenizer, configs, bindings, assets, out):
        if not configs or len(configs) != len(bindings) or len(configs) > 3:
            raise ValueError('One to three explicitly bound runs required')
        if len({digest(b) for b in bindings}) != len(bindings):
            raise ValueError('Duplicate run binding')
        common = ('teacher', 'context', 'temperature', 'teacher_temperature', 'collection_parallel')
        for cfg in configs:
            if any(cfg.get(k) != configs[0].get(k) for k in common):
                raise ValueError('Shared teacher inference settings differ')
            if not cfg.get('collection_parallel'):
                raise ValueError('Shared teacher requires bounded parallel worker')
        self.run_configs = {digest(b): (b, cfg) for b, cfg in zip(bindings, configs)}
        super().__init__(model, tokenizer, configs[0], bindings[0], assets, out)

    def resolve_config(self, binding):
        entry = self.run_configs.get(digest(binding))
        if entry is None or entry[0] != binding:
            raise ValueError('Unregistered experiment binding')
        return entry[1]


def serve(service, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            try:
                size = int(self.headers["Content-Length"])
                if self.path != "/" or not 0 < size <= 16 * 1024**2:
                    raise ValueError("Invalid request size/path")
                result = service.dispatch(json.loads(self.rfile.read(size)))
                status = 200
            except Exception as error:
                traceback.print_exc()
                result, status = {"error": repr(error)}, 500
            body = json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    from budgetsi.http_runtime import BurstHTTPServer

    server_class = BurstHTTPServer if getattr(service.engine, "parallel", False) else HTTPServer
    return server_class(("127.0.0.1", port), Handler)


def contract(config, commit):
    return dict(config_sha256=digest(config), git_commit=commit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, action="append")
    parser.add_argument("--approval", required=True, action="append")
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=18740)
    args = parser.parse_args()
    from budgetsi.formal_gate import check_launch
    from budgetsi.social_loop import file_hash, atomic
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if len(args.config) != len(args.approval) or not 1 <= len(args.config) <= 3:
        raise ValueError('Supply a paired config/approval for each of at most three runs')
    configs = [json.loads(Path(p).read_text()) for p in args.config]
    cfg = configs[0]
    repo = Path(__file__).resolve().parents[1]
    gates = [check_launch(c, a, repo) for c, a in zip(args.config, args.approval)]
    gate = gates[0]
    bindings = [contract(c, g['verified_bindings']['git_commit']) for c, g in zip(configs, gates)]
    teacher_assets = lambda c: {p: h for p, h in c['files_sha256'].items() if Path(p).parent == Path(c['teacher'])}
    if any(teacher_assets(c) != teacher_assets(cfg) for c in configs):
        raise ValueError('Shared teacher assets differ between runs')
    common = ('teacher', 'context', 'temperature', 'teacher_temperature', 'collection_parallel')
    if any(any(c.get(k) != cfg.get(k) for k in common) for c in configs):
        raise ValueError('Shared inference settings differ')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    atomic(out / "gate.json", gates)
    assets = {
        p: file_hash(p)
        for p in cfg["files_sha256"]
        if Path(p).parent == Path(cfg["teacher"])
    }
    if any(h != cfg["files_sha256"][p] for p, h in assets.items()):
        raise ValueError("Teacher assets changed")
    from budgetsi.model_runtime import load_model
    gpu_count = 2 if Path(cfg['teacher']).name == 'Qwen3.5-27B' else 1
    if torch.cuda.device_count() != gpu_count:
        raise ValueError(f'Remote teacher requires exactly {gpu_count} visible GPUs')
    model = load_model(cfg['teacher'], 'balanced' if gpu_count == 2 else {'': 0})
    model.requires_grad_(False).eval()
    tok = AutoTokenizer.from_pretrained(cfg["teacher"], local_files_only=True)
    binding = contract(cfg, gate["verified_bindings"]["git_commit"])
    service = SharedTeacherService(model, tok, configs, bindings, assets, out) if len(configs) > 1 else TeacherService(model, tok, cfg, binding, assets, out)
    server = serve(service, args.port)
    atomic(
        out / "ready.json",
        dict(binding=binding, bindings=bindings, assets=assets, port=server.server_port),
    )
    print(json.dumps({"status": "ready", "port": server.server_port}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
