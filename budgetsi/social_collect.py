"""Run the unchanged social protocol with a loopback HF transport.

This subprocess uses the existing Sotopia environment, while the parent owns
the two resident GPU models. No candidate/prompt/selection logic is replaced.
"""

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "social_protocol"))
import runner


class LocalClient(runner.Client):
    async def request(self, payload, kind):
        return await super().request(payload, kind)

    async def score(self, prompt_ids, target_ids, kind):
        if not target_ids or len(prompt_ids) + len(target_ids) + 1 > self.max_context:
            raise runner.InvalidAction("score_context_exhausted", {"prompt_token_ids":prompt_ids,"target_ids":target_ids})
        result = await self.request(
            {"operation": "score", "prompt_ids": prompt_ids, "target_ids": target_ids},
            kind,
        )
        values = result["raw_logprobs"]
        if len(values) != len(target_ids) or any(
            not math.isfinite(x) or x > 0 for x in values
        ):
            raise RuntimeError("Invalid HF scoring receipt")
        return {
            "prompt_token_ids": prompt_ids,
            "target_ids": target_ids,
            "raw_logprobs": values,
            "probability_mode": "raw_prompt_logprobs",
            "model": self.model,
            "discarded_score_completion_ids": [],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--student-tokenizer", required=True)
    parser.add_argument("--teacher-tokenizer", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--max-context", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--goal-instruction", required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--teacher-context-mode",choices=runner.TEACHER_CONTEXT_MODES,required=True)
    parser.add_argument("--temperature",type=float,default=.7)
    parser.add_argument("--formal-training", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    args.student_endpoint = args.teacher_endpoint = args.endpoint
    if not 1 <= args.concurrency <= 16:
        raise ValueError("Concurrency must be between 1 and 16")
    runner.Client = LocalClient
    asyncio.run(runner.main(args))
    # Preserve original collector evidence and identify the caller scope.
    path = Path(args.output) / "summary.json"
    summary = json.loads(path.read_text())
    summary["engineering_only"] = not args.formal_training
    summary["transport"] = "resident_hf_loopback"
    summary["dialogue_concurrency"] = args.concurrency
    path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
