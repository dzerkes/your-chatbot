"""Run an eval dataset against the chatbot: uv run python -m evals.runner [--help].

Each case gets its own session for a per-dataset eval user and is sent through
Agent.respond, built exactly as the CLI builds it (Postgres history, real tools,
tracing when enabled). Needs the demo data: make seed ingest.
"""

import argparse
import asyncio
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from chatbot.agent import Agent
from chatbot.app import build_chatbot
from chatbot.config import Settings
from chatbot.llm.registry import ProviderRegistry
from chatbot.logging import configure_logging
from chatbot.types import User
from pydantic import ValidationError

from evals.dataset import DATASETS, Case, Dataset, load_dataset
from evals.judge import Judge, needs_judge
from evals.langfuse import send_scores
from evals.observe import observe
from evals.report import CaseResult, RunResult, run_name, summarize, text_report, write_reports
from evals.scorers import Scored, score_case

RESULTS = Path(__file__).resolve().parent / "results"
type Progress = Callable[[int, int, CaseResult], None]


async def evaluate(
    dataset: Dataset,
    agent: Agent,
    judge: Judge | None,
    *,
    model: str,
    judge_model: str | None = None,
    concurrency: int = 1,
    progress: Progress | None = None,
) -> RunResult:
    """Run every case, judge and score it. Results keep the dataset's order."""
    started = datetime.now(UTC)
    users: dict[str, User] = {}
    for key, spec in dataset.users.items():
        users[key] = await agent.store.upsert_user(
            User(
                external_id=f"eval-{dataset.name}-{key}",
                display_name=spec.display_name or f"Eval {key}",
                groups=sorted(set(spec.groups)),
            )
        )
    limit = asyncio.Semaphore(concurrency)
    done = 0

    async def run(case: Case) -> CaseResult:
        nonlocal done
        async with limit:
            observation = await observe(agent, users[case.user], case.question)
            judgement = None
            if needs_judge(case) and judge is not None:
                judgement = await judge.grade(case, observation.answer)
        scores = score_case(Scored(case, observation, judgement, dataset.budgets_for(case)))
        result = CaseResult(
            case_id=case.id,
            user=case.user,
            question=case.question,
            observation=observation,
            judgement=judgement,
            scores=scores,
        )
        done += 1
        if progress is not None:
            progress(done, len(dataset.cases), result)
        return result

    results = await asyncio.gather(*(run(case) for case in dataset.cases))
    return RunResult(
        run_name=run_name(dataset.name, started),
        dataset=dataset.name,
        model=model,
        judge_model=judge_model,
        started_at=started,
        finished_at=datetime.now(UTC),
        summary=summarize(results),
        results=results,
    )


def _print_progress(done: int, total: int, result: CaseResult) -> None:
    failed = [s.name for s in result.scores if not s.passed]
    status = result.observation.status if result.observation.status != "completed" else "ok"
    detail = f"failed {', '.join(failed)}" if failed else "passed"
    print(
        f"[{done}/{total}] {result.case_id}: {status}, {detail}, "
        f"{result.observation.latency_seconds:.1f}s",
        file=sys.stderr,
        flush=True,
    )


def select_cases(dataset: Dataset, patterns: list[str]) -> Dataset:
    if not patterns:
        return dataset
    cases = [c for c in dataset.cases if any(p in c.id for p in patterns)]
    if not cases:
        raise SystemExit(f"No case id contains {', '.join(patterns)}")
    return dataset.model_copy(update={"cases": cases})


async def main_async(args: argparse.Namespace, settings: Settings) -> RunResult:
    dataset = select_cases(load_dataset(args.dataset), args.case)
    judge_model = args.judge_model or settings.llm_primary
    registry = ProviderRegistry(settings)
    judge_provider = None if args.no_judge else registry.gateway(judge_model)
    try:
        async with build_chatbot(settings) as agent:
            judge = None
            if judge_provider is not None:
                judge = Judge(judge_provider, timeout_seconds=settings.llm_timeout_seconds)
            run = await evaluate(
                dataset,
                agent,
                judge,
                model=settings.llm_primary,
                judge_model=None if args.no_judge else judge_model,
                concurrency=args.concurrency,
                progress=_print_progress,
            )
        # Leaving build_chatbot flushed the turn traces, so scores attach to existing traces.
    finally:
        if judge_provider is not None:
            await judge_provider.aclose()
    return run


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an eval dataset against the chatbot")
    parser.add_argument("--dataset", type=Path, default=DATASETS / "demo.yaml")
    parser.add_argument(
        "--case", action="append", default=[], help="Run cases whose id contains this"
    )
    parser.add_argument(
        "--judge-model", help="provider:model for graded scorers (default LLM_PRIMARY)"
    )
    parser.add_argument("--no-judge", action="store_true", help="Skip model-graded checks")
    parser.add_argument("--concurrency", type=int, default=3, choices=range(1, 17), metavar="1-16")
    parser.add_argument(
        "--output", type=Path, help="Results directory (default evals/results/<run>)"
    )
    parser.add_argument("--no-langfuse", action="store_true", help="Do not send scores to Langfuse")
    parser.add_argument("--verbose", action="store_true", help="Show application logs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = Settings()
        # Memory would carry facts from one case into the next for the same eval user.
        settings = settings.model_copy(
            update={
                "memory_enabled": False,
                "log_level": settings.log_level if args.verbose else "WARNING",
            }
        )
        configure_logging(settings)
        run = asyncio.run(main_async(args, settings))
    except ValidationError as exc:
        print(f"Invalid dataset or settings:\n{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nEval run interrupted.", file=sys.stderr)
        return 130
    json_path, md_path = write_reports(run, args.output or RESULTS / run.run_name)
    print(text_report(run))
    print(f"\nWrote {json_path} and {md_path}")
    if settings.tracing_enabled and not args.no_langfuse:
        try:
            accepted = asyncio.run(send_scores(run, settings))
            print(f"Sent {accepted} scores to Langfuse (filter by eval_run={run.run_name})")
        except Exception as exc:
            print(f"Could not send scores to Langfuse: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
