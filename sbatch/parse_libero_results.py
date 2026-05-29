#!/usr/bin/env python3
"""Parse OpenVLA LIBERO eval results from local logs and W&B.

For each suite, pairs the latest pretrained run (run_id_note `eval_<suite>_seed*`,
public LIBERO-finetuned checkpoints) with the latest fine-tuned run
(run_id_note `mine_<suite>_seed*`, our own fine-tuned checkpoints).

Sources:
  - Local logs in `experiments/logs/EVAL-*.txt` (full per-task / per-episode trace).
  - W&B runs in `<entity>/<project>` (uses `success_rate/total` from summary, or
    aggregates per-task metrics if the run is still in progress).

For a given (suite, tag), the source with more completed episodes wins.
Disable W&B with --no_wandb to use local logs only.
"""
import argparse
import re
import sys
from pathlib import Path

# Reference numbers from the OpenVLA paper, kept for sanity checking.
PUBLISHED = {
    "libero_spatial": (84.7, 0.9),
    "libero_object":  (88.4, 0.8),
    "libero_goal":    (79.2, 1.0),
    "libero_long":    (53.7, 1.3),
}

# EVAL-<suite>-openvla-<YYYY_MM_DD-HH_MM_SS>--<tag>_<suite>_seed<N>
RUN_RE = re.compile(
    r"^EVAL-(?P<suite>libero_\w+)-openvla-"
    r"(?P<ts>\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2})"
    r"--(?P<tag>eval|mine)_(?P=suite)_seed\d+$"
)


def parse_local_log(path: Path):
    """Return (per_task_dict, overall_sr_or_None, n_episodes)."""
    text = path.read_text(errors="ignore")

    tasks = {}
    for m in re.finditer(
        r"Task:\s*(.+?)\n.*?Current task success rate:\s*([0-9.]+)",
        text, flags=re.DOTALL,
    ):
        tasks[m.group(1).strip()] = float(m.group(2))

    overall = None
    for m in re.finditer(r"Current total success rate:\s*([0-9.]+)", text):
        overall = float(m.group(1))

    eps = re.findall(r"# episodes completed so far:\s*(\d+)", text)
    n_eps = int(eps[-1]) if eps else 0

    if overall is None:
        succ = re.findall(r"# successes:\s*(\d+)", text)
        if succ and eps:
            overall = int(succ[-1]) / int(eps[-1])

    return tasks, overall, n_eps


def collect_local(logs_dir: Path):
    """{(suite, tag): record} from local txt files; latest per key by timestamp."""
    by_key = {}
    for log in logs_dir.glob("EVAL-libero_*-openvla-*.txt"):
        m = RUN_RE.match(log.stem)
        if not m:
            continue
        key = (m.group("suite"), m.group("tag"))
        ts = m.group("ts")
        if key in by_key and by_key[key]["ts"] >= ts:
            continue
        tasks, overall, n_eps = parse_local_log(log)
        by_key[key] = {
            "ts": ts, "source": "local", "run_id": log.stem, "path": str(log),
            "overall": overall, "tasks": tasks, "n_eps": n_eps,
        }
    return by_key


def collect_wandb(entity: str, project: str):
    """{(suite, tag): record} from W&B; empty if W&B is unavailable."""
    try:
        import wandb
    except ImportError:
        print("[wandb] package not installed — skipping", file=sys.stderr)
        return {}
    try:
        api = wandb.Api(timeout=30)
        runs = list(api.runs(path=f"{entity}/{project}", per_page=200))
    except Exception as e:
        print(f"[wandb] cannot fetch ({type(e).__name__}: {e}) — skipping",
              file=sys.stderr)
        return {}

    by_key = {}
    for run in runs:
        m = RUN_RE.match(run.name)
        if not m:
            continue
        key = (m.group("suite"), m.group("tag"))
        ts = m.group("ts")
        if key in by_key and by_key[key]["ts"] >= ts:
            continue

        summary = dict(run.summary)
        tasks = {
            k[len("success_rate/"):]: float(v)
            for k, v in summary.items()
            if k.startswith("success_rate/") and k != "success_rate/total"
            and isinstance(v, (int, float))
        }
        ep_counts = {
            k[len("num_episodes/"):]: int(v)
            for k, v in summary.items()
            if k.startswith("num_episodes/") and k != "num_episodes/total"
            and isinstance(v, (int, float))
        }

        if "success_rate/total" in summary:
            overall = float(summary["success_rate/total"])
            n_eps = int(summary.get("num_episodes/total", 0))
        elif tasks:
            # In-progress run: aggregate completed tasks.
            n_eps = sum(ep_counts.get(t, 50) for t in tasks)
            num = sum(tasks[t] * ep_counts.get(t, 50) for t in tasks)
            overall = num / n_eps if n_eps else None
        else:
            overall, n_eps = None, 0

        by_key[key] = {
            "ts": ts, "source": "wandb", "run_id": run.name,
            "url": run.url, "state": run.state,
            "overall": overall, "tasks": tasks, "n_eps": n_eps,
        }
    return by_key


def pick_primary(l, w):
    """For one (suite, tag), pick the source with more episodes."""
    if l and w:
        return w if w["n_eps"] > l["n_eps"] else l
    return l or w


def fmt_pct(x):
    return f"{x * 100:.1f}" if x is not None else "N/A"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs_dir", default="./experiments/logs")
    ap.add_argument("--wandb_entity",
                    default="ssilenzi-university-of-modena-and-reggio-emilia")
    ap.add_argument("--wandb_project", default="openvla-libero-eval")
    ap.add_argument("--no_wandb", action="store_true",
                    help="Skip W&B; use local logs only.")
    ap.add_argument("--per_task", action="store_true",
                    help="Also print a per-task comparison.")
    args = ap.parse_args()

    local = collect_local(Path(args.logs_dir))
    wb = {} if args.no_wandb else collect_wandb(args.wandb_entity,
                                                args.wandb_project)

    if not local and not wb:
        print("No eval runs found in logs or W&B.", file=sys.stderr)
        sys.exit(1)

    keys = set(local) | set(wb)
    suites = sorted({s for s, _ in keys})

    picked = {}
    for suite in suites:
        for tag in ("eval", "mine"):
            l = local.get((suite, tag))
            w = wb.get((suite, tag))
            picked[(suite, tag)] = {
                "primary": pick_primary(l, w), "local": l, "wandb": w,
            }

    # ---- Per-run summary ----
    for (suite, tag), info in sorted(picked.items()):
        p = info["primary"]
        if not p:
            print(f"[{suite}/{tag}] missing in both sources")
            continue
        present = "+".join(s for s, v in [("local", info["local"]),
                                            ("wandb", info["wandb"])] if v)
        extra = f" state={p['state']}" if p["source"] == "wandb" else ""
        print(f"[{suite}/{tag}] {p['run_id']}  ts={p['ts']}  "
              f"overall={fmt_pct(p['overall'])}%  ep={p['n_eps']}  "
              f"primary={p['source']} ({present}){extra}")

    # ---- Comparison table ----
    print()
    sep = "=" * 96
    print(sep)
    print(f"{'Suite':<16} {'Pretrained':>14} {'Finetuned':>14} "
          f"{'Δ (pp)':>10} {'Paper':>16} {'Src e/m':>10}")
    print(sep)
    for suite in suites:
        ev = picked.get((suite, "eval"), {}).get("primary")
        mn = picked.get((suite, "mine"), {}).get("primary")
        ev_ov = ev["overall"] if ev else None
        mn_ov = mn["overall"] if mn else None
        if ev_ov is not None and mn_ov is not None:
            delta = f"{(mn_ov - ev_ov) * 100:+.1f}"
        else:
            delta = "N/A"
        if suite in PUBLISHED:
            pm, ps = PUBLISHED[suite]
            paper = f"{pm:.1f} ± {ps:.1f}"
        else:
            paper = "-"
        srcs = f"{(ev or {}).get('source', '-')[:1]}/{(mn or {}).get('source', '-')[:1]}"
        print(f"{suite:<16} {fmt_pct(ev_ov):>14} {fmt_pct(mn_ov):>14} "
              f"{delta:>10} {paper:>16} {srcs:>10}")
    print(sep)

    # ---- Per-task comparison ----
    if args.per_task:
        for suite in suites:
            ev = picked.get((suite, "eval"), {}).get("primary")
            mn = picked.get((suite, "mine"), {}).get("primary")
            ev_d = ev["tasks"] if ev else {}
            mn_d = mn["tasks"] if mn else {}
            tasks = sorted(set(ev_d) | set(mn_d))
            if not tasks:
                continue
            print(f"\n--- {suite} per task ---")
            print(f"{'Task':<60} {'Pre':>6} {'Fine':>6} {'Δ':>7}")
            for t in tasks:
                ev_v = ev_d.get(t)
                mn_v = mn_d.get(t)
                if ev_v is not None and mn_v is not None:
                    d = f"{(mn_v - ev_v) * 100:+.1f}"
                else:
                    d = "-"
                tname = t if len(t) <= 60 else t[:57] + "..."
                print(f"{tname:<60} {fmt_pct(ev_v):>6} "
                      f"{fmt_pct(mn_v):>6} {d:>7}")


if __name__ == "__main__":
    main()
