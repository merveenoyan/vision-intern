#!/usr/bin/env python3
"""Job duration + cost ledger for HF Jobs.

The HF Jobs API exposes no start/finish timestamps and the logs carry no inline
timestamps, so wall-clock duration is unrecoverable after the fact. This tool
*observes* job state transitions live and stamps local time, persisting them to
a JSON ledger so costs can be estimated per job (judging, training, etc.).

Usage:
    # run in the background; polls every INTERVAL secs and updates the ledger
    python jobs/job_ledger.py poll [--interval 45]

    # print a duration + cost table from whatever has been recorded so far
    python jobs/job_ledger.py report

Notes:
- Duration is measured RUNNING -> terminal (a proxy for billed GPU time; it
  excludes queue/image-build, which HF generally does not bill for compute).
- Jobs that were already terminal before the poller first saw them have an
  unknown duration (we never observed them RUNNING) and are marked as such.
- Rates below are approximate USD/hr; edit them to match current HF pricing.
"""
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

LEDGER = os.path.join(os.path.dirname(__file__), os.pardir, ".job_ledger.json")
LEDGER = os.path.abspath(LEDGER)

# Approximate HF Jobs compute rates, USD/hr. Edit to match current pricing.
FLAVOR_RATES = {
    "cpu-basic": 0.00,
    "cpu-upgrade": 0.03,
    "t4-small": 0.40,
    "t4-medium": 0.60,
    "l4x1": 0.80,
    "l4x4": 3.20,
    "a10g-small": 1.00,
    "a10g-large": 1.50,
    "l40sx1": 1.80,
    "l40sx4": 7.20,
    "a100-large": 4.00,
}

TERMINAL = {"COMPLETED", "ERROR", "CANCELED", "FAILED", "TIMEOUT"}


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load():
    if os.path.exists(LEDGER):
        with open(LEDGER) as f:
            return json.load(f)
    return {}


def save(d):
    tmp = LEDGER + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, LEDGER)


def ps_all():
    """Return list of (job_id, status) from `hf jobs ps -a`."""
    out = subprocess.run(
        ["hf", "jobs", "ps", "-a"], capture_output=True, text=True
    ).stdout
    rows = []
    for line in out.splitlines():
        m = re.match(r"^([0-9a-f]{24})\s+.*\s+(\S+)\s*$", line)
        if m:
            rows.append((m.group(1), m.group(2)))
    return rows


def inspect(job_id):
    """One-time fetch of flavor + command label for a new job."""
    out = subprocess.run(
        ["hf", "jobs", "inspect", job_id], capture_output=True, text=True
    ).stdout
    try:
        d = json.loads(out)
        if isinstance(d, list):
            d = d[0]
    except Exception:
        return {"flavor": "?", "label": "?", "created_at": None}
    cmd = " ".join(d.get("command", []))
    # pull the script name (e.g. judge_one.py, train_rfdetr_job.py)
    sm = re.search(r"uv run '([^']+\.py)'", cmd) or re.search(r"(\S+\.py)", cmd)
    label = sm.group(1) if sm else cmd[:40]
    return {
        "flavor": d.get("flavor", "?"),
        "label": label,
        "created_at": str(d.get("created_at")),
    }


def poll_once(d):
    seen = ps_all()
    ts = now_iso()
    for job_id, status in seen:
        rec = d.get(job_id)
        if rec is None:
            meta = inspect(job_id)
            rec = {
                **meta,
                "first_seen": ts,
                "first_seen_status": status,
                "running_at": ts if status == "RUNNING" else None,
                "done_at": ts if status in TERMINAL else None,
                "status": status,
            }
            d[job_id] = rec
        else:
            rec["status"] = status
            if status == "RUNNING" and not rec.get("running_at"):
                rec["running_at"] = ts
            if status in TERMINAL and not rec.get("done_at"):
                rec["done_at"] = ts
    save(d)
    return d


def parse_iso(s):
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def duration_min(rec):
    start = parse_iso(rec.get("running_at"))
    end = parse_iso(rec.get("done_at"))
    if start and end:
        return (end - start).total_seconds() / 60.0, "measured"
    if start and rec.get("status") not in TERMINAL:
        return (datetime.now(timezone.utc) - start).total_seconds() / 60.0, "live"
    return None, "unknown"


def report():
    d = load()
    if not d:
        print("Ledger empty. Start the poller: python jobs/job_ledger.py poll &")
        return
    rows = sorted(d.items(), key=lambda kv: kv[1].get("first_seen", ""))
    print(f"{'JOB':24}  {'LABEL':22}  {'FLAVOR':9}  {'STATUS':10}  "
          f"{'DUR(min)':>9}  {'$/hr':>5}  {'COST$':>7}  KIND")
    total = 0.0
    for job_id, rec in rows:
        dur, kind = duration_min(rec)
        rate = FLAVOR_RATES.get(rec.get("flavor"), None)
        if dur is not None and rate is not None:
            cost = dur / 60.0 * rate
            total += cost
            cost_s, dur_s = f"{cost:7.2f}", f"{dur:9.1f}"
        else:
            cost_s, dur_s = "      ?", "        ?"
        rate_s = f"{rate:5.2f}" if rate is not None else "    ?"
        print(f"{job_id:24}  {rec.get('label','?'):22.22}  "
              f"{rec.get('flavor','?'):9}  {rec.get('status','?'):10}  "
              f"{dur_s}  {rate_s}  {cost_s}  {kind}")
    print("-" * 100)
    print(f"{'TOTAL (measured+live)':>88}  {total:7.2f}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "report":
        report()
    elif cmd == "poll":
        interval = 45
        if "--interval" in sys.argv:
            interval = int(sys.argv[sys.argv.index("--interval") + 1])
        d = load()
        while True:
            try:
                poll_once(d)
            except Exception as e:
                sys.stderr.write(f"[ledger] poll error: {e}\n")
            time.sleep(interval)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
