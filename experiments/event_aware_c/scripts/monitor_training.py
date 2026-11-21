"""Read-only NNDL training monitor. No train process is launched or signalled."""
import argparse
from collections import deque
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
import time


def query_nvidia(query):
    try:
        result = subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=2, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))


def process_start(pid):
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return int(stat[19]) if stat[0] != "Z" else None
    except (OSError, ValueError, IndexError, TypeError):
        return None


def gpu_info(lines, status, alive, now):
    gpus = query_nvidia("--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu")
    if gpus is None:
        lines.append("GPU telemetry unavailable")
        return
    selected = status.get("gpu_uuid")
    selected_gpu = next((g for g in gpus if len(g) == 7 and g[1] == selected), None)
    for index, uuid, name, utilization, used, total, temperature in (g for g in gpus if len(g) == 7):
        marker = " (NNDL)" if uuid == selected else ""
        lines.append(f"GPU {index} {name}{marker}  |  utilization {utilization}%  |  VRAM {used}/{total} MiB"
                     f"  |  temperature {temperature} °C")
    apps = query_nvidia("--query-compute-apps=gpu_uuid,pid,used_gpu_memory")
    if apps is not None:
        relevant = [app for app in apps if len(app) == 3 and (selected_gpu is None or app[0] == selected)]
        mine = [app for app in relevant if alive and app[1] == str(status.get("pid"))]
        if mine:
            lines.append(f"NNDL process VRAM (NVML)  {mine[0][2]} MiB")
        lines.append("Other or unmapped GPU compute PIDs  " +
                     (", ".join(app[1] for app in relevant if app not in mine) or "none"))
    else:
        lines.append("GPU process list unavailable")
    reserved = status.get("cuda_reserved_mib")
    if alive and reserved is not None:
        lines.append(f"NNDL PyTorch reserved VRAM (last batch)  {reserved:.0f} MiB")
    last = status.get("last_batch_at")
    if alive and status.get("cuda_verified") and last:
        recent = now - last <= 15
        lines.append("NNDL CUDA  " + ("recent CUDA optimizer update observed" if recent else
                                     f"last confirmed update {age(last, now)} ago"))


def age(stamp, now):
    return f"{max(0, now - stamp):.0f}s" if stamp is not None else "—"


def date(stamp):
    return datetime.fromtimestamp(stamp).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z") if stamp else "—"


def metric(value):
    return f"{value:.4f}%" if value is not None else "—"


def tail_project_log(path, workspace):
    target = Path(path).resolve()
    if not target.is_relative_to(workspace / "logs"):
        return []
    try:
        with target.open("rb") as stream:
            stream.seek(max(0, target.stat().st_size - 16384))
            return [line.decode(errors="replace").rstrip() for line in deque(stream, maxlen=7)]
    except OSError:
        return []


def render(workspace, interval):
    now = time.time()
    lines = [f"NNDL Plan C  |  {date(now)}  |  refresh {interval:g}s"]
    try:
        state = json.loads((workspace / "logs" / "active.json").read_text())
    except (OSError, ValueError):
        state = {}
    if not state:
        lines.append("Waiting for NNDL training status")
        gpu_info(lines, state, False, now)
        return "\n".join(lines)

    pid = state.get("pid")
    start = process_start(pid)
    alive = start is not None and state.get("pid_start") == start
    phase = state["phase"]
    if phase in ("Preparing", "Training", "Validating") and not alive:
        phase = "Interrupted (training process exited before writing final state)"
    updates, total = state.get("updates", 0), state.get("total_updates")
    percent = 100 * updates / total if total else None
    started = state.get("started_at")
    elapsed = age(started, state.get("finished_at") or now) if started else "—"
    loss = state.get("loss")
    lines.extend([
        f"{state['arm']}  |  {phase}  |  seed {state.get('seed', '—')}  |  mode {state.get('mode', '—')}",
        f"PID {pid or '—'} ({'alive' if alive else 'not running'})  |  elapsed {elapsed}",
        f"Epoch {state.get('epoch', '—')}/{state.get('total_epochs', '—')}  |  "
        f"batch {state.get('batch', '—')}/{state.get('total_batches', '—')}  |  "
        f"updates {updates}/{total or '—'}  |  optimizer progress {f'{percent:.2f}%' if percent is not None else '—'}",
        "Last batch loss  " + (f"{loss:.5f}" if loss is not None else "—"),
        f"Latest DEV  Event-F1 {metric(state.get('last_event_f1'))}  |  "
        f"Real Duration FPR {metric(state.get('last_real_fpr'))}",
        f"Best DEV  epoch {state.get('best_epoch', '—')}  |  Event-F1 {metric(state.get('best_event_f1'))}  |  "
        f"Real Duration FPR {metric(state.get('best_real_fpr'))}",
        f"State updated  {date(state.get('updated_at'))} ({age(state.get('updated_at'), now)} ago)",
        f"Batch data age  {age(state.get('last_batch_at'), now)}  |  last batch {date(state.get('last_batch_at'))}",
    ])
    if state.get("probe_total"):
        lines.append(f"CUDA driver probe  {state['probe_steps']}/{state['probe_total']} operations verified")
    if state.get("detail"):
        lines.append(state["detail"])
    gpu_info(lines, state, alive, now)
    lines += [f"Log  {state.get('log_path', '—')}", f"Checkpoints  {state.get('checkpoint_dir', '—')}"]
    if state.get("error"):
        lines.append(f"Error  {state['error']}")
    if state.get("log_path"):
        lines.extend(tail_project_log(state["log_path"], workspace))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("interval", nargs="?", type=float, default=5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("interval must be a positive number of seconds")
    workspace = Path(__file__).resolve().parents[1]
    interactive = sys.stdout.isatty() and not args.once
    try:
        while True:
            started = time.monotonic()
            print(("\033[2J\033[H" if interactive else "") + render(workspace, args.interval), flush=True)
            if args.once:
                return
            time.sleep(max(0, args.interval - (time.monotonic() - started)))
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
