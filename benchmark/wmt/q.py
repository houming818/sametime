#!/usr/bin/env python3
"""q.py — single-GPU serial experiment queue manager. No dependencies beyond Python + Docker."""
import json, os, subprocess, sys, time, signal, shlex
from pathlib import Path

QUEUE_FILE = Path(os.environ.get("Q_FILE", os.path.join(os.path.dirname(__file__), "queue.json")))
LOG_DIR = Path(os.environ.get("Q_LOG_DIR", os.path.join(os.path.dirname(__file__), "q_logs")))

# Default cgroup limits — prevents OOM on the host
DOCKER_DEFAULTS = os.environ.get("Q_DOCKER_DEFAULTS", "--memory=12g --memory-swap=12g")

STATUS_QUEUED = "QUEUED"
STATUS_RUNNING = "RUNNING"
STATUS_DONE = "DONE"
STATUS_FAILED = "FAILED"


def load():
    if QUEUE_FILE.exists():
        return json.loads(QUEUE_FILE.read_text())
    return []


def save(jobs):
    QUEUE_FILE.write_text(json.dumps(jobs, indent=2))


def clean_cmd(cmd_str):
    """Replace newlines with spaces, collapse whitespace."""
    return " ".join(cmd_str.split())


def container_running(container_name):
    r = subprocess.run(["docker", "inspect", "-f", "{{.State.Status}}", container_name],
                       capture_output=True, text=True)
    return r.stdout.strip() == "running"


def container_exit_code(container_name):
    r = subprocess.run(["docker", "inspect", "-f", "{{.State.ExitCode}}", container_name],
                       capture_output=True, text=True)
    return int(r.stdout.strip()) if r.stdout.strip().isdigit() else -1


def start_one(job):
    name = job["name"]
    cmd = job["cmd"]
    log_path = LOG_DIR / f"{name}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"[q] starting {name}")
    container_cmd = (
        f"{cmd} 2>&1 | tee -a {log_path}"
    )
    # Use bash -c so pipe works, and nohup for detach
    full_cmd = f"docker run --rm --name q-{name} {cmd} > {log_path} 2>&1"
    
    # Split into docker run + redirect
    parts = shlex.split(cmd)
    proc = subprocess.Popen(
        parts + [">", str(log_path), "2>&1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setpgrp
    )
    return proc


def cmd_add(args):
    name = args[0]
    cmd = " ".join(args[1:])
    jobs = load()
    for j in jobs:
        if j["name"] == name:
            print(f"[q] {name} already exists (status={j['status']}). Use 'retry' or 'remove' first.")
            return
    jobs.append({"name": name, "cmd": clean_cmd(cmd), "status": STATUS_QUEUED})
    save(jobs)
    print(f"[q] added {name}")


DEFAULT_TIMEOUT = int(os.environ.get("Q_TIMEOUT", "7200"))


def cmd_start(args):
    timeout = DEFAULT_TIMEOUT
    if args and args[0].isdigit():
        timeout = int(args[0])

    jobs = load()
    running = [j for j in jobs if j["status"] == STATUS_RUNNING]
    if running:
        print(f"[q] {running[0]['name']} already running. Use 'kill' first.")
        return

    queued = [j for j in jobs if j["status"] == STATUS_QUEUED]
    if not queued:
        print("[q] no queued jobs")
        return

    job = queued[0]
    name = job["name"]
    container_name = f"q-{name}"

    # Clean up stale containers from previous failed runs
    subprocess.run(["docker", "kill", container_name], capture_output=True)
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

    job["status"] = STATUS_RUNNING
    job["started"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save(jobs)

    log_path = LOG_DIR / f"{name}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[q] launching {name} → log: {log_path}  (timeout={timeout}s)")
    full_cmd = f"docker run --name {container_name} {DOCKER_DEFAULTS} {job['cmd']} > {log_path} 2>&1"
    proc = subprocess.Popen(
        ["bash", "-c", full_cmd],
        env={**os.environ, "DOCKER_HOST": "unix:///var/run/docker.sock"},
        preexec_fn=os.setpgrp
    )

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(5)
        if not container_running(container_name):
            break
    else:
        print(f"[q] {name} TIMEOUT after {timeout}s", flush=True)
        subprocess.run(["docker", "kill", container_name], capture_output=True)

    # Read exit code (docker wait returns immediately since container has stopped)
    exit_code = -1
    dkr = subprocess.run(["docker", "wait", container_name], capture_output=True, text=True)
    if dkr.stdout.strip().isdigit():
        exit_code = int(dkr.stdout.strip())
    if exit_code == -1:
        exit_code = container_exit_code(container_name)

    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

    jobs = load()
    jobs[0]["status"] = STATUS_DONE if exit_code == 0 else STATUS_FAILED
    jobs[0]["exit_code"] = exit_code
    jobs[0]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save(jobs)

    if exit_code == 0:
        print(f"[q] {name} DONE")
    else:
        print(f"[q] {name} FAILED (exit={exit_code})")

    # Auto-start next job (preserve timeout)
    cmd_start([str(timeout)])


def cmd_status(args):
    jobs = load()
    if not jobs:
        print("[q] queue empty")
        return
    print(f"{'NAME':20s} {'STATUS':10s} {'TIME':20s}")
    print("-" * 52)
    for j in jobs:
        print(f"{j['name']:20s} {j['status']:10s} {j.get('started','') or j.get('finished',''):20s}")


def cmd_log(args):
    name = args[0] if args else None
    if name:
        log_path = LOG_DIR / f"{name}.log"
        if log_path.exists():
            os.execvp("tail", ["tail", "-f", str(log_path)])
        else:
            print(f"[q] no log for {name}")
    else:
        # Show running job's log
        jobs = load()
        running = [j for j in jobs if j["status"] == STATUS_RUNNING]
        if running:
            log_path = LOG_DIR / f"{running[0]['name']}.log"
            if log_path.exists():
                os.execvp("tail", ["tail", "-f", str(log_path)])
        print("[q] no running job")


def cmd_kill(args):
    jobs = load()
    running = [j for j in jobs if j["status"] == STATUS_RUNNING]
    if running:
        name = running[0]["name"]
        subprocess.run(["docker", "kill", f"q-{name}"], capture_output=True)
        jobs = load()
        jobs[0]["status"] = STATUS_QUEUED
        jobs[0].pop("started", None)
        save(jobs)
        print(f"[q] killed {name}, back to QUEUED")
    else:
        print("[q] nothing running")


def cmd_retry(args):
    name = args[0]
    jobs = load()
    for j in jobs:
        if j["name"] == name and j["status"] in (STATUS_FAILED, STATUS_DONE):
            j["status"] = STATUS_QUEUED
            j.pop("exit_code", None)
            j.pop("finished", None)
            save(jobs)
            print(f"[q] {name} reset to QUEUED")
            return
    print(f"[q] {name} not found or not failed")


def cmd_remove(args):
    name = args[0]
    jobs = load()
    jobs = [j for j in jobs if j["name"] != name]
    save(jobs)
    print(f"[q] removed {name}")


def cmd_clear(args):
    jobs = load()
    running = [j for j in jobs if j["status"] == STATUS_RUNNING]
    if running:
        print(f"[q] kill {running[0]['name']} first")
        return
    save([])
    print("[q] queue cleared")


def main():
    if len(sys.argv) < 2:
        print("Usage: q.py <add|start|status|log|kill|retry|remove|clear> [args...]")
        print("  add <name> <docker run command...>")
        print("  start                     — start processing queue")
        print("  status                    — show all jobs")
        print("  log [name]                — tail logs (default: running job)")
        print("  kill                      — stop running job, reset to QUEUED")
        print("  retry <name>              — reset failed job to QUEUED")
        print("  remove <name>             — delete job from queue")
        print("  clear                     — empty whole queue")
        return
    
    cmd = sys.argv[1]
    args = sys.argv[2:]
    
    handlers = {
        "add": cmd_add, "start": cmd_start, "status": cmd_status,
        "log": cmd_log, "kill": cmd_kill, "retry": cmd_retry,
        "remove": cmd_remove, "clear": cmd_clear,
    }
    
    if cmd in handlers:
        handlers[cmd](args)
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
