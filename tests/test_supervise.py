"""Offline tests of scripts/supervise.sh (--allowed-host, PREPARE=0, FOREGROUND=1) and the start/stop dispatch.

No Docker, GPU, ssh or network: `docker`, `ssh` and the stop script are fakes on PATH that keep a container's state in
a file. Run: python3 -m pytest tests/ (or python3 tests/test_supervise.py).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

FAKE_DOCKER = r"""#!/usr/bin/env bash
# the container's state: "running", "exited:<code>", or no file (removed)
s="$FAKE/state"
echo "docker $*" >> "$FAKE/calls"
[[ ! -e "$FAKE/daemon_down" ]] || exit 1   # every call fails while the daemon is away
state() { cat "$s" 2>/dev/null || echo gone; }
case "$1" in
  info) exit 0 ;;
  logs) while [[ "$(state)" == running ]]; do sleep 0.1; done; echo "rank 0 log line"; exit 0 ;;
  wait) while [[ "$(state)" == running ]]; do sleep 0.1; done
        st=$(state); [[ "$st" == gone ]] && exit 1; echo "${st#exited:}"; exit 0 ;;
  container) [[ "$(state)" != gone ]]; exit ;;
  inspect)
    st=$(state); [[ "$st" == gone ]] && exit 1
    case "$3" in
      *State.Running*) [[ "$st" == running ]] && echo true || echo false ;;
      *State.ExitCode*) echo "${st#exited:}" ;;
      *) echo "" ;;
    esac ;;
  *) exit 1 ;;
esac
"""

FAKE_STOP = r"""#!/usr/bin/env bash
# the recipe's stop script: records how it was called and removes the container
echo "stop lock_held=${GLM_STOP_LOCK_HELD:-0} memguard=${MEMGUARD:-?}" >> "$FAKE/calls"
rm -f "$FAKE/state"
"""

# Sources the recipe's config, helpers and scripts/supervise.sh as a start script does, with a fake stop script
HARNESS_HEAD = r"""
set -euo pipefail
cd "$REPO"
source ./scripts/prepare.sh
source ./scripts/supervise.sh
CONTAINER_NAME=glm53-spark
STATE_DIR="$FAKE/state-dir"; mkdir -p "$STATE_DIR"
MEMGUARD_INTERVAL_S=1
fg_init "$FAKE/stop.sh"
take_lock() { exec 8>"$STATE_DIR/start.lock"; flock -n 8; fg_locked; }
"""


def setup(tmp_path: Path, state: str | None = "running") -> dict:
    fake = tmp_path / "fake"
    (fake / "bin").mkdir(parents=True, exist_ok=True)
    (fake / "home").mkdir(exist_ok=True)
    for name, text in (("bin/docker", FAKE_DOCKER), ("stop.sh", FAKE_STOP),
                       ("bin/ssh", "#!/usr/bin/env bash\necho \"ssh $*\" >> \"$FAKE/calls\"; exit 255\n")):
        p = fake / name
        p.write_text(text)
        p.chmod(0o755)
    if state:
        (fake / "state").write_text(state)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TF_", "TENSORFOLD_", "NCCL_"))}
    for k in ("FOREGROUND", "PREPARE", "DRY_RUN", "FABRIC_EXPECT", "WORKER", "WORKER2", "WORKER3", "MEMGUARD_PID"):
        env.pop(k, None)
    env.update(FAKE=str(fake), REPO=str(REPO), HOME=str(fake / "home"), PATH=f"{fake / 'bin'}:{os.environ['PATH']}")
    return env


def harness(tmp_path: Path, body: str, env: dict, **kw) -> subprocess.Popen:
    script = tmp_path / "harness.sh"
    script.write_text(HARNESS_HEAD + textwrap.dedent(body))
    return subprocess.Popen(["bash", str(script)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True, **kw)


def finish(p: subprocess.Popen, timeout: float = 30) -> tuple[int, str]:
    out, _ = p.communicate(timeout=timeout)
    return p.returncode, out


def calls(env: dict) -> list[str]:
    f = Path(env["FAKE"]) / "calls"
    return f.read_text().splitlines() if f.exists() else []


def stops(env: dict) -> list[str]:
    return [c for c in calls(env) if c.startswith("stop ")]


# --- arguments -----------------------------------------------------------------------------------------------------

def run_args(tmp_path, *args):
    env = setup(tmp_path)
    script = tmp_path / "args.sh"
    script.write_text(HARNESS_HEAD + 'start_args ./start.sh "$@"; echo "MODE=$MODE HOSTS=${ALLOWED_HOSTS[*]}"\n')
    return subprocess.run(["bash", str(script), *args], env=env, capture_output=True, text=True)


def test_args_restart_and_allowed_hosts(tmp_path):
    r = run_args(tmp_path, "restart", "--allowed-host", "spark1", "--allowed-host=192.168.1.5", "--allowed-host", "[::1]")
    assert r.returncode == 0, r.stderr
    assert "MODE=restart HOSTS=spark1 192.168.1.5 [::1]" in r.stdout
    r = run_args(tmp_path)
    assert "MODE=start HOSTS=" in r.stdout


def test_args_empty_first_argument_is_a_plain_start(tmp_path):
    r = run_args(tmp_path, "")
    assert r.returncode == 0 and "MODE=start HOSTS=" in r.stdout, r.stderr


def test_args_refused(tmp_path):
    for args in (["bogus"], ["--allowed-host"], ["--allowed-host", "a:80"], ["--allowed-host", "-x"],
                 ["start"], ["--allowed-host", "a b"], ["restart", "restart"]):
        r = run_args(tmp_path, *args)
        assert r.returncode == 1 and "E_CONFIG" in r.stderr, (args, r.stdout, r.stderr)
    r = run_args(tmp_path, *[x for i in range(65) for x in ("--allowed-host", f"h{i}")])
    assert r.returncode == 1 and "At most 64" in r.stderr


def test_prepare_and_foreground_values(tmp_path):
    env = setup(tmp_path)
    for k, v in (("PREPARE", "yes"), ("FOREGROUND", "2")):
        p = harness(tmp_path, "echo unreachable\n", {**env, k: v})
        rc, out = finish(p)
        assert rc == 1 and "E_CONFIG" in out and "unreachable" not in out, (k, out)


def test_prepare_0_turns_rsync_into_check(tmp_path):
    env = setup(tmp_path)
    p = harness(tmp_path, 'echo "WW=$WORKER_WEIGHTS"\n', {**env, "PREPARE": "0", "WORKER_WEIGHTS": "rsync"})
    assert "WW=check" in finish(p)[1]
    p = harness(tmp_path, 'echo "WW=$WORKER_WEIGHTS"\n', {**env, "WORKER_WEIGHTS": "rsync"})
    assert "WW=rsync" in finish(p)[1]


def test_prepare_0_refuses_missing_weights(tmp_path):
    env = setup(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    body = 'MODEL_DIR=$1; MODEL_SOURCE=download; DRAFTER=none; check_weights; echo checked\n'
    script = tmp_path / "h.sh"
    script.write_text(HARNESS_HEAD + body)
    r = subprocess.run(["bash", str(script), str(model)], env={**env, "PREPARE": "0"}, capture_output=True, text=True)
    assert r.returncode == 1 and "E_PREPARE" in r.stderr and "checked" not in r.stdout
    (model / "config.json").write_text("{}")
    (model / ".revision").write_text("76c0b5173166d2795dd48860f45d8224817f894c\n")
    r = subprocess.run(["bash", str(script), str(model)], env={**env, "PREPARE": "0"}, capture_output=True, text=True)
    assert r.returncode == 0 and "Weights are ready" in r.stdout, r.stderr


# --- FOREGROUND=1 --------------------------------------------------------------------------------------------------

def later(env: dict, seconds: float, state: str | None) -> None:
    """The container's state changes after `seconds` (None: removed)."""
    f = Path(env["FAKE"]) / "state"
    cmd = f"sleep {seconds}; " + (f"echo {state} > {f}" if state else f"rm -f {f}")
    subprocess.Popen(["bash", "-c", cmd])


def test_attach_ends_with_a_stop_that_was_asked_for(tmp_path):
    env = setup(tmp_path)
    fake = Path(env["FAKE"])
    # a supervisor's ExecStop: the stop script takes start.lock itself (flock -n), which the attached start let go
    ext = (f"sleep 1; exec 8>{fake}/state-dir/start.lock; if flock -n 8; then echo external-locked >> {fake}/calls; "
           f"sleep 1; rm -f {fake}/state; else echo external-E_LOCK >> {fake}/calls; fi")
    p = harness(tmp_path, "take_lock; fg_launch; fg_attach --since 0\n", {**env, "FOREGROUND": "1"})
    subprocess.Popen(["bash", "-c", ext])
    rc, out = finish(p)
    assert rc == 0, out
    assert "external-locked" in calls(env)
    assert stops(env) == ["stop lock_held=1 memguard=0"]
    assert "removed by a stop" in out and "rank 0 log line" in out


def test_attach_rank_0_crash_gives_its_code(tmp_path):
    env = setup(tmp_path)
    p = harness(tmp_path, "take_lock; fg_launch; fg_attach --since 0\n", {**env, "FOREGROUND": "1"})
    later(env, 1, "exited:7")
    rc, out = finish(p)
    assert rc == 7, out
    assert stops(env) == ["stop lock_held=1 memguard=0"]
    assert "ended by itself (exit code 7)" in out


def test_attach_rank_0_ending_unasked_with_0_is_a_failure(tmp_path):
    env = setup(tmp_path)
    p = harness(tmp_path, "take_lock; fg_launch; fg_attach --since 0\n", {**env, "FOREGROUND": "1"})
    later(env, 1, "exited:0")
    rc, _ = finish(p)
    assert rc == 1
    assert len(stops(env)) == 1


def test_attach_sigterm_stops_every_rank(tmp_path):
    env = setup(tmp_path)
    p = harness(tmp_path, "take_lock; fg_launch; fg_attach --since 0\n", {**env, "FOREGROUND": "1"})
    time.sleep(1.5)
    os.kill(p.pid, signal.SIGTERM)
    rc, out = finish(p)
    assert rc == 143, out
    assert stops(env) == ["stop lock_held=1 memguard=0"]
    assert not (Path(env["FAKE"]) / "state").exists()


def test_attach_memory_guard_stop_is_a_failure(tmp_path):
    env = setup(tmp_path)
    fake = Path(env["FAKE"])
    body = f"""
    take_lock; fg_launch
    ( sleep 1; exec 8>{fake}/state-dir/start.lock; flock 8; rm -f {fake}/state; exit 3 ) &
    MEMGUARD_PID=$!
    fg_attach --since 0
    """
    p = harness(tmp_path, body, {**env, "FOREGROUND": "1"})
    rc, out = finish(p)
    assert rc == 1, out
    assert "memory guard stopped the server" in out


def test_attach_survives_a_docker_wait_that_ends_early(tmp_path):
    env = setup(tmp_path)
    fake = Path(env["FAKE"])
    # the first docker wait returns at once (the daemon went away for a moment); rank 0 still runs
    (fake / "bin" / "docker").write_text(FAKE_DOCKER.replace(
        '  wait) while', '  wait) if [[ ! -e "$FAKE/waited" ]]; then touch "$FAKE/waited"; exit 1; fi\n        while', 1))
    p = harness(tmp_path, "take_lock; fg_launch; fg_attach --since 0\n", {**env, "FOREGROUND": "1"})
    time.sleep(2)
    assert p.poll() is None, "the start ended while rank 0 was still running"
    later(env, 0, "exited:9")
    rc, _ = finish(p)
    assert rc == 9


def test_attach_waits_out_a_docker_daemon_outage(tmp_path):
    env = setup(tmp_path)
    fake = Path(env["FAKE"])
    p = harness(tmp_path, "take_lock; fg_launch; fg_attach --since 0\n", {**env, "FOREGROUND": "1"})
    time.sleep(1)
    (fake / "daemon_down").write_text("")       # docker wait ends, and nothing about rank 0 can be read
    time.sleep(2.5)
    assert p.poll() is None, "the start ended while Docker did not answer"
    (fake / "state").write_text("exited:5")
    (fake / "daemon_down").unlink()
    rc, out = finish(p)
    assert rc == 5, out
    assert stops(env) == ["stop lock_held=1 memguard=0"]


def test_attach_does_not_wait_for_ever_on_start_lock(tmp_path):
    env = setup(tmp_path)
    fake = Path(env["FAKE"])
    # something else holds start.lock (a preparation) when rank 0 ends
    holder = subprocess.Popen(["bash", "-c", f"sleep 1; exec 8>{fake}/state-dir/start.lock; flock 8; sleep 30"])
    try:
        p = harness(tmp_path, "take_lock; fg_launch; fg_attach --since 0\n",
                    {**env, "FOREGROUND": "1", "FOREGROUND_LOCK_WAIT_S": "2"})
        later(env, 2, "exited:4")
        t = time.monotonic()
        rc, out = finish(p)
        assert rc == 4 and "W_LOCK" in out and time.monotonic() - t < 15, out
        assert len(stops(env)) == 1
    finally:
        holder.kill()


def test_attach_memory_guard_status_is_read_reliably(tmp_path):
    fake_root = tmp_path
    for n in range(8):           # the guard's status was lost now and then when read from a subshell
        tmp = fake_root / f"run{n}"
        tmp.mkdir()
        env = setup(tmp)
        fake = Path(env["FAKE"])
        body = f"""
        take_lock; fg_launch
        ( exec 8>{fake}/state-dir/start.lock; flock 8; rm -f {fake}/state; exit 3 ) &
        MEMGUARD_PID=$!
        fg_attach --since 0
        """
        p = harness(tmp, body, {**env, "FOREGROUND": "1"})
        rc, out = finish(p)
        assert rc == 1 and "memory guard stopped the server" in out, (n, out)


def test_failure_after_launch_stops_every_rank(tmp_path):
    env = setup(tmp_path)
    p = harness(tmp_path, "take_lock; fg_launch; fail E_SMOKE 'no answer'\n", {**env, "FOREGROUND": "1"})
    rc, out = finish(p)
    assert rc == 1 and "E_SMOKE" in out and "every rank is stopped" in out
    assert stops(env) == ["stop lock_held=1 memguard=0"]


def test_failure_before_launch_stops_nothing(tmp_path):
    env = setup(tmp_path)
    p = harness(tmp_path, "take_lock; fail E_PORT busy\n", {**env, "FOREGROUND": "1"})
    rc, _ = finish(p)
    assert rc == 1 and stops(env) == []


def test_without_foreground_nothing_changes(tmp_path):
    env = setup(tmp_path)
    p = harness(tmp_path, "take_lock; fg_launch; fg_attach --since 0; echo returned; fail E_SMOKE x\n", env)
    rc, out = finish(p)
    assert rc == 1 and "returned" in out and stops(env) == []


def test_fg_run_failure_stops_once(tmp_path):
    env = setup(tmp_path)
    body = "take_lock; fg_launch; smoke() { fail E_SMOKE 'empty reply'; }; fg_run smoke; echo reached-the-end\n"
    p = harness(tmp_path, body, {**env, "FOREGROUND": "1"})
    rc, out = finish(p)
    assert rc == 1 and "E_SMOKE" in out and "reached-the-end" not in out
    assert stops(env) == ["stop lock_held=1 memguard=0"]


def test_fg_run_is_interrupted_by_sigterm(tmp_path):
    env = setup(tmp_path)
    body = "take_lock; fg_launch; slow() { sleep 60; }; fg_run slow\n"
    p = harness(tmp_path, body, {**env, "FOREGROUND": "1"})
    time.sleep(1)
    t = time.monotonic()
    os.kill(p.pid, signal.SIGTERM)
    rc, _ = finish(p, timeout=20)
    assert rc == 143 and time.monotonic() - t < 10
    assert len(stops(env)) == 1


# --- the start and stop scripts --------------------------------------------------------------------------------------

def checkpoint(tmp_path: Path, env: dict) -> dict:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "glm5_next", "quantization_config": {"quant_method": "exl3", "bits": 4}}')
    for f in ("a.safetensors", "tokenizer.json", "chat_template.jinja"):
        (model / f).write_text("")
    dflash = Path(env["HOME"]) / ".cache/aevonix-glm53/dflash2"
    dflash.mkdir(parents=True)
    for f in ("config.json", "a.safetensors"):
        (dflash / f).write_text("")
    return {**env, "MODEL_SOURCE": "local", "MODEL_DIR": str(model), "WORKER": "u@10.0.0.2",
            "STATE_DIR": str(tmp_path / "state"), "LOG_DIR": str(tmp_path / "logs")}


def test_dry_run_plan(tmp_path):
    env = checkpoint(tmp_path, setup(tmp_path, None))
    r = subprocess.run(["./start.sh", "--allowed-host", "spark1", "--allowed-host", "192.168.1.5"], cwd=REPO,
                       env={**env, "DRY_RUN": "1", "PREPARE": "0", "FOREGROUND": "1"}, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    serve = [line for line in r.stdout.splitlines() if " serve /model " in line]
    assert len(serve) == 4, serve      # each rank's line, then each rank's docker run
    assert all(line.rstrip().endswith("--allowed-host spark1 --allowed-host 192.168.1.5") for line in serve)
    assert "(PREPARE=0)" in r.stdout and "nothing is built" in r.stdout and "Build $IMAGE" not in r.stdout
    assert "would stay attached to rank 0" in r.stdout


def test_dry_run_unchanged_without_the_new_settings(tmp_path):
    env = checkpoint(tmp_path, setup(tmp_path, None))
    r = subprocess.run(["./start.sh"], cwd=REPO, env={**env, "DRY_RUN": "1"}, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--allowed-host" not in r.stdout and "PREPARE=0" not in r.stdout and "FOREGROUND" not in r.stdout
    assert "Build tensorfold-glm53-spark:" in r.stdout


def test_start_refuses_unknown_arguments(tmp_path):
    env = checkpoint(tmp_path, setup(tmp_path, None))
    r = subprocess.run(["./start.sh", "now"], cwd=REPO, env={**env, "DRY_RUN": "1"}, capture_output=True, text=True)
    assert r.returncode == 1 and "Usage: ./start.sh [restart] [--allowed-host NAME]..." in r.stderr


def test_fabric_expect_runs_the_four_spark_scripts(tmp_path):
    env = checkpoint(tmp_path, setup(tmp_path, None))
    env = {**env, "DRY_RUN": "1", "WORKER2": "u@10.0.0.3", "WORKER3": "u@10.0.0.4", "WORKER_WEIGHTS": "nfs", "MASTER_ADDR": "10.200.0.1"}
    r = subprocess.run(["./start.sh"], cwd=REPO, env={**env, "FABRIC_EXPECT": "ring"}, capture_output=True, text=True)
    assert "Check all four Sparks" in r.stdout, r.stdout + r.stderr
    r = subprocess.run(["./stop.sh"], cwd=REPO, env={**env, "FABRIC_EXPECT": "ring"}, capture_output=True, text=True)
    assert r.returncode == 0 and "four ranks' Docker output" in r.stdout, r.stdout + r.stderr
    r = subprocess.run(["./stop.sh"], cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "Save both ranks' Docker output" in r.stdout, r.stdout + r.stderr
    for script in ("./start.sh", "./stop.sh"):
        r = subprocess.run([script], cwd=REPO, env={**env, "FABRIC_EXPECT": "triangle"}, capture_output=True, text=True)
        assert r.returncode == 1 and "FABRIC_EXPECT must be ring, mesh or unset" in r.stderr


if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-q", __file__]))
