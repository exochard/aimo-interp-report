"""Read-only launch check: Python compute jobs and substantial graphics allocations block."""
import subprocess
import os
import signal
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def blockers(allowed=()):
    output = subprocess.run(['nvidia-smi', '-q', '-x'], capture_output=True, text=True, check=True, timeout=10).stdout
    found = []
    for process in ET.fromstring(output).findall('.//process_info'):
        name = process.findtext('process_name', '')
        used = process.findtext('used_memory', '')
        kind = process.findtext('type', '')
        try:
            mib = int(used.split()[0])
        except (ValueError, IndexError):
            mib = 1024
        if int(process.findtext('pid', '0')) not in allowed and ('python' in name.lower() or mib >= 1024):
            found.append(dict(pid=process.findtext('pid'), name=name, mib=mib, type=kind))
    return found


def descendants(pid):
    found = {pid}
    path = Path(f'/proc/{pid}/task/{pid}/children')
    try:
        for child in path.read_text().split():
            found.update(descendants(int(child)))
    except FileNotFoundError:
        pass
    return found


def run(command):
    if blockers():
        raise RuntimeError('GPU became occupied before extraction')
    process = subprocess.Popen(command, start_new_session=True)
    try:
        while process.poll() is None:
            foreign = blockers(descendants(process.pid))
            if foreign:
                raise RuntimeError('foreground GPU allocation appeared: ' + str(foreign))
            time.sleep(3)
        return process.returncode
    finally:
        if process.poll() is None:
            # Signal only this invocation; exclusive_gpu_run handles its own model child.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == '__main__':
    import json
    import sys
    if sys.argv[1:3] == ['--run', '--']:
        raise SystemExit(run(sys.argv[3:]))
    rows = blockers()
    if rows:
        print(json.dumps(rows), flush=True)
    raise SystemExit(1 if rows else 0)
