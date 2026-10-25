"""Update only the round report's activation-result marker from saved terminal evidence."""
import datetime
import hashlib
import json
import sys
from pathlib import Path


def record(result_dir, report):
    terminal = json.loads((result_dir/'terminal.json').read_text())
    probe_path = result_dir/'probe.json'
    valid = (terminal['extraction_exit'] == 0 and terminal['fit_exit'] == 0 and probe_path.exists()
             and hashlib.sha256(probe_path.read_bytes()).hexdigest() == terminal['probe_sha256'])
    result = json.loads(probe_path.read_text()) if valid else None
    if result:
        p, b = result['probe'], result['majority']
        state = 'PASS' if result['passed'] else 'FAIL'
        text = (f"{state}: activation probe {p['correct']}/{p['n']}, majority {b['correct']}/{b['n']}; "
                f"balanced accuracy {p['balanced_accuracy']:.4f}. Predicate results: "
                + ', '.join(f"{k}={v}" for k,v in result['gate'].items()) + '. '
                "Measured with local BF16/NF4 weights; official precision remains untested. "
                "Evidence: `contests/aimo-interp/runs/activation-r3/probe.json`. ")
        text += ('Next: export the probe and verify official precision before any upload recommendation.'
                 if result['passed'] else 'Next: retain the existing entry; this activation method failed its frozen gate.')
    else:
        reason = ('terminal/probe hash mismatch' if terminal['extraction_exit'] == 0 and terminal['fit_exit'] == 0
                  else terminal['reason'])
        text = (f"INCOMPLETE: {reason}. No activation quality score is available. "
                "Evidence: `contests/aimo-interp/runs/activation-r3/terminal.json`. "
                "Next: review the resource/launch failure before a bounded retry of the same method.")
    text += f" Slot outcome recorded {terminal['finished']} (extraction exit {terminal['extraction_exit']})."
    content = report.read_text()
    start, end = '<!-- AIMO_RESULT_START -->', '<!-- AIMO_RESULT_END -->'
    if content.count(start) != 1 or content.count(end) != 1:
        raise ValueError('report marker drift; preserve report and terminal evidence')
    prefix, remaining = content.split(start)
    _, suffix = remaining.split(end)
    report.write_text(prefix + start + '\n' + text + '\n' + end + suffix)


if __name__ == '__main__':
    record(Path(sys.argv[1]), Path(sys.argv[2]))
