# Peer-review checklist (AIMO Interpretability, review period 2026-11-05 to 11-12 (live page https://aimo-interp.github.io, read 2026-10-24); report and open-sourced implementation due 2026-11-05 AoE)

Teams considered for awards review up to three other reports (rules, https://aimo-interp.github.io/,
read 2026-10-13). The same list is run on our own report before it goes out.

## Compliance (rules: Report, Transparency)

- [ ] The report documents the training data, the training code and the submission bundle.
- [ ] The submission depends only on documented resources (no undisclosed data or weights).
- [ ] The score reproduces from the report's instructions alone: run them from a clean copy, note
      the time and every step that needed a guess.
- [ ] The implementation is open-sourced, with a licence and pinned versions.
- [ ] No leakage of evaluation data (rules: Data); extra labelled data is allowed since 19 Sep.

## Science (writeup prize: scientific contribution; generalisation, actionable, negative or efficient)

- [ ] The claim is stated once, with its number and the set it was measured on.
- [ ] Splits hold out whole problems (each problem appears for up to four models); no choice is
      made on the evaluation folds.
- [ ] Gains are compared with the always-"robust" baseline (24/34 on the small-track train set)
      and come with an interval; at about 20 independent problems, a 1-2 configuration gain is noise.
- [ ] Transfer is tested: a held-out model family, the main track, or the validation set.
- [ ] Controls: a perturbation that should change the answer, a shuffled-label or constant baseline.
- [ ] Negative results report the predeclared gate and what was not tried.
- [ ] Runtime and hardware stated against the evaluation budget; quantisation noted.

## Our report, checked 2026-10-13

- [x] Data, code and bundle documented (Section 6, REPRODUCE.md).
- [x] Section 1 reproduces from a clean extract of the `package.sh` archive (anonymous archive 2026-10-28 at commit c162a8a, `supplementary.zip` 144 files, sha256 99fd115a…: manifest matches, identity gate clean, licence reads "the authors", fresh Python 3.12 venv from `requirements-cpu.txt` (torch 2.14.1+cpu), pinned data hash matched, problem text restored, 74 tests pass, `intervals.py` output identical, checks 3 min on 2 CPU threads; earlier final archive 2026-10-25 at commit 43d0baf, 130 files, zip sha256 94a63c6e…: manifest matches, no problem text, both data hashes match, 74 tests pass, `intervals.py` output identical, `split_audit.py` output identical for the pinned and the 2026-10-15 files, `artifact_gate.py score` identical; earlier rerun 2026-10-25 at commit ed29cc3 with the 43-row feature files in place, 123 files, zip sha256 dd0ed2be…: manifest and pinned data hash match, 69 tests pass, `intervals.py` output identical, `posthoc_auc.py` gives the 5 saved offline rows unchanged; first run 2026-10-24, archive at commit c9aba55: fresh Python 3.12 venv from `requirements-cpu.txt`, pinned data hash matched, problem text restored, the 69 tests in the REPRODUCE.md line pass, `intervals.py` output identical, `posthoc_auc.py` prints its 5 offline rows, 7 min on 2 CPU threads; the hidden-activation row needs the GPU feature run). The 2026-10-13 run also found the calibration and trace reports identical.
- [x] Problem-held-out nested protocol, predeclared gates, whole-problem bootstrap intervals.
- [x] Negative control in the trace method; untested baselines named.
- [ ] Team name, repository URL and the disclosure lines: entrant to fill (`report.tex` TODOs).

## Review template (paste one per reviewed report)

```
Summary of the claim (2 sentences):
Reproduction: ran / could not run; steps that needed guessing:
Compliance issues (data, code, bundle, leakage):
Main methodological concern, with the section:
What would change my assessment:
Strengths (2 lines):
```
