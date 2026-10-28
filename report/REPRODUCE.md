# Reproducing the report

Sections 1 to 4 of `report.tex` reproduce on a CPU in about seven minutes on two CPU threads from the saved features in
this archive and the data files below. Item 1 of Section 1 (label agreement) is recomputed by
`report/lesson9_shuffle.py` from the 2026-10-15 file (section 5 below). The hidden-activation
features need a GPU to regenerate (the CPU run reads their saved predictions); regenerating any
feature set needs a GPU and is described after that.

## 1. Environment (Python 3.12, CPU)

```
python3.12 -m venv .venv
.venv/bin/pip install -r report/requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## 2. Data

The organisers' training set is not redistributed here (licence "other"). Download it at the
revision the runs used and check its hash:

```
.venv/bin/python -c "from huggingface_hub import hf_hub_download as d; print(d('aimo-interp/train-main-v2', 'data/train-00000-of-00001.parquet', repo_type='dataset', revision='23791d9b120132087877b0459f953d5959fc8455', local_dir='data/train-main-v2-hf'))"
mkdir -p data/train-main-v2 && cp data/train-main-v2-hf/data/train-00000-of-00001.parquet data/train-main-v2/
sha256sum data/train-main-v2/train-00000-of-00001.parquet
# bedfb551ea0f4d95d5e70e1df7cf61395ce2d36eeba2869aba6a90fc3c198824
```

The other files, all Hugging Face datasets of `aimo-interp`, are not in this archive. Their sha256
values are below; only the revision of `augmented-sample-math-agg` is known (the organisers' kit pins
`f972ced0705096f8d7ca7fac30825900b8b7fb6a`, and our copy's revision was not recorded). For the rest, no
revision was recorded; the files are dated in the data folder as shown.

| Dataset (file) | Folder date | sha256 |
|---|---|---|
| `aimo-interp/train-main-v2` at revision `fa8af758c6519fc97a9d0bd9017feb6f12232b52` (2026-10-15; `train-2026-10-15.parquet`) | 2026-10-17 | `e18eb44207e7987781ec2ff214793196038442bdc1466fdf5498f3e87ec58285` |
| `aimo-interp/aimo-interp-challenge-sample-full` (`validation-00000-of-00001.parquet`) | 2026-10-11 | `2989ab6a018f2779a821209f414bd0049c9ef1c2488f589ff2d3bf4ed16942ff` |
| `aimo-interp/augmented-sample-math-agg` (`validation-00000-of-00001.parquet`) | 2026-10-11 | `b785750259a3f4524cbaf988f51660197cbdfca19d8f45cc35347d4ef29aa4db` |
| `aimo-interp/augmented-sample-math-agg-filtered` (`validation-00000-of-00001.parquet`) | 2026-10-11 | `b8a485868ce5a14d0c961e009e30bc61f850ad2d2224a01ae1ada401fb73ed3d` |

The saved run files key rows by problem; in this archive the problem text is replaced by
`sha256:<16 hex>`. Put it back:

```
.venv/bin/python report/rehydrate.py restore . data/train-main-v2/train-00000-of-00001.parquet
```

## 3. Checks and the report's numbers

```
.venv/bin/python -m pytest -q test_calibration.py test_invariance.py test_trace_features.py report/test_lesson6_counts.py report/test_lesson9_counts.py report/test_lesson9_shuffle.py report/test_per_model_auc.py report/test_flips_by_source.py report/test_trace_control.py report/test_intervals_activation.py report/test_posthoc_auc.py report/test_sample_transfer.py report/test_forest.py report/test_split_audit.py
.venv/bin/python report/intervals.py > intervals-rerun.json
diff <(python3 -m json.tool report/intervals.json) <(python3 -m json.tool intervals-rerun.json) && echo "Section 1 reproduced"
```

`intervals.py` refits every method with the code that produced it (`fit_probe.py`,
`sub/calibration.py`, `fit_trace_probe.py`), reads the hidden-activation probe from its saved
predictions (`runs/activation-r3/probe.json`: problem hash, model, label, probe and majority
prediction per row; the problem text is matched by sha256), asserts that the correct counts match the saved
reports, and recomputes the paired bootstrap intervals (10,000 whole-problem resamples, fixed seed).
Section 1, item 1 (needs the 2026-10-15 file) and the post hoc ranking of Appendix B.2 (the same
refits as `intervals.py`; its hidden-activation row also reads `runs/activation-r3/features`, which
the archive does not ship: without it the script prints the other five rows and notes the skipped row on stderr; the
activation row needs the GPU feature run of section 4 first):

```
.venv/bin/python report/lesson9_shuffle.py data/train-main-v2/train-2026-10-15.parquet
.venv/bin/python report/split_audit.py --out split-audit-rerun.json              # compare with report/split-audit.json (about 9 min on 2 CPU threads)
.venv/bin/python report/split_audit.py --train 1015 --out split-audit-1015-rerun.json   # report/split-audit-1015.json (about 14 min)
.venv/bin/python report/posthoc_auc.py > posthoc_auc-rerun.json    # compare with report/posthoc_auc.json
.venv/bin/python report/forest.py                                  # rewrites report/forest.png from intervals.json
```

Each method can also be run on its own:

```
.venv/bin/python fit_probe.py runs/feat/features.jsonl              # prompt-only features -> runs/probe-gate.json
.venv/bin/python sub/calibration.py runs/feat/features.jsonl        # calibration -> runs/calibration-nested/report.json
.venv/bin/python fit_trace_probe.py runs/trace-pilot/rows.jsonl trace-probe.json
```

## 4. Regenerating the features (GPU)

- Prompt-only features (`runs/feat/features.jsonl`): `python kaggle/make_feat.py` writes the
  Kaggle kernel `kaggle/feat/run.py` (the modules are inlined with their hashes); it ran on one
  L4 (`kaggle/feat`) and on CPU (`kaggle/feat-cpu`).
- Counterfactual trace features (`runs/trace-pilot/rows.jsonl`): `python run_trace_pilot.py
  OUT.jsonl [MINUTES]` on one 16 GB GPU, reading local copies of the four small-track models
  (the `MODELS` path at the top of the script). Qwen3.5-4B in BF16, the three 7-8B models in 4-bit
  NF4; 6.5-10.9 s per problem.
- Generation stability (`runs/agreement-vs-labels.csv`): `python kaggle/make_perturb_gen.py`, then
  the `kaggle/gen-*` kernels, then `python analyze_gen.py`.

### 43-row refit

Data: `train-2026-10-15.parquet`, sha256 `e18eb44207e7987781ec2ff214793196038442bdc1466fdf5498f3e87ec58285`;
94 rows, 43 labelled small-track rows, 22 problems. The extraction commands only add missing rows:

```
python run_trace_pilot.py runs/trace-pilot/rows.jsonl 60 --train 1015 --append --models Skywork/Skywork-OR1-Math-7B
python run_activation_probe.py --out runs/activation-r3/features --train 1015 --append --only Skywork/Skywork-OR1-Math-7B
python kaggle/make_feat.py
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 AIMO_LANE=<lane> OUT=<dir with features.jsonl> ONLY_LABELLED=1 SKIP_INSTALL=1 python <lane>/kaggle/feat/run.py --train 1015 --append --models Skywork/Skywork-OR1-Math-7B
```

Run the last one from a scratch folder holding a `Skywork` link to the local model folder. Then,
in a copy of the lane (the fitters overwrite `runs/probe.json`, `runs/probe-gate.json`,
`runs/calibration-nested/report.json` and the two `probe.json` files):

```
python fit_probe.py runs/feat/features.jsonl --train 1015
python sub/calibration.py runs/feat/features.jsonl --train 1015
python fit_trace_probe.py runs/trace-pilot/rows.jsonl runs/trace-pilot/probe.json --train 1015
python fit_activation_probe.py --features runs/activation-r3/features --out runs/activation-r3/probe.json --train 1015
python report/intervals.py --train 1015 > report/intervals-43.json
python report/posthoc_auc.py --train 1015 > report/posthoc_auc-43.json
python report/balanced_gates.py runs
```

The default (pinned) commands still reproduce `report/intervals.json` and `report/posthoc_auc.json`
byte for byte with the extended feature files in place.

## 5. Data revisions and the later lessons

| Part of the report | Data | Script and run folder |
|---|---|---|
| Sections 1-4 (results table in Section 4) | `train-main-v2` at Hugging Face revision `23791d9b`, 82 rows (`train-00000-of-00001.parquet`, sha256 `bedfb551…`) | `fit_probe.py`, `sub/calibration.py`, `fit_trace_probe.py`, `report/intervals.py`; `runs/feat`, `runs/probe-gate.json`, `runs/calibration-nested`, `runs/trace-pilot` |
| Lesson 5 (hidden states) | same revision | `run_activation_probe.py`, `fit_activation_probe.py`; `runs/activation-r3/probe.json` |
| Lesson 6 (solved problems that are still non-robust) | the organisers' `augmented-sample-math-agg-filtered/validation-00000-of-00001.parquet` (sha256 `b8a485868ce5a14d0c961e009e30bc61f850ad2d2224a01ae1ada401fb73ed3d`) and `augmented-sample-math-agg/validation-00000-of-00001.parquet` (sha256 `b785750259a3f4524cbaf988f51660197cbdfca19d8f45cc35347d4ef29aa4db`) (the same model, DeepSeek-R1-0528-Qwen3-8B, whose id appears as `qwen3-8b:low` in the second; the first's 52 problems are all in the second) | `python report/lesson6_counts.py FILTERED.parquet AGG.parquet`; tested by `report/test_lesson6_counts.py` |
| Lesson 7 (no final answer within 4,096 tokens) | not recorded | `answer_flips.py generate --no-force`; no run folder for the 36-sample check was kept, so this number cannot be reproduced from the archive |
| Lesson 8 (forced final answer) | the `train-main-v2` file of 2026-10-15 (`data/train-main-v2/train-2026-10-15.parquet`, sha256 `e18eb44207e7987781ec2ff214793196038442bdc1466fdf5498f3e87ec58285`; 94 rows, not the pinned revision; 80 labelled) plus the 52 DeepSeek rows of the organisers' `augmented-sample-math-agg-filtered` set (hash above), as `runs/flip-rows-v2.parquet` (95 rows; not in the archive, since it carries the organisers' problem text: rebuild it from those two files) | `answer_flips.py generate` (forcing is the default) and `evaluate`; `runs/force-r4/` (`*.jsonl`, `eval-3models.json`); per-model AUCs: `python report/per_model_auc.py runs/force-r4/{Qwen3.5-4B,Skywork-OR1-Math-7B,DeepSeek-R1-0528-Qwen3-8B}.jsonl --pool Qwen3.5 --pool Skywork` (0.703 and 0.685 per model, 0.697 pooled on 24 rows; DeepSeek 0.348 on 62, which mixes two label sources); by label source: `python report/flips_by_source.py runs/flip-rows-v2.parquet runs/force-r4/{Qwen3.5-4B,Skywork-OR1-Math-7B,DeepSeek-R1-0528-Qwen3-8B}.jsonl` (train-file rows: AUC 0.706 on 34 rows, held-out 18 against a majority of 24; augmented rows: AUC 0.402 on 52, held-out 41 against 41; it also prints problem-grouped bootstrap 95 % intervals, 4,000 draws, seed 0: train file all [0.44, 0.87], organisers' set [0.22, 0.59], DeepSeek train rows [0.12, 1.00], difference 0.304 [-0.01, 0.54]; with `--permutations data/augmented-sample-math-agg-filtered/validation-00000-of-00001.parquet` it adds the organisers' rows whose problem has only typos, distract or rename permutations in every source row: 12 rows, 4 robust, AUC 0.547 [0.10, 0.89]), tested by `report/test_per_model_auc.py`, `report/test_flips_by_source.py` |
| Lesson 3 (control moved the likelihood more in 33 of 42) | `runs/trace-pilot/rows.jsonl` | `python report/trace_control.py runs/trace-pilot/rows.jsonl --labels data/train-main-v2/train-00000-of-00001.parquet` (49 rows: the 37-row pilot plus 12 Skywork rows of the 43-row refit; 42 with a control, 33 moved more, median ratio 0.62; labelled: 29 with a control, 22, 0.62); tested by `report/test_trace_control.py` |
| Generation stability (Section 4) | `runs/gen-qwen3-5-4b` (not in the archive: the generations quote the problems; regenerate them as in section 4) | `python analyze_gen.py --labelled` prints the largest Spearman on the 12 labelled rows (0.24, p = 0.46); needs `scipy` |
| Lesson 9 (counts) | pinned revision and the 2026-10-15 file, plus `aimo-interp-challenge-sample-full` | `python report/lesson9_counts.py TRAIN.parquet [TRAIN2.parquet] --public PUBLIC.parquet`; label-shuffle null (Section 1, item 1): `python report/lesson9_shuffle.py data/train-main-v2/train-2026-10-15.parquet`; tests `report/test_lesson9_counts.py`, `report/test_lesson9_shuffle.py`; sample transfer (Appendix C): `python report/sample_transfer.py data/train-main-v2/train-00000-of-00001.parquet PUBLIC.parquet`, tested by `report/test_sample_transfer.py` |
| Length baseline for the artifact's 0.626 (Section 5, Appendix C) | the organisers' `augmented-sample-math-agg` validation parquet (141 rows, 137 problems) | `python report/length_baseline.py` (run from the lane root; prints balanced accuracy over 20 seeds, the 200-shuffle null and the reference lines) |
| The organisers' artifact on our labels (Section 5, Appendix C) | the 2026-10-15 file (10 DeepSeek rows) and the kit's `uncertainty_artifacts/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.joblib` | `python report/artifact_gate.py features --train data/train-main-v2/train-2026-10-15.parquet --model-path <DeepSeek-R1-0528-Qwen3-8B> --out runs/artifact-gate --batch-size 4 --gpu-gib 5` (GPU; 4 h 50 min on a 12 GB card), then `python report/artifact_gate.py score --out runs/artifact-gate` in the kit's environment (the joblib needs scikit-learn 1.8.0 and torch); `runs/artifact-gate/` (features, manifest, score) is in the archive |
| Post hoc AUC and prior-matched accuracy (Appendix B.2); forest plot (Section 4) | same as Sections 1-4; the activation row needs the unshipped activation features | `python report/posthoc_auc.py` (asserts that each score reproduces the reported correct count), `python report/forest.py`; tested by `report/test_posthoc_auc.py`, `report/test_forest.py` |
| Balanced accuracy and prevalence (Section 1) | saved run files | `python report/balanced_gates.py runs` |

The 2026-10-15 file, the aggregated DeepSeek set and the two augmented sets of Lesson 6 are not in
this archive.

The two hybrids ship the organisers' uncertainty-profiling solution, which comes from their
getting-started repository, `https://github.com/aimo-interp/getting-started`, at commit
`de79405` (path `solutions/uncertainty-profiling`). The files bundled in the hybrid zips are byte-identical to the kit's and have these sha256 values:

| File | sha256 |
|---|---|
| `uncertainty_artifacts/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.joblib` | `bc90bd8eed4d9f6383cbb9ae1626896afea90b0e71c46d9a56bc87d376f85592` |
| `uncertainty_profile/__init__.py` | `e6a6044f83d60c86154155754d2d4b979ade9c3ed86a34997c746f9b266493ec` |
| `uncertainty_profile/artifact.py` | `7191fedd8b1a5374dc81551a58ace7a2469497834c358988c049091bfafecc79` |
| `uncertainty_profile/config.py` | `49085a83aad82cbe9ab743de923d1f1d07c2728e78729fe3bc71acf149319430` |
| `uncertainty_profile/extraction.py` | `b3f159faf948bd2525ba0de262bc80942b035d232512b6d27d0906e4917b6614` |
| `uncertainty_profile/inference.py` | `33070a8b730647892e4efabf7bc49e952e9f1c3df680e392353b0fab9eb66daa` |
| `uncertainty_profile/metrics.py` | `5d545aeaa2456decce93fb88cb4b0765ce36367c522fa9d9b66dc57273fcabcc` |

Its README documents a grouped out-of-fold balanced accuracy of 0.6262 and accuracy of 0.6879
(lines 127-128; DeepSeek only, on the organisers' aggregated set, not on our labels).

## 6. Submission history

Hashes are `sha256sum` of the zips in the lane's `dist/`; `report/package.sh` ships them all under
`dist/` in the archive (set `DIST` to that folder). The two hybrids of 2026-10-19 take
DeepSeek's predictions from the organisers' bundled uncertainty-profiling artifact, which is not
one of the six signals tested in the report; scored afterwards on the 10 DeepSeek rows of the
2026-10-15 file, it gets 2 against a majority of 7 (`runs/artifact-gate/score.json`).

| File | Uploaded | Task | Id | Public score | sha256 |
|---|---|---|---:|---:|---|
| `dist/always-true-main.zip` | 2026-10-11 | main | 977178 | 0.48 | `ba38cb0e7b55dd3a325c9d01b6d461f3effb836f3a245f57fe09fe7d905fe392` |
| `dist/always-true-small.zip` | 2026-10-11 | small | 977183 | not recorded (first small upload went to the main task and failed ingestion) | `a3748c648a5a0d9a56fa67cf013c6690d755b7f0a55647f37ccade75825ddbc2` |
| `dist/probe/main-qwen4b.zip` | 2026-10-18 | main | not recorded | net 0 | `d1b2b9e455dfd397c0d9738b05627f4bfd03670f6b23b3123d101239a79a57c7` |
| `dist/probe/main-olmo.zip` | 2026-10-18 | main | not recorded | net 0 | `5036f524db8a15a16ac85fc957e4eedf76fca92350f423fc03843a6a8ab415e0` |
| `dist/probe/main-deepseek.zip` | 2026-10-18 | main | not recorded | net -1 (cancelled on 2026-10-19 per our submission log) | `4102034bdf0cf000f0b34c375f783e52f748c05dad0830b176847fd654440882` |
| `dist/probe/main-gptoss-low.zip` | 2026-10-18 | main | not recorded | net -1 | `ca407e397507ccaab4f4024935c924d6802727523b187e3aed5c203ba3c80c43` |
| `dist/probe/main-gptoss-medium.zip` | 2026-10-18 | main | not recorded | net +1 (0.57) | `12414c2268900c4f37420b7a9881521dc297c991b17fe0686fbfac3e87cc4122` |
| `dist/probe/main-skywork.zip` | 2026-10-18 | main | not recorded | none (pending, cancelled) | `8a9b311ef173ebc4d2b854295650a153ac0b42c23eff0a0387fee65688a643d1` |
| `dist/probe/small-qwen4b.zip` | 2026-10-18 | small | not recorded | net +1 (0.57) | `0e8b54c7517dca563e8482b7ca0934f4962c877415d0c7efcaa515fbc75e65e5` |
| `dist/probe/small-skywork.zip` | 2026-10-18 | small | not recorded | net -1 | `14082fe5cf9928660e4a4285a05bf34f00aec9aeec257d24f8ef22b0764020b3` |
| `dist/probe/small-olmo.zip` | 2026-10-18 | small | not recorded | net -1 | `29a6aec09ba052133f86359dc11b5330b35783357d098dd612aaae5c27312246` |
| `dist/entry/main-hybrid-v1.zip` | 2026-10-19 | main | 1002314 | 0.57 | `d2df7dc8a0daaf850636c717e267579058a5fd53856c56365dfae3809faf671b` |
| `dist/entry/small-groups-v1.zip` | 2026-10-19 | small | 1002316 | 0.64 | `82344d44e9b9d86e68cee3750125c13fa39d9597c4069f379412b79dfa6cfafc` |
| `dist/entry/small-hybrid-v1.zip` | 2026-10-19 | small | 1002317 | 0.57 | `376c170a05b963c8706195f4cc26730375c5f984a02ebf2e4e46e6c7e706dc0c` |
| `dist/entry/main-training-v3.zip` | 2026-10-27 | main | 1014009 | 0.48 | `c315e32fc6392fb322351dcd96980e180dec248e2c381ac48f7c90dd90a908b7` |
| `dist/entry/small-training-v3.zip` | 2026-10-27 | small | 1014010 | 0.50 | `470ceba57ca85278458726dc5d990f1a2db4a850379a3c8f1745b26ae4c937bf` |

"Net" is our submission log's net robust count: score times problems, minus the all-non-robust score
(public sets: 21 main and 14 small problems). The log lists ten probe uploads; the tenth
(small, DeepSeek, net +1) has no zip in `dist/probe/`. The probe zips are single-configuration
groups; the group for each hybrid and for `small-groups-v1` was chosen from these per-configuration
public counts, which the log itself calls noise-sized at +/-1. Uploading a zip to
https://www.codabench.org/competitions/16180/ reproduces its score only if the public set is
unchanged. Guarded v2 hybrids (`entry/`, built by `entry/build.sh`) are prepared, not uploaded; their zips
ship too: `dist/entry/main-hybrid-v2.zip` (`39d1752a8d6d8fe143e5e9f9341c469f57473ef5ecc78ca51dfdfcd19bae565e`)
and `dist/entry/small-hybrid-v2.zip` (`b6d1e6e363c554f82044848cb0dec975d7bf7d96b540f023a082fc3ee5d1ebef`).

The v3 training-only entries of 2026-10-27 are rebuilt with `entry/build.sh v3`, which writes
`main-training-v3.zip` and `small-training-v3.zip` to `dist/entry/` (`OUT_DIR` overrides the folder)
from `solution.py` and `entry/training-prior.json` (plus `small.txt` on small), with fixed timestamps and
file order, so two builds give the same bytes (`entry/test_entry_v3.py`, line 175). The prior is fitted by
`entry/fit_training_prior.py` on the organisers' 2026-10-15 training file (Hugging Face revision
`fa8af758c6519fc97a9d0bd9017feb6f12232b52`, sha256
`e18eb44207e7987781ec2ff214793196038442bdc1466fdf5498f3e87ec58285`) and uses no public-set signal.

```
.venv/bin/python -c "from huggingface_hub import hf_hub_download as d; print(d('aimo-interp/train-main-v2', 'data/train-00000-of-00001.parquet', repo_type='dataset', revision='fa8af758c6519fc97a9d0bd9017feb6f12232b52', local_dir='data/train-main-v2-1015'))"
cp data/train-main-v2-1015/data/train-00000-of-00001.parquet data/train-main-v2/train-2026-10-15.parquet
.venv/bin/python entry/fit_training_prior.py --train data/train-main-v2/train-2026-10-15.parquet --pinned data/train-main-v2/train-00000-of-00001.parquet --out prior-rerun.json
cmp prior-rerun.json entry/training-prior.json && echo "prior reproduced"
OUT_DIR=v3-rerun bash entry/build.sh v3 && sha256sum v3-rerun/*.zip
# c315e32fc6392fb322351dcd96980e180dec248e2c381ac48f7c90dd90a908b7  v3-rerun/main-training-v3.zip
# 470ceba57ca85278458726dc5d990f1a2db4a850379a3c8f1745b26ae4c937bf  v3-rerun/small-training-v3.zip
```

Both steps were rerun on 2026-10-27: the refitted prior is byte-identical and the two zips match the uploaded hashes.

`MANIFEST.sha256` lists every file in this archive with its hash (`sha256sum -c MANIFEST.sha256`).
