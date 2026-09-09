# Conference results and model code (public copy)

Per-seed result files, model definitions, and experiment/training-launch
scripts backing an ICASSP 2027 submission on three-class classification of
sleep-stage transitions in mouse cortical EEG: `NREM_to_MA`, `NREM_to_Awake`,
`Stable_NREM`.

Labels are expert-reviewed. Citation to a manuscript in preparation.

## Layout

```
results/<model>/seed_<n>/
  provenance.json             library/config metadata for that run
  summary/protocol_summary.json
                               reported metrics: locked-test-set macro/weighted
                               F1, accuracy, balanced accuracy, per-class F1,
                               best epoch, etc.
results_eeg_only/<model>/seed_<n>/
  same, for the single-channel (EEG-only) input variant

models/
  multiscale_dual_path.py     MultiScaleDualPath7ChannelEEGNet (MSDPN) -- the
                               paper's own architecture: dual-path (raw
                               time-domain + FFT frequency-domain) with
                               multi-scale convolutions and hierarchical
                               attention.
  eegnet.py, baselines.py,
  eegnextformer.py,
  braindecode_arm.py          in-house and reference architectures.
  registry.py                 single entry point (`build_model`) routing
                               every arm in the experiment matrix, including
                               calls into `foundation.build_foundation_model`
                               for the pretrained foundation-model arms.

experiments/
  run_protocol.py, run_cv.py,
  run_sweep.py                training/evaluation entry points.
  protocols.py                protocol definitions (subject-held-out, etc.).
  arm_*.json, repro_*.json    per-arm experiment configs.
  _sweep_overrides/           per-arm config overrides and sweep manifests.
  slurm/                      HPC job-submission scripts for the above
                               (paths sanitized to <REPO_ROOT>/<REPO_PARENT>;
                               these encode the exact per-arm interpreter and
                               launch commands used, not just the configs).
```

All results shown are evaluated once on a fixed, held-out set of test
subjects that never entered training or model selection for any arm.

Model directories named after specific architectures (REVE, LaBraM, BIOT,
BENDR, SleepFM, EEGNet, and the braindecode reference architectures) are
included, alongside the paper's own architecture (`multiscale_dual_path.py`).
