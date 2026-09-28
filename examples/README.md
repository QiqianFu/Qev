# Small support examples

`requests.jsonl` contains one unlabelled request. `train.jsonl` has six labelled requests and `dev.jsonl` has two. Each request asks a Choice, Noul and Score question. These original examples demonstrate formats and exercise the offline smoke workflow; they are not a meaningful training or evaluation corpus.

Run `python scripts/smoke.py --out runs/smoke` from the repository root to train and run a tiny random model without a download. For Qev-9B, follow the main README with an exported checkpoint.
