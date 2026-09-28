"""Resolve local checkpoints or explicitly pinned Hugging Face repositories."""
from pathlib import Path


INFERENCE_PATTERNS = ["model.json", "head.safetensors", "joint.safetensors",
                      "adapter/*.json", "adapter/*.safetensors", "tokenizer/*",
                      "backbone/*.json", "backbone/*.safetensors", "SHA256SUMS.json"]


def resolve_checkpoint(value, revision=None):
    value = str(value)
    local = Path(value).expanduser()
    if local.is_dir():
        if revision is not None:
            raise ValueError("--revision applies only to a Hub checkpoint")
        if not (local / "model.json").is_file():
            raise ValueError(f"missing model.json in {local}")
        return local
    if local.is_absolute() or value.startswith(('.', '~')):
        raise FileNotFoundError(local)
    repo, separator, embedded = value.removeprefix('hf://').partition('@')
    if separator:
        if revision is not None and revision != embedded:
            raise ValueError("conflicting checkpoint revisions")
        revision = embedded
    if not revision:
        raise ValueError("checkpoint directory does not exist; Hub downloads require owner/repo@revision")
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(repo_id=repo, revision=revision, allow_patterns=INFERENCE_PATTERNS))
    if not (path / 'model.json').is_file():
        raise ValueError("the Hub repository is not a Qev checkpoint (model.json missing)")
    return path
