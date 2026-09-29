"""Resolve local checkpoints or Hugging Face model repositories."""
from pathlib import Path


INFERENCE_PATTERNS = ["model.json", "head.safetensors", "joint.safetensors",
                      "adapter/*.json", "adapter/*.safetensors", "tokenizer/*",
                      "backbone/*.json", "backbone/*.safetensors",
                      "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "licenses/*"]


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
    if separator and not embedded:
        raise ValueError("checkpoint revision after @ must not be empty")
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(repo_id=repo, revision=revision, allow_patterns=INFERENCE_PATTERNS))
    if not (path / 'model.json').is_file():
        raise ValueError("the Hub repository is not a Qev checkpoint (model.json missing)")
    return path
