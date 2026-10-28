"""Location, status and download of third-party checkpoints (the TabPFN regressor).

The TabPFN predictors use the released TabPFN v3 regressor ``tabpfn-v3-regressor-v3_default.ckpt``
(Hugging Face repository ``Prior-Labs/tabpfn_3``). It is always loaded from
``$PFN_HOME/checkpoints/tabpfn/`` by absolute path, so TabPFN's own model cache and its default
model version are never consulted.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from esbopfn.config import paths

log = logging.getLogger(__name__)

TABPFN_FILENAME = "tabpfn-v3-regressor-v3_default.ckpt"
TABPFN_REPO = "Prior-Labs/tabpfn_3"
#: SHA-256 of the checkpoint file the thesis experiments ran with (233,289,807 bytes).
TABPFN_SHA256 = "311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301"
TABPFN_SIZE = 233_289_807


class CheckpointMissing(FileNotFoundError):
    """A required checkpoint file is absent. ``instructions`` says how to obtain it."""

    def __init__(self, path: Path, instructions: str) -> None:
        self.path = Path(path)
        self.instructions = instructions
        super().__init__(f"checkpoint missing: {self.path}\n{instructions}")


def tabpfn_dir(home: Path | str | None = None) -> Path:
    """Directory holding the TabPFN checkpoint (not created)."""
    return paths(home).tabpfn_checkpoints


def tabpfn_instructions(path: Path) -> str:
    return (
        "The TabPFN v3 regressor is gated behind a license acceptance by Prior Labs.\n"
        "Option 1: run `pfn checkpoints download tabpfn`. TabPFN opens a browser login once,\n"
        "  or uses an API key from https://ux.priorlabs.ai set as TABPFN_TOKEN (accept the\n"
        "  license on the Licenses tab first). A Hugging Face token with access to\n"
        f"  {TABPFN_REPO} may be needed as HF_TOKEN.\n"
        f"Option 2: download {TABPFN_FILENAME} from https://huggingface.co/{TABPFN_REPO}\n"
        f"  and run `pfn checkpoints download tabpfn --from <file>` or copy it to\n  {path}"
    )


def tabpfn_checkpoint(home: Path | str | None = None) -> Path:
    """Path of the TabPFN checkpoint, raising :class:`CheckpointMissing` if it is absent."""
    path = tabpfn_dir(home) / TABPFN_FILENAME
    if not path.is_file():
        raise CheckpointMissing(path, tabpfn_instructions(path))
    return path


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class CheckpointStatus:
    name: str
    path: Path
    present: bool
    size: int | None = None
    matches_thesis: bool | None = None

    def line(self) -> str:
        if not self.present:
            return f"{self.name:<10}missing   {self.path}"
        tag = {True: "thesis checkpoint", False: "differs from the thesis checkpoint",
               None: "not hashed"}[self.matches_thesis]
        return f"{self.name:<10}present   {self.path} ({self.size / 1e6:.1f} MB, {tag})"


def tabpfn_status(home: Path | str | None = None, verify: bool = False) -> CheckpointStatus:
    """Presence (and with ``verify`` the SHA-256 match) of the TabPFN checkpoint."""
    path = tabpfn_dir(home) / TABPFN_FILENAME
    if not path.is_file():
        return CheckpointStatus("tabpfn", path, False)
    size = path.stat().st_size
    match = (sha256(path) == TABPFN_SHA256) if verify else None
    return CheckpointStatus("tabpfn", path, True, size, match)


def download_tabpfn(home: Path | str | None = None, source: Path | str | None = None) -> Path:
    """Place the TabPFN checkpoint in ``$PFN_HOME/checkpoints/tabpfn``.

    With ``source`` the given local file is copied. Otherwise TabPFN's own downloader fetches the
    file from Hugging Face, which may ask for the license acceptance (see
    :func:`tabpfn_instructions`). Raises ``RuntimeError`` with the instructions on failure.
    """
    target_dir = tabpfn_dir(home)
    target = target_dir / TABPFN_FILENAME
    if target.is_file():
        log.info("TabPFN checkpoint already present at %s", target)
        return target
    target_dir.mkdir(parents=True, exist_ok=True)
    if source is not None:
        source = Path(source).expanduser()
        if not source.is_file():
            raise FileNotFoundError(source)
        tmp = target.with_suffix(".part")
        shutil.copyfile(source, tmp)
        tmp.rename(target)
        return target
    try:
        from tabpfn.constants import ModelVersion
        from tabpfn.model_loading import download_model
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(f"tabpfn is not importable: {exc}") from exc
    result = download_model(target, version=ModelVersion.V3, which="regressor",
                            model_name=TABPFN_FILENAME)
    if result != "ok" or not target.is_file():
        errors = "\n".join(str(e) for e in (result if isinstance(result, list) else []))
        raise RuntimeError(f"TabPFN download failed:\n{errors}\n\n{tabpfn_instructions(target)}")
    return target


#: Known checkpoints: name -> (status function, download function).
CHECKPOINTS = {"tabpfn": (tabpfn_status, download_tabpfn)}
