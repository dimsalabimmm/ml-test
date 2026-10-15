"""Package the local checkpoint, keeping large files out of Git history."""

import hashlib
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts" / "rubert_tiny2_best"
ASSET_DIR = ROOT / ".release-assets"
FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]


def main():
    for filename in FILES:
        if not (MODEL_DIR / filename).is_file():
            raise FileNotFoundError(MODEL_DIR / filename)
    ASSET_DIR.mkdir(exist_ok=True)
    archive = ASSET_DIR / "rubert_tiny2_best.zip"
    temporary = archive.with_suffix(".zip.part")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for filename in FILES:
            path = MODEL_DIR / filename
            zipped.write(path, path.relative_to(ROOT))
    temporary.replace(archive)
    checksum = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
    checksum_path = ASSET_DIR / "SHA256SUMS.txt"
    checksum_path.write_text(f"{checksum}  {archive.name}\n", encoding="utf-8")
    print(archive)
    print(checksum_path)


if __name__ == "__main__":
    main()
