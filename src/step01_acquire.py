"""
Step 01 - Acquire the CFPB Consumer Complaint Database and verify integrity.

Reproduces the acquisition described in report Section 4.1, including the ZIP
end-of-central-directory check that caught the truncated first download.

Run:  python src/step01_acquire.py
"""

import sys
import zipfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

EOCD_SIGNATURE = b"PK\x05\x06"


def download(url: str, target: Path) -> None:
    """Stream the archive to disk with progress reporting."""
    print(f"Downloading {url}")
    print(f"       ->  {target}")

    def _hook(block_num, block_size, total_size):
        if total_size <= 0:
            return
        done = min(block_num * block_size, total_size)
        pct = 100.0 * done / total_size
        print(f"\r  {done/1e9:.2f} / {total_size/1e9:.2f} GB  ({pct:5.1f}%)", end="")

    urllib.request.urlretrieve(url, target, reporthook=_hook)
    print()


def verify_zip(path: Path) -> bool:
    """
    Check the archive is complete before spending time on extraction.

    A truncated download is still a readable file, so the failure mode is
    a confusing error deep inside extraction. Checking for the end-of-central-
    directory signature in the trailing bytes catches it immediately.
    """
    if not path.exists():
        print(f"FAIL  archive not found: {path}")
        return False

    size = path.stat().st_size
    print(f"Archive size: {size/1e9:.3f} GB")

    with open(path, "rb") as fh:
        fh.seek(max(0, size - 66_000))
        tail = fh.read()

    if EOCD_SIGNATURE not in tail:
        print("FAIL  no ZIP end-of-central-directory signature.")
        print("      The download is incomplete. Delete the file and re-run.")
        return False
    print("PASS  end-of-central-directory signature present")

    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            if bad is not None:
                print(f"FAIL  corrupt member: {bad}")
                return False
            for info in zf.infolist():
                print(f"      member: {info.filename}  "
                      f"({info.file_size/1e9:.3f} GB uncompressed)")
        print("PASS  archive structure valid")
        return True
    except zipfile.BadZipFile as exc:
        print(f"FAIL  not a valid ZIP: {exc}")
        return False


def extract(archive: Path, target_csv: Path) -> None:
    print(f"Extracting to {target_csv}")
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not members:
            raise RuntimeError("No CSV member found inside the archive.")
        member = members[0]
        with zf.open(member) as src, open(target_csv, "wb") as dst:
            while True:
                block = src.read(8 * 1024 * 1024)
                if not block:
                    break
                dst.write(block)
    print(f"Extracted: {target_csv.stat().st_size/1e9:.3f} GB")


def main() -> int:
    if cfg.RAW_CSV.exists():
        print(f"CSV already present: {cfg.RAW_CSV} "
              f"({cfg.RAW_CSV.stat().st_size/1e9:.3f} GB)")
        print("Delete it if you want a fresh download. Nothing to do.")
        return 0

    if not cfg.RAW_ZIP.exists():
        download(cfg.CFPB_BULK_URL, cfg.RAW_ZIP)
    else:
        print(f"Archive already present: {cfg.RAW_ZIP}")

    if not verify_zip(cfg.RAW_ZIP):
        return 1

    extract(cfg.RAW_ZIP, cfg.RAW_CSV)
    print("\nStep 01 complete. Next: python src/step02_audit.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
