"""Check the released files against RELEASE_SHA256.txt and the minutes register.

    python scripts/verify_release.py
"""
from pathlib import Path
import csv
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    bad = []
    lines = (ROOT / "RELEASE_SHA256.txt").read_text().splitlines()
    for line in lines:
        digest, name = line.split("  ", 1)
        p = ROOT / name
        if not p.exists():
            bad.append(f"missing: {name}")
        elif sha(p) != digest:
            bad.append(f"changed: {name}")
    n_text = 0
    with open(ROOT / "provenance" / "minutes_source_register.csv", newline="") as f:
        for row in csv.DictReader(f):
            p = ROOT / row["text_file"]
            if p.exists():
                n_text += 1
                if sha(p) != row["text_sha256"]:
                    bad.append(f"minutes text hash mismatch: {row['text_file']}")
    print(f"{len(lines)} released files checked; {n_text} minutes texts checked against the source register.")
    if bad:
        print("\n".join(bad))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
