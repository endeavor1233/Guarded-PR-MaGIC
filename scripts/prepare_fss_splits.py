"""Download and validate the public FSS-1000 category splits.

The split files are published in the official Matcher release at:
https://github.com/aim-uofa/Matcher/tree/main/datasets/FSS-1000/splits

This script writes trn.txt/val.txt/test.txt below the supplied FSS-1000
directory. It does not move or modify image data.
"""

import argparse
import base64
import os
import sys
from urllib.request import Request, urlopen


SPLITS = {
    "trn.txt": {
        "raw": "https://raw.githubusercontent.com/aim-uofa/Matcher/main/datasets/FSS-1000/splits/trn.txt",
        "api": "https://api.github.com/repos/aim-uofa/Matcher/git/blobs/7e131276d38efc8543accc6e3048bd313294643e",
        "expected": 520,
    },
    "val.txt": {
        "raw": "https://raw.githubusercontent.com/aim-uofa/Matcher/main/datasets/FSS-1000/splits/val.txt",
        "api": "https://api.github.com/repos/aim-uofa/Matcher/git/blobs/30dfb203710764df702e74d2f7e5e8763e92c7fe",
        "expected": 240,
    },
    "test.txt": {
        "raw": "https://raw.githubusercontent.com/aim-uofa/Matcher/main/datasets/FSS-1000/splits/test.txt",
        "api": "https://api.github.com/repos/aim-uofa/Matcher/git/blobs/32d0bd083facea98ee28c04383c7511da5d21c9d",
        "expected": 240,
    },
}


def fetch_text(url):
    request = Request(url, headers={"User-Agent": "PR-MaGIC-FSS-split-preparer"})
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8-sig")


def fetch_split(spec):
    errors = []
    for url in (spec["raw"], spec["api"]):
        try:
            text = fetch_text(url)
            # GitHub API blob responses are JSON; avoid importing requests.
            if url.startswith("https://api.github.com/"):
                import json
                payload = json.loads(text)
                text = base64.b64decode(payload["content"]).decode("utf-8-sig")
            return text
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Unable to download split:\n" + "\n".join(errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fss-root",
        required=True,
        help="Absolute path to .../FSS-1000",
    )
    args = parser.parse_args()

    fss_root = os.path.abspath(args.fss_root)
    if not os.path.isdir(fss_root):
        raise FileNotFoundError(fss_root)

    split_dir = os.path.join(fss_root, "splits")
    os.makedirs(split_dir, exist_ok=True)

    written = {}
    all_categories = []
    for filename, spec in SPLITS.items():
        text = fetch_split(spec)
        categories = [line.strip() for line in text.splitlines() if line.strip()]
        if len(categories) != spec["expected"]:
            raise RuntimeError(
                f"{filename}: expected {spec['expected']} categories, "
                f"received {len(categories)}"
            )
        if len(set(categories)) != len(categories):
            raise RuntimeError(f"{filename}: duplicate categories found")
        all_categories.extend(categories)

        output_path = os.path.join(split_dir, filename)
        with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(categories) + "\n")
        written[filename] = output_path

    if len(set(all_categories)) != 1000:
        raise RuntimeError("The three split files do not contain 1000 unique categories")

    data_dir = os.path.join(fss_root, "data")
    category_dir = data_dir if os.path.isdir(data_dir) else fss_root
    local_categories = {
        name for name in os.listdir(category_dir)
        if os.path.isdir(os.path.join(category_dir, name)) and name != "splits"
    }
    missing = sorted(set(all_categories) - local_categories)
    if missing:
        print(
            f"WARNING: {len(missing)} split categories are not present under "
            f"{category_dir}. First entries: {missing[:10]}",
            file=sys.stderr,
        )
    else:
        print(f"Validated {len(local_categories)} local category directories under {category_dir}")

    print("Wrote public FSS-1000 splits:")
    for filename, path in written.items():
        print(f"  {filename}: {path}")
    print("Source: https://github.com/aim-uofa/Matcher/tree/main/datasets/FSS-1000/splits")


if __name__ == "__main__":
    main()

