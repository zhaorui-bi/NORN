#!/usr/bin/env python3
"""Fetch hash-pinned model support for the workspace's Zahirovic2022 GMTs.

No raw observations/GMTs are downloaded. Existing model files are never
replaced. The current PMM CombinedRotations.rot has a different absolute frame
and is deliberately NOT used; the export's original rotation file is pinned.
"""

import argparse
import hashlib
import json
from pathlib import Path
import stat
import urllib.request
import zipfile

ROTATION = "Zahirovic_etal_2022_OptimisedMantleRef_and_NNRMantleRef.rot"
ROTATION_URL = (
    "https://raw.githubusercontent.com/Computational-Planet/reconstructable-tile-layer/"
    "05c0417e067a1bedca8e81f93d374a29dab83fb7/"
    f"apps/reconstructable-tile-layer-demo/public/rotations/{ROTATION}"
)
ROTATION_SHA256 = "a2edf515af7f7e4771d58f438b398cb9332826bacffcd0c7606331615a3f4348"
ARCHIVES = {
    "StaticPolygons": "397a72d5ec6db4945a7ae1c9bb68e2109c2e7bb5df6a3fd90c399473bfb8b314",
    "ContinentalPolygons": "9d00215fa750dc3d2045ff600dfe9605dbb7a40ac138423dfb358bdd854629d7",
    "Topologies": "14856b6fc5e631b5e7edd1f63c35f9e1692433373170441716e0d739ce9c54c7",
}


def store(path, content):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Refusing to replace an existing model file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


def fetch(url, sha256, path):
    path = Path(path)
    if path.exists():
        content = path.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=120) as response:
            content = response.read()
    if hashlib.sha256(content).hexdigest() != sha256:
        raise ValueError(f"Model download hash mismatch: {url}; do not substitute another version")
    store(path, content)


def unpack(archive, root):
    root = Path(root).resolve()
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            destination = (root / info.filename).resolve()
            # Confine all resolved archive paths to the requested root.
            if not destination.is_relative_to(root) or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Unsafe archive member: {info.filename}")
            if not info.is_dir():
                store(destination, bundle.read(info))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/external/zahirovic2022")
    args = parser.parse_args()
    root = Path(args.output)
    manifest = {}
    for name, sha256 in ARCHIVES.items():
        url = f"https://repo.gplates.org/webdav/pmm/zahirovic2022/{name}.zip"
        archive = root / f"{name}.zip"
        fetch(url, sha256, archive)
        unpack(archive, root)
        manifest[name] = {"url": url, "sha256": sha256}
    fetch(ROTATION_URL, ROTATION_SHA256, root / "Rotations" / ROTATION)
    manifest["rotation"] = {"url": ROTATION_URL, "sha256": ROTATION_SHA256}
    manifest["frame"] = (
        "GMT original optimised mantle frame; ANCHOR=0; numerical registration required"
    )
    store(root / "gmt_model_manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))
    print(root / "gmt_model_manifest.json")


if __name__ == "__main__":
    main()
