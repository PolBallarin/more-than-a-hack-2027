"""Fetch Metrica Sports Sample_Game_3 at a pinned commit into the data cache.

The files are public (github.com/metrica-sports/sample-data) but they are never committed here:
they land in $MTAH_DATA_DIR/cache/, outside the repository, like every other cache. Pinning the
commit makes the validation set byte-identical on every machine; the SHA-256 of each file is
checked after download, so a silent upstream change or a truncated transfer is an error, not a
different score.
"""

import hashlib
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from games import DATA_DIR  # noqa: E402

# Params
REPO = "metrica-sports/sample-data"
COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"  # master on 2021-04-15, the last commit
GAME = "Sample_Game_3"
# file -> (size in bytes, sha256). Sizes come from the GitHub tree at COMMIT; hashes were taken
# on first download and are what makes the set reproducible.
FILES = {
    "Sample_Game_3_tracking.txt": (53443115, "570e6ce82c2d93bc92723af331d93c31110bec1d6dc0be07c6d44ca3f981d658"),
    "Sample_Game_3_events.json": (1709486, "28a53bfe9eaf32535362335553d517cdce0eabb588af6166009081753751f4e0"),
    "Sample_Game_3_metadata.xml": (81806, "009c7dcfcb66192014917ca1559c158f612353e390eafb85f8214b06a1c7dec1"),
}


def cache_dir(data_dir=None):
    """Folder holding the pinned Sample_Game_3 files."""
    return os.path.join(data_dir or DATA_DIR, "cache", "metrica_sample_data", COMMIT[:7], GAME)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _check(path, size, sha):
    if os.path.getsize(path) != size:
        return "size {} != {}".format(os.path.getsize(path), size)
    if sha is not None and _sha256(path) != sha:
        return "sha256 mismatch"
    return None


def fetch(data_dir=None, quiet=False):
    """Paths {file name: path} of the pinned files, downloading whichever are missing."""
    folder = cache_dir(data_dir)
    os.makedirs(folder, exist_ok=True)
    paths = {}
    for name, (size, sha) in FILES.items():
        path = os.path.join(folder, name)
        if not os.path.isfile(path) or _check(path, size, sha):
            url = "https://raw.githubusercontent.com/{}/{}/data/{}/{}".format(
                REPO, COMMIT, GAME, name)
            if not quiet:
                print("downloading {} ({:.1f} MB)".format(name, size / 1e6))
            partial = path + ".part"
            urllib.request.urlretrieve(url, partial)
            problem = _check(partial, size, sha)
            if problem:
                os.remove(partial)
                raise IOError("{}: {} -- refusing to use it".format(url, problem))
            os.replace(partial, path)
        paths[name] = path
    return paths


if __name__ == "__main__":
    for name, path in fetch().items():
        print("{}  {}".format(_sha256(path), path))
