#!/usr/bin/env bash
# Build the environment the TiSR worker runs in (src/srbf/worker/models/tisr_worker.py, docs/models.md#tisr).
#
#   scripts/envs/build_tisr_env.sh <prefix>          e.g. scripts/envs/build_tisr_env.sh "$FLASH_ANSR_ROOT/envs/tisr"
#
# What it builds, all inside <prefix> (nothing outside it is written except uv's cache):
#
#   <prefix>/bin/python          a Python 3.12 virtual environment with juliacall and numpy (tisr/requirements.txt)
#   <prefix>/julia-1.11.5/       the official Julia 1.11.5 binaries, checked against their published SHA-256
#   <prefix>/julia_env/          the Julia project: tisr/Project.toml and tisr/Manifest.toml
#   <prefix>/julia_depot/        the Julia depot (packages, compiled caches) of this environment alone
#
# The Julia project pins TiSR (github.com/scoop-group/TiSR) at commit 5b541b3, the commit the FastSRB paper
# (arXiv 2508.14481) ran, and every dependency at the version that paper's lock file (Manifest.toml of
# github.com/viktmar/FastSRB-paper-repo, Julia 1.11.5) recorded; the only additions are PythonCall and its own
# dependencies, which juliacall needs. The worker finds this layout from its interpreter and points juliacall at it
# (offline, one thread), so no shared depot or Julia installation is touched.
#
# Needs uv (https://docs.astral.sh/uv/; UV=... to choose), curl, tar, sha256sum and git (Pkg clones TiSR).
set -euo pipefail

PREFIX=${1:?usage: build_tisr_env.sh <prefix>}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
UV=${UV:-$(command -v uv || true)}
PYTHON_VERSION=3.12.13
JULIA_VERSION=1.11.5
JULIA_URL=https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.5-linux-x86_64.tar.gz
JULIA_SHA256=723e878c642220cc0251a0e13758c059a389cadc7f01376feaf1ea7388fe8f9c
TISR_COMMIT=5b541b390de153def44c548bf4842cc956db672f
TISR_TREE=4406e769ba6f6dd2d70f48809e8a1aa13a9073f0   # the git tree of TISR_COMMIT, which the Manifest pins

if [ -e "$PREFIX" ]; then
    echo "$PREFIX exists; remove it first" >&2
    exit 1
fi
if [ -z "$UV" ]; then
    echo "uv is not on PATH; install it or set UV" >&2
    exit 1
fi
grep -q "git-tree-sha1 = \"$TISR_TREE\"" "$HERE/tisr/Manifest.toml"

echo ">>> Python $PYTHON_VERSION environment"
"$UV" venv --quiet --python "$PYTHON_VERSION" "$PREFIX"
"$UV" pip install --quiet --python "$PREFIX/bin/python" -r "$HERE/tisr/requirements.txt"

echo ">>> Julia $JULIA_VERSION"
TARBALL="$PREFIX/julia-$JULIA_VERSION-linux-x86_64.tar.gz"
curl --fail --silent --show-error --location -o "$TARBALL" "$JULIA_URL"
echo "$JULIA_SHA256  $TARBALL" | sha256sum --check --quiet
tar -xzf "$TARBALL" -C "$PREFIX"
rm "$TARBALL"
JULIA="$PREFIX/julia-$JULIA_VERSION/bin/julia"
test "$("$JULIA" --startup-file=no -e 'print(VERSION)')" = "$JULIA_VERSION"

echo ">>> the Julia project (TiSR $TISR_COMMIT) in a depot of its own"
mkdir -p "$PREFIX/julia_env" "$PREFIX/julia_depot"
cp "$HERE/tisr/Project.toml" "$HERE/tisr/Manifest.toml" "$PREFIX/julia_env/"
export JULIA_DEPOT_PATH="$PREFIX/julia_depot"
export JULIA_PROJECT="$PREFIX/julia_env"
export JULIA_CONDAPKG_BACKEND=Null
"$JULIA" --startup-file=no -e '
    using Pkg
    Pkg.instantiate()                       # exactly the Manifest; Pkg checks every git tree hash
    Pkg.precompile()
    info = Pkg.dependencies()[Base.UUID("e1088136-a916-4db6-8e31-3a049800401f")]
    println("TiSR tree ", info.tree_hash)
    @assert string(info.tree_hash) == ARGS[1]
' "$TISR_TREE"

echo ">>> smoke fit through juliacall"
cd "$PREFIX"
"$PREFIX/bin/python" - "$HERE/../../src/srbf/worker/models/tisr_worker.py" <<'EOF'
import importlib.util
import sys

import numpy as np

spec = importlib.util.spec_from_file_location("tisr_worker", sys.argv[1])
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
state = worker.load({"generations": 30, "warmup": False})
rng = np.random.default_rng(0)
X = rng.uniform(1.0, 5.0, size=(200, 2))
y = 2.5 * X[:, 0] * X[:, 1] + 1.0
result = worker.fit(X.tolist(), y.tolist(), x_val=[], variables=["a", "b"], meta={}, options={}, state=state)
print("info:", worker.info(state))
print("expression:", result["expression"], "| deviation:", result["extra"]["string_deviation"])
assert result["extra"]["string_deviation"] is not None and result["extra"]["string_deviation"] < 1e-9
EOF

echo "$TISR_COMMIT" > "$PREFIX/tisr-commit.txt"
"$UV" pip freeze --python "$PREFIX/bin/python" > "$PREFIX/pip-freeze.txt"
echo ">>> done: $PREFIX/bin/python runs TiSR $TISR_COMMIT on Julia $JULIA_VERSION"
