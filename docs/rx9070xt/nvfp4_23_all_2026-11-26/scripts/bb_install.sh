#!/usr/bin/env bash
set -x
python3 -m venv /workspace/bbvenv
/workspace/bbvenv/bin/pip install -q --upgrade pip
/workspace/bbvenv/bin/pip install "git+https://github.com/GGZ14/BetterBench.git@f72bb9011a830fd98e82ab89d01e549efa76e9c5" "numpy>=1.24"
/workspace/bbvenv/bin/betterbench --version
/workspace/bbvenv/bin/betterbench run --help
echo BB_DONE
