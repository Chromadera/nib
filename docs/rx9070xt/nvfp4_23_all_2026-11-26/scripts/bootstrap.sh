#!/usr/bin/env bash
set -x
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends git cmake ca-certificates curl aria2 python3 python3-venv python3-pip socat libgomp1 build-essential
cd /workspace/models
Q=https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/resolve/main/Inkling-2-Mini-Q8_0.gguf
P=https://huggingface.co/degenmeowster/Inkling-2-Mini-Nib-Pack/resolve/main
for a in 1 2 3 4; do
aria2c -x 16 -s 16 -k 8M --file-allocation=none --continue=true --max-tries=5 --retry-wait=5 --summary-interval=60 --console-log-level=warn -d /workspace/models -o Inkling-2-Mini-Q8_0.gguf "$Q" && break; done
for f in mtp.safetensors mmproj-Inkling-2-Mini-Q8_0.gguf; do
aria2c -x 16 -s 16 -k 8M --file-allocation=none --continue=true --max-tries=5 --console-log-level=warn -d /workspace/models -o $f "$P/$f"; done
ls -la /workspace/models
echo BOOTSTRAP_DONE
