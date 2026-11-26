cd /workspace/models; HF=https://huggingface.co/degenmeowster/Inkling-2-Mini-Nib-Pack/resolve/main; f=Inkling-2-Mini-NVFP4-Mixed.gguf
aria2c -x16 -s16 -k8M --file-allocation=none -c --console-log-level=warn --summary-interval=0 -o $f $HF/$f
sha=$(curl -sIL $HF/$f | awk 'tolower($1)=="x-linked-etag:"{gsub("[\r\"]","",$2);v=$2} END{print v}')
got=$(sha256sum $f | cut -d' ' -f1); echo "want=$sha got=$got size=$(stat -c%s $f)"
