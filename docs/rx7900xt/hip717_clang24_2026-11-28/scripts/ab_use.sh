#!/bin/bash
# A/B switch for the nib server binary.  Swaps /opt/nib/bin/nib between the
# zero-spill build (nib.fixed) and the pre-fix build (nib.pre-fix).
# The change only takes effect after the service is restarted:
#     sudo systemctl restart load_gpu0
set -euo pipefail

which_variant=${1:?usage: ab_use.sh fixed|pre-fix|status}
BIN=/opt/nib/bin/nib

case "$which_variant" in
  fixed)   src=/opt/nib/bin/nib.fixed ;;
  pre-fix) src=/opt/nib/bin/nib.pre-fix ;;
  status)
    md5sum "$BIN" /opt/nib/bin/nib.fixed /opt/nib/bin/nib.pre-fix
    exit 0 ;;
  *) echo "usage: ab_use.sh fixed|pre-fix|status"; exit 1 ;;
esac

if cmp -s "$src" "$BIN"; then
  echo "already running the $which_variant build; no restart needed"
  exit 0
fi

cmp -s "$BIN" /opt/nib/bin/nib.fixed && cp "$BIN" /opt/nib/bin/nib.keep-fixed
cmp -s "$BIN" /opt/nib/bin/nib.pre-fix && cp "$BIN" /opt/nib/bin/nib.keep-pre-fix
install -m 0775 "$src" "$BIN"
echo "installed: $which_variant  (restart load_gpu0 to pick it up)"
md5sum "$BIN"
