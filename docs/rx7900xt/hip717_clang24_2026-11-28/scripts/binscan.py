#!/usr/bin/env python3
"""Audit the device code actually embedded in a HIP host binary.

A linked HIP executable keeps one clang offload bundle per translation unit in
`.hip_fatbin`, so `clang-offload-bundler -unbundle` alone only ever returns the
first module.  This splits the section on the bundle magic, unbundles every
module for the requested arch, and prints AMDGPU kernel metadata.

usage: binscan.py <binary> [gfx1100] [name-filter]
"""
import os, re, subprocess, sys, tempfile

ROCM = os.environ.get("ROCM_PATH", "/opt/rocm")
BIN = os.path.join(ROCM, "llvm", "bin")
MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main():
    path = sys.argv[1]
    arch = sys.argv[2] if len(sys.argv) > 2 else "gfx1100"
    flt = sys.argv[3] if len(sys.argv) > 3 else ""

    W = tempfile.mkdtemp(prefix="binscan")
    fat = os.path.join(W, "fat.bin")
    r = run([os.path.join(BIN, "llvm-objcopy"), "--dump-section", f".hip_fatbin={fat}", path])
    if not os.path.exists(fat):
        sys.exit(f"no .hip_fatbin section in {path}: {r.stderr.strip()}")
    data = open(fat, "rb").read()

    chunks, pos = [], 0
    while True:
        i = data.find(MAGIC, pos)
        if i < 0:
            break
        j = data.find(MAGIC, i + len(MAGIC))
        chunks.append(data[i:(j if j >= 0 else len(data))])
        pos = i + len(MAGIC)

    rows, nmod = [], 0
    for n, c in enumerate(chunks):
        p = os.path.join(W, f"b{n}")
        open(p, "wb").write(c)
        out = os.path.join(W, f"d{n}.o")
        u = run([os.path.join(BIN, "clang-offload-bundler"), "-type=o", "-unbundle", f"-input={p}",
                 f"-targets=hip-amdgcn-amd-amdhsa--{arch}", f"-output={out}"])
        if u.returncode or not os.path.exists(out) or os.path.getsize(out) == 0:
            continue
        nmod += 1
        notes = run([os.path.join(BIN, "llvm-readobj"), "--notes", out]).stdout
        for blk in re.split(r"\n  - (?=\.)", notes)[1:]:
            def g(k, d=0):
                m = re.search(rf"\.{k}:\s*(\S+)", blk)
                return m.group(1) if m else d
            name = g("name", "?")
            if name == "?" or ".vgpr_count" not in blk:
                continue
            try:
                rows.append((int(g("vgpr_spill_count")), int(g("sgpr_spill_count")),
                             int(g("private_segment_fixed_size")), int(g("vgpr_count")),
                             int(g("group_segment_fixed_size")), name))
            except ValueError:
                pass
    subprocess.run(["rm", "-rf", W])

    shown = [r for r in rows if not flt or flt in r[5]]
    try:
        import subprocess as sp
        dem = sp.run(["c++filt"], input="\n".join(r[5] for r in shown), capture_output=True, text=True).stdout.splitlines()
    except Exception:
        dem = [r[5] for r in shown]
    print(f"# {path} [{arch}]: {len(chunks)} bundles, {nmod} modules, {len(rows)} kernels, {len(shown)} shown")
    for r, nm in zip(shown, dem):
        mark = "***" if (r[0] or r[1] or r[2]) else "   "
        print(f"{mark} vSpill={r[0]:>4} sSpill={r[1]:>3} priv={r[2]:>5} vGPR={r[3]:>4} LDS={r[4]:>6}  {nm[:120]}")
    bad = [r for r in rows if r[0] or r[1] or r[2]]
    print(f"# TOTAL kernels={len(rows)} spilling={len(bad)} "
          f"vgprSpills={sum(r[0] for r in rows)} sgprSpills={sum(r[1] for r in rows)} "
          f"maxPriv={max((r[2] for r in rows), default=0)}")


main()
