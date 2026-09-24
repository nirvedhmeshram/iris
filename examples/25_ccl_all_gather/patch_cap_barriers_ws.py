#!/usr/bin/env python3
"""Neutralise the TDM named-barrier sequence in a warp-specialized roccap .cap.

The warp_specialized kernel uses two different barrier families:

    s_wait_tensorcnt 0x0000   BFCB0000   TDM completion  -- kept
    s_barrier_join   m0       BE80527D   TDM named barrier -- removed
    s_barrier_signal m0       BE804E7D   TDM named barrier -- removed
    s_barrier_wait   0x0001   BF940001   TDM named barrier -- removed

    s_barrier_signal -1       BE804EC1   warp_specialize partition sync -- KEPT
    s_barrier_wait   0xffff   BF94FFFF   warp_specialize partition sync -- KEPT

Removing the `-1` pair would break the warp_specialize entry/exit protocol, so
only the m0/id-1 family is touched.  Unlike the `hoisted` kernel these three are
not contiguous with the tensor wait, so we cannot match a fixed byte run.
Instead we locate the exact ISA span inside the lz4 blob (via `roccap extract`)
and rewrite only within that span, so a byte pattern that happens to occur in
buffer data cannot be corrupted.  All encodings are 4 bytes, so the patch is
length-preserving and every address in the cap stays valid.

Usage:  patch_cap_barriers_ws.py <in.cap> <out.cap>
"""

from __future__ import annotations

import glob
import io
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import lz4.frame
import xxhash

# little-endian byte order of the 32-bit encodings above
REMOVE = {
    "s_barrier_join m0": bytes.fromhex("7D5280BE"),
    "s_barrier_signal m0": bytes.fromhex("7D4E80BE"),
    "s_barrier_wait 0x1": bytes.fromhex("010094BF"),
}
KEEP = {
    "s_barrier_signal -1": bytes.fromhex("C14E80BE"),
    "s_barrier_wait 0xffff": bytes.fromhex("FFFF94BF"),
}
S_NOP0 = bytes.fromhex("000080BF")


def _lz4_decompress_all(raw: bytes) -> tuple[bytes, int]:
    """Decompress a possibly *concatenated* lz4 frame stream.

    lz4.frame.decompress() stops after the first frame, so on a concatenated
    stream it silently returns a prefix.  roccap writes large buffers as many
    frames, and both the recorded xxHash64 and the packet `size` cover the
    whole stream -- reading only the first frame yields a wrong hash and, if we
    then rewrote the blob, would discard every frame after the first.

    Returns (payload, frame_count).
    """
    out = bytearray()
    frames = 0
    rest = raw
    while rest:
        d = lz4.frame.LZ4FrameDecompressor()
        out += d.decompress(rest)
        frames += 1
        rest = d.unused_data
    return bytes(out), frames


def _update_loaddata_hash(cap_json: bytes, blob_name: str, payload: bytes) -> bytes:
    """Rewrite the LoadPacket hash for `blob_name` to match `payload`.

    roc_capture.json records an xxHash64 (seed 0) of each loaddata packet's
    *decompressed* payload, formatted with %x (no zero padding).  aqltoolkit
    validates it on load and aborts if it disagrees, and --skip-parser-checks
    does not bypass the check -- so rewriting the ISA without refreshing this
    leaves an unloadable cap.

    The edit is textual rather than a json round-trip: the file carries leading
    // comments and a specific layout, and we only want the one field to move.
    Within a LoadPacket the key order is address, size, hash, filename, so the
    hash to fix is the last one before the matching filename.
    """
    text = cap_json.decode("utf-8", errors="surrogateescape")
    m = re.search(r'"filename"\s*:\s*"[^"]*%s"' % re.escape(blob_name), text)
    if not m:
        raise RuntimeError("no LoadPacket references %s" % blob_name)
    hashes = list(re.finditer(r'"hash"\s*:\s*"([0-9a-fA-F]*)"', text[: m.start()]))
    if not hashes:
        raise RuntimeError("no hash field precedes %s" % blob_name)
    h = hashes[-1]
    new = "%x" % xxhash.xxh64(payload, seed=0).intdigest()
    print("    hash %s -> %s" % (h.group(1), new))
    text = text[: h.start(1)] + new + text[h.end(1) :]
    return text.encode("utf-8", errors="surrogateescape")


def extract_isa(cap: Path) -> bytes:
    """Pull the raw kernel ISA out of the cap so we can locate it in the blob."""
    with tempfile.TemporaryDirectory(prefix="ws_isa_") as tmp:
        prefix = Path(tmp) / "out"
        subprocess.run(["roccap", "extract", "--sp3", "0-", "-o", str(prefix), str(cap)],
                       check=True, capture_output=True)
        bins = sorted(glob.glob(str(prefix) + "*.bin"))
        if not bins:
            raise RuntimeError(f"no isa-data.bin from roccap extract on {cap}")
        return Path(bins[0]).read_bytes()


def patch_span(data: bytes, start: int, end: int) -> tuple[bytes, dict[str, int]]:
    buf = bytearray(data)
    counts = {}
    for name, enc in REMOVE.items():
        n = 0
        pos = start
        while True:
            i = buf.find(enc, pos, end)
            if i < 0:
                break
            # instructions are 4-byte aligned within the ISA
            if (i - start) % 4 == 0:
                buf[i:i + 4] = S_NOP0
                n += 1
                pos = i + 4
            else:
                pos = i + 1
        counts[name] = n
    return bytes(buf), counts


def main() -> int:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])

    isa = extract_isa(src)
    sig = isa[:64]
    total = 0

    # Read the whole archive first: roc_capture.json has to be rewritten with
    # the hash of the patched blob, and we cannot rely on it following the
    # loaddata members in the tar.
    members = []
    with tarfile.open(src, "r") as tin:
        for member in tin.getmembers():
            data = tin.extractfile(member).read() if member.isfile() else None
            members.append((member, data))

    patched_blob = None  # (basename, decompressed payload)
    for idx, (member, payload) in enumerate(members):
        if payload is None or not member.name.endswith(".lz4"):
            continue
        try:
            raw, nframes = _lz4_decompress_all(payload)
        except Exception:
            continue
        at = raw.find(sig)
        if at < 0:
            continue
        patched, counts = patch_span(raw, at, at + len(isa))
        n = sum(counts.values())
        if not n:
            continue
        total += n
        print(f"  {member.name}: ISA at 0x{at:X} len {len(isa)}")
        for k, v in counts.items():
            print(f"    removed {v:3d}  {k}")
        for k, v in KEEP.items():
            print(f"    kept    {raw.count(v):3d}  {k}")
        patched_blob = (Path(member.name).name, patched)
        comp = lz4.frame.compress(patched)
        member.size = len(comp)
        members[idx] = (member, comp)
        break

    if total == 0 or patched_blob is None:
        print("ERROR: no TDM barrier encodings found -- cap left unpatched", file=sys.stderr)
        return 1

    # Refresh the LoadPacket hash so aqltoolkit will accept the cap.
    for i, (member, payload) in enumerate(members):
        if member.name.endswith("roc_capture.json"):
            new_json = _update_loaddata_hash(payload, patched_blob[0], patched_blob[1])
            member.size = len(new_json)
            members[i] = (member, new_json)
            break
    else:
        print("ERROR: roc_capture.json not found -- cap would fail hash validation", file=sys.stderr)
        return 1

    with tarfile.open(dst, "w", format=tarfile.GNU_FORMAT) as tout:
        for member, payload in members:
            if payload is None:
                tout.addfile(member)
            else:
                tout.addfile(member, io.BytesIO(payload))

    print(f"Wrote {dst} ({total} instructions neutralised)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
