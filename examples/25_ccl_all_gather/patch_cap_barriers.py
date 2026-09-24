#!/usr/bin/env python3
"""Rewrite the TDM barrier pairs in a roccap .cap to s_nop, in place.

The cap is a POSIX tar of lz4-frame blobs plus roc_capture.json.  The kernel
ISA lives inside one of those blobs.  We locate the exact instruction triple

    s_wait_tensorcnt 0x0   BFCB0000
    s_barrier_signal -1    BE804EC1
    s_barrier_wait   0xffff BF94FFFF

and overwrite only the two barrier words with s_nop 0 (BF800000), keeping the
tensor wait -- which is the real TDM completion semantic -- untouched.  All
three are 4-byte SOPP/SOP1 encodings, so the patch is length-preserving: the
ISA blob size, every buffer address, and the dispatch packet all stay valid.

Usage:  patch_cap_barriers.py <in.cap> <out.cap>
"""

from __future__ import annotations

import io
import re
import sys
import tarfile
from pathlib import Path

import lz4.frame
import xxhash

S_WAIT_TENSORCNT = bytes.fromhex("0000CBBF")  # little-endian BFCB0000
S_BARRIER_SIGNAL = bytes.fromhex("C14E80BE")  # little-endian BE804EC1
S_BARRIER_WAIT = bytes.fromhex("FFFF94BF")  # little-endian BF94FFFF
S_NOP0 = bytes.fromhex("000080BF")  # little-endian BF800000

PATTERN = S_WAIT_TENSORCNT + S_BARRIER_SIGNAL + S_BARRIER_WAIT
REPLACEMENT = S_WAIT_TENSORCNT + S_NOP0 + S_NOP0


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


def patch_blob(data: bytes) -> tuple[bytes, int]:
    """Replace every wait+barrier+barrier triple; return (new_data, count)."""
    count = data.count(PATTERN)
    if count == 0:
        return data, 0
    return data.replace(PATTERN, REPLACEMENT), count


def main() -> int:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])

    total = 0
    # Read the archive up front: roc_capture.json must be rewritten with the
    # hash of the patched blob and may precede it in the tar.
    members = []
    with tarfile.open(src, "r") as tin:
        for member in tin.getmembers():
            data = tin.extractfile(member).read() if member.isfile() else None
            members.append((member, data))

    patched_blob = None
    for i, (member, payload) in enumerate(members):
        if payload is None or not member.name.endswith(".lz4"):
            continue
        try:
            raw, nframes = _lz4_decompress_all(payload)
        except Exception:
            continue
        patched, n = patch_blob(raw)
        if not n:
            continue
        total += n
        print(f"  patched {n} barrier pair(s) in {member.name}")
        comp = lz4.frame.compress(patched)
        member.size = len(comp)
        members[i] = (member, comp)
        patched_blob = (Path(member.name).name, patched)
        break

    if total == 0 or patched_blob is None:
        print("ERROR: no barrier pattern found -- cap left unpatched", file=sys.stderr)
        return 1

    for i, (member, payload) in enumerate(members):
        if member.name.endswith("roc_capture.json"):
            new_json = _update_loaddata_hash(payload, patched_blob[0], patched_blob[1])
            member.size = len(new_json)
            members[i] = (member, new_json)
            break
    else:
        print("ERROR: roc_capture.json not found", file=sys.stderr)
        return 1

    with tarfile.open(dst, "w", format=tarfile.GNU_FORMAT) as tout:
        for member, payload in members:
            tout.addfile(member) if payload is None else tout.addfile(member, io.BytesIO(payload))

    print(f"Wrote {dst} ({total} barrier pair(s) neutralised)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
