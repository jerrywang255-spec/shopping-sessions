#!/usr/bin/env python3
"""Minimal dependency-free reader for the Parquet files produced by pq.rs.

The writer in pq.rs emits exactly one row group with one uncompressed PLAIN
data page per column, Thrift-compact file metadata, physical types
BYTE_ARRAY (Str), INT64 (I64), INT32 (Date), and optional columns carrying a
single RLE/bit-packed definition-level run (bit width 1). This reader decodes
only that shape. It exists because pyarrow is not installable in the offline
sandbox; where pyarrow is available, prefer tests/smoke.py and
tests/compare_performance.py unchanged.
"""
import datetime as dt
import struct
from pathlib import Path


class Thrift:
    """Compact-protocol reader, just enough for parquet FileMetaData."""

    def __init__(self, buf, pos=0):
        self.b = buf
        self.p = pos

    def u8(self):
        v = self.b[self.p]
        self.p += 1
        return v

    def varint(self):
        shift = 0
        out = 0
        while True:
            x = self.u8()
            out |= (x & 0x7F) << shift
            if not x & 0x80:
                return out
            shift += 7

    def zigzag(self):
        v = self.varint()
        return (v >> 1) ^ -(v & 1)

    def binary(self):
        n = self.varint()
        s = self.b[self.p:self.p + n]
        self.p += n
        return bytes(s)

    def struct_fields(self):
        """Yield (field_id, type_nibble, reader) until the struct STOP byte."""
        last = 0
        while True:
            h = self.u8()
            if h == 0:
                return
            ty = h & 0x0F
            delta = h >> 4
            fid = last + delta if delta else self.zigzag()
            last = fid
            yield fid, ty

    def skip(self, ty):
        if ty in (1, 2):  # bool true/false
            return
        if ty in (5, 6):  # i32 / i64
            self.zigzag()
        elif ty == 8:  # binary
            self.binary()
        elif ty in (9, 10):  # list / set
            h = self.u8()
            n = h >> 4
            ety = h & 0x0F
            if n == 15:
                n = self.varint()
            for _ in range(n):
                self.skip(ety)
        elif ty == 12:  # struct
            for _, t in self.struct_fields():
                self.skip(t)
        elif ty in (3, 4):  # byte / double
            self.p += 1 if ty == 3 else 8
        else:
            raise ValueError(f"unhandled thrift type {ty}")

    def list_header(self):
        h = self.u8()
        n = h >> 4
        ety = h & 0x0F
        if n == 15:
            n = self.varint()
        return n, ety


def _read_columns_meta(th):
    """Parse FileMetaData, return (num_rows, schema_names, [(name, phys, conv, file_off, size)])."""
    num_rows = 0
    schema = []  # (name, phys_type, converted_type, optional)
    rowgroups = []
    for fid, ty in th.struct_fields():
        if fid == 1:  # version
            th.zigzag()
        elif fid == 2:  # schema list
            n, _ = th.list_header()
            for _ in range(n):
                name = None
                phys = None
                conv = None
                rep = 0
                for sfid, sty in th.struct_fields():
                    if sfid == 1:
                        phys = th.zigzag()
                    elif sfid == 3:
                        rep = th.zigzag()
                    elif sfid == 4:
                        name = th.binary().decode()
                    elif sfid == 6:
                        conv = th.zigzag()
                    else:
                        th.skip(sty)
                schema.append((name, phys, conv, rep))
        elif fid == 3:  # num_rows
            num_rows = th.zigzag()
        elif fid == 4:  # row groups
            n, _ = th.list_header()
            for _ in range(n):
                rowgroups.append(_read_row_group(th))
        else:
            th.skip(ty)
    return num_rows, schema, rowgroups


def _read_row_group(th):
    cols = []
    for fid, ty in th.struct_fields():
        if fid == 1:  # columns list
            n, _ = th.list_header()
            for _ in range(n):
                cols.append(_read_column_chunk(th))
        else:
            th.skip(ty)
    return cols


def _read_column_chunk(th):
    off = None
    for fid, ty in th.struct_fields():
        if fid == 2:  # file_offset
            off = th.zigzag()
        elif fid == 3:  # ColumnMetaData struct
            for mfid, mty in th.struct_fields():
                if mfid == 9:  # data_page_offset
                    off = th.zigzag()
                else:
                    th.skip(mty)
        else:
            th.skip(ty)
    return off


def _read_page(buf, off, num_rows, phys, optional):
    th = Thrift(buf, off)
    page_type = None
    uncompressed = None
    num_values = None
    for fid, ty in th.struct_fields():
        if fid == 1:
            page_type = th.zigzag()
        elif fid == 2:
            uncompressed = th.zigzag()
        elif fid == 3:
            th.zigzag()  # compressed size
        elif fid == 5:  # DataPageHeader
            for dfid, dty in th.struct_fields():
                if dfid == 1:
                    num_values = th.zigzag()
                else:
                    th.skip(dty)
        else:
            th.skip(ty)
    data = buf[th.p:th.p + uncompressed]
    return _decode_values(data, num_values, phys, optional)


def _decode_values(data, n, phys, optional):
    pos = 0
    defs = [True] * n
    if optional:
        rle_len = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        rle = data[pos:pos + rle_len]
        pos += rle_len
        defs = _decode_bool_rle(rle, n)
    values = []
    present = sum(1 for d in defs if d)
    raw = []
    if phys == 6:  # BYTE_ARRAY
        for _ in range(present):
            ln = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            raw.append(bytes(data[pos:pos + ln]))
            pos += ln
    elif phys == 2:  # INT64
        for _ in range(present):
            raw.append(struct.unpack_from("<q", data, pos)[0])
            pos += 8
    elif phys == 1:  # INT32 (DATE)
        for _ in range(present):
            raw.append(struct.unpack_from("<i", data, pos)[0])
            pos += 4
    else:
        raise ValueError(f"unhandled physical type {phys}")
    it = iter(raw)
    for d in defs:
        values.append(next(it) if d else None)
    return values


def _decode_bool_rle(rle, n):
    """Decode a bit-width-1 RLE/bit-packed hybrid definition-level stream."""
    out = []
    p = 0
    while len(out) < n and p < len(rle):
        header = 0
        shift = 0
        while True:
            x = rle[p]
            p += 1
            header |= (x & 0x7F) << shift
            if not x & 0x80:
                break
            shift += 7
        if header & 1:  # bit-packed run
            groups = header >> 1
            for _ in range(groups):
                b = rle[p]
                p += 1
                for j in range(8):
                    if len(out) < n:
                        out.append(bool(b >> j & 1))
        else:  # RLE run
            count = header >> 1
            val = rle[p]
            p += 1
            out.extend([bool(val)] * count)
    while len(out) < n:
        out.append(False)
    return out[:n]


def read_table(path):
    """Return (list-of-dict rows, schema-field-name->converted_type)."""
    buf = Path(path).read_bytes()
    assert buf[:4] == b"PAR1" and buf[-4:] == b"PAR1", "not a parquet file"
    footer_len = struct.unpack_from("<I", buf, len(buf) - 8)[0]
    meta_start = len(buf) - 8 - footer_len
    th = Thrift(buf, meta_start)
    num_rows, schema, rowgroups = _read_columns_meta(th)
    data_cols = [s for s in schema if s[1] is not None]
    rows = [dict() for _ in range(num_rows)]
    conv = {}
    if num_rows and rowgroups:
        offsets = rowgroups[0]
        for (name, phys, cv, rep), off in zip(data_cols, offsets):
            conv[name] = cv
            optional = rep == 1
            vals = _read_page(buf, off, num_rows, phys, optional)
            for i, v in enumerate(vals):
                if phys == 1 and cv == 6 and v is not None:  # INT32 DATE -> date
                    v = dt.date(1970, 1, 1) + dt.timedelta(days=v)
                elif phys == 6 and v is not None:
                    v = v.decode()
                rows[i][name] = v
    return rows, conv


def glob_table(folder, name):
    files = sorted(Path(folder, name).glob("*.parquet"))
    assert files, f"no parquet in {name}"
    out = []
    conv = {}
    for f in files:
        rows, conv = read_table(f)
        out.extend(rows)
    return out, conv


if __name__ == "__main__":
    import sys
    rows, conv = read_table(sys.argv[1])
    print("converted_types:", conv)
    for r in rows:
        print(r)
