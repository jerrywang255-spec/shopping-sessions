//! Minimal Parquet writer: one row group, one uncompressed PLAIN data page per column.

pub enum Kind {
    Str,       // BYTE_ARRAY, UTF8/STRING
    I64,       // INT64
    Date,      // INT32, DATE
}

pub struct Column {
    pub name: &'static str,
    pub kind: Kind,
    pub optional: bool,
    /// Encoded page body (definition levels, if optional, followed by PLAIN values).
    pub data: Vec<u8>,
}

/// Builders for column page bodies.
pub struct StrCol {
    pub buf: Vec<u8>,
}
impl StrCol {
    pub fn with_capacity(n: usize) -> Self {
        StrCol { buf: Vec::with_capacity(n) }
    }
    #[inline]
    pub fn push(&mut self, s: &[u8]) {
        self.buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
        self.buf.extend_from_slice(s);
    }
}

pub struct OptStrCol {
    pub defs: Vec<bool>,
    pub vals: Vec<u8>,
}
impl OptStrCol {
    pub fn with_capacity(n: usize) -> Self {
        OptStrCol { defs: Vec::with_capacity(n), vals: Vec::with_capacity(n * 8) }
    }
    #[inline]
    pub fn push(&mut self, s: Option<&[u8]>) {
        match s {
            Some(s) => {
                self.defs.push(true);
                self.vals.extend_from_slice(&(s.len() as u32).to_le_bytes());
                self.vals.extend_from_slice(s);
            }
            None => self.defs.push(false),
        }
    }
    pub fn finish(self) -> Vec<u8> {
        // RLE/bit-packed hybrid, bit width 1, as bit-packed groups of 8.
        let n = self.defs.len();
        let groups = (n + 7) / 8;
        let mut rle = Vec::with_capacity(groups + 10);
        if groups > 0 {
            varint(&mut rle, ((groups as u64) << 1) | 1);
            for g in 0..groups {
                let mut b = 0u8;
                for j in 0..8 {
                    let i = g * 8 + j;
                    if i < n && self.defs[i] {
                        b |= 1 << j;
                    }
                }
                rle.push(b);
            }
        }
        let mut out = Vec::with_capacity(4 + rle.len() + self.vals.len());
        out.extend_from_slice(&(rle.len() as u32).to_le_bytes());
        out.extend_from_slice(&rle);
        out.extend_from_slice(&self.vals);
        out
    }
}

pub fn i64_col(v: &[i64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 8);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}
pub fn i32_col(v: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Thrift compact protocol writer.
struct Tw {
    buf: Vec<u8>,
    last: i16,
    stack: Vec<i16>,
}
const T_I32: u8 = 5;
const T_I64: u8 = 6;
const T_BIN: u8 = 8;
const T_LIST: u8 = 9;
const T_STRUCT: u8 = 12;
impl Tw {
    fn new() -> Self {
        Tw { buf: Vec::new(), last: 0, stack: Vec::new() }
    }
    fn fh(&mut self, id: i16, ty: u8) {
        let d = id - self.last;
        if d > 0 && d <= 15 {
            self.buf.push(((d as u8) << 4) | ty);
        } else {
            self.buf.push(ty);
            let z = ((id as i32) << 1) ^ ((id as i32) >> 31);
            varint(&mut self.buf, z as u32 as u64);
        }
        self.last = id;
    }
    fn i32f(&mut self, id: i16, v: i32) {
        self.fh(id, T_I32);
        let z = (v << 1) ^ (v >> 31);
        varint(&mut self.buf, z as u32 as u64);
    }
    fn i64f(&mut self, id: i16, v: i64) {
        self.fh(id, T_I64);
        let z = (v << 1) ^ (v >> 63);
        varint(&mut self.buf, z as u64);
    }
    fn binf(&mut self, id: i16, b: &[u8]) {
        self.fh(id, T_BIN);
        varint(&mut self.buf, b.len() as u64);
        self.buf.extend_from_slice(b);
    }
    fn sbegin(&mut self, id: i16) {
        self.fh(id, T_STRUCT);
        self.stack.push(self.last);
        self.last = 0;
    }
    fn send(&mut self) {
        self.buf.push(0);
        self.last = self.stack.pop().unwrap_or(0);
    }
    fn lbegin(&mut self, id: i16, ety: u8, n: usize) {
        self.fh(id, T_LIST);
        if n < 15 {
            self.buf.push(((n as u8) << 4) | ety);
        } else {
            self.buf.push(0xF0 | ety);
            varint(&mut self.buf, n as u64);
        }
    }
    fn elem_sbegin(&mut self) {
        self.stack.push(self.last);
        self.last = 0;
    }
    fn elem_i32(&mut self, v: i32) {
        let z = (v << 1) ^ (v >> 31);
        varint(&mut self.buf, z as u32 as u64);
    }
    fn elem_bin(&mut self, b: &[u8]) {
        varint(&mut self.buf, b.len() as u64);
        self.buf.extend_from_slice(b);
    }
}

fn phys(k: &Kind) -> i32 {
    match k {
        Kind::Str => 6,
        Kind::I64 => 2,
        Kind::Date => 1,
    }
}

/// Write a parquet file holding `rows` rows in the given columns.
/// Column bodies go straight from their buffers to the file, without assembling a second copy.
pub fn write_file(path: &std::path::Path, rows: usize, cols: Vec<Column>) -> std::io::Result<()> {
    use std::io::Write;
    let mut heads: Vec<Vec<u8>> = Vec::with_capacity(cols.len());
    let mut metas = Vec::new(); // (offset, total size)
    let mut end = 4; // after the leading "PAR1"
    if rows > 0 {
        for c in &cols {
            let mut h = Tw::new();
            h.i32f(1, 0); // DATA_PAGE
            h.i32f(2, c.data.len() as i32);
            h.i32f(3, c.data.len() as i32);
            h.sbegin(5);
            h.i32f(1, rows as i32);
            h.i32f(2, 0); // PLAIN
            h.i32f(3, 3); // RLE
            h.i32f(4, 3); // RLE
            h.send();
            h.buf.push(0);
            let size = h.buf.len() + c.data.len();
            metas.push((end, size));
            end += size;
            heads.push(h.buf);
        }
    }
    let mut m = Tw::new();
    m.i32f(1, 1);
    m.lbegin(2, T_STRUCT, cols.len() + 1);
    // root
    m.elem_sbegin();
    m.binf(4, b"schema");
    m.i32f(5, cols.len() as i32);
    m.send();
    for c in &cols {
        m.elem_sbegin();
        m.i32f(1, phys(&c.kind));
        m.i32f(3, if c.optional { 1 } else { 0 });
        m.binf(4, c.name.as_bytes());
        match c.kind {
            Kind::Str => {
                m.i32f(6, 0); // UTF8
                m.sbegin(10);
                m.sbegin(1); // STRING
                m.send();
                m.send();
            }
            Kind::Date => {
                m.i32f(6, 6); // DATE
                m.sbegin(10);
                m.sbegin(6); // DATE
                m.send();
                m.send();
            }
            Kind::I64 => {}
        }
        m.send();
    }
    m.i64f(3, rows as i64);
    if rows > 0 {
        m.lbegin(4, T_STRUCT, 1);
        m.elem_sbegin();
        m.lbegin(1, T_STRUCT, cols.len());
        let mut total = 0i64;
        for (c, &(off, size)) in cols.iter().zip(metas.iter()) {
            total += size as i64;
            m.elem_sbegin();
            m.i64f(2, off as i64);
            m.sbegin(3);
            m.i32f(1, phys(&c.kind));
            if c.optional {
                m.lbegin(2, T_I32, 2);
                m.elem_i32(0);
                m.elem_i32(3);
            } else {
                m.lbegin(2, T_I32, 1);
                m.elem_i32(0);
            }
            m.lbegin(3, T_BIN, 1);
            m.elem_bin(c.name.as_bytes());
            m.i32f(4, 0); // UNCOMPRESSED
            m.i64f(5, rows as i64);
            m.i64f(6, size as i64);
            m.i64f(7, size as i64);
            m.i64f(9, off as i64);
            m.send();
            m.send();
        }
        m.i64f(2, total);
        m.i64f(3, rows as i64);
        m.send();
    } else {
        m.lbegin(4, T_STRUCT, 0);
    }
    m.binf(6, b"ss");
    m.buf.push(0);
    let mut f = std::fs::File::create(path)?;
    f.write_all(b"PAR1")?;
    for (h, c) in heads.iter().zip(&cols) {
        f.write_all(h)?;
        f.write_all(&c.data)?;
    }
    f.write_all(&m.buf)?;
    f.write_all(&(m.buf.len() as u32).to_le_bytes())?;
    f.write_all(b"PAR1")
}

// ------------------------------------------------------------------ streaming writer
/// Physical/logical type of a streamed column.
#[derive(Clone, Copy)]
pub enum SKind {
    Str,    // BYTE_ARRAY UTF8
    OptStr, // optional BYTE_ARRAY UTF8
    I32,    // INT32
    I64,    // INT64
    Date,   // INT32 DATE
}
fn sphys(k: SKind) -> i32 {
    match k {
        SKind::Str | SKind::OptStr => 6,
        SKind::I32 | SKind::Date => 1,
        SKind::I64 => 2,
    }
}

/// Writes a Parquet file row group by row group from small reusable column buffers.
pub struct Stream {
    file: std::io::BufWriter<std::fs::File>,
    off: u64,
    names: Vec<&'static str>,
    kinds: Vec<SKind>,
    pub vals: Vec<Vec<u8>>, // PLAIN values per column
    pub defs: Vec<u8>,      // definition bits of the (single) optional column, packed LSB first
    pub nnull_rows: usize,  // rows in the current group
    groups: Vec<(usize, Vec<(u64, u64)>)>, // (rows, per column (offset, size))
    total_rows: usize,
}
impl Stream {
    pub fn create(path: &std::path::Path, cols: &[(&'static str, SKind)]) -> std::io::Result<Self> {
        let f = std::fs::File::create(path)?;
        let mut file = std::io::BufWriter::with_capacity(1 << 20, f);
        use std::io::Write;
        file.write_all(b"PAR1")?;
        Ok(Stream {
            file,
            off: 4,
            names: cols.iter().map(|c| c.0).collect(),
            kinds: cols.iter().map(|c| c.1).collect(),
            vals: cols.iter().map(|_| Vec::with_capacity(1 << 20)).collect(),
            defs: Vec::with_capacity(1 << 14),
            nnull_rows: 0,
            groups: Vec::new(),
            total_rows: 0,
        })
    }
    #[inline(always)]
    pub fn str(&mut self, c: usize, s: &[u8]) {
        let v = &mut self.vals[c];
        v.extend_from_slice(&(s.len() as u32).to_le_bytes());
        v.extend_from_slice(s);
    }
    #[inline(always)]
    pub fn opt_str(&mut self, c: usize, s: Option<&[u8]>) {
        let i = self.nnull_rows;
        if i & 7 == 0 {
            self.defs.push(0);
        }
        if let Some(s) = s {
            *self.defs.last_mut().unwrap() |= 1 << (i & 7);
            self.str(c, s);
        }
    }
    #[inline(always)]
    pub fn i32(&mut self, c: usize, x: i32) {
        self.vals[c].extend_from_slice(&x.to_le_bytes());
    }
    #[inline(always)]
    pub fn i64(&mut self, c: usize, x: i64) {
        self.vals[c].extend_from_slice(&x.to_le_bytes());
    }
    /// Call after all columns of a row are pushed.
    #[inline(always)]
    pub fn end_row(&mut self) -> std::io::Result<()> {
        self.nnull_rows += 1;
        if self.nnull_rows == 65536 {
            self.flush()?;
        }
        Ok(())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        use std::io::Write;
        let rows = self.nnull_rows;
        if rows == 0 {
            return Ok(());
        }
        let mut metas = Vec::with_capacity(self.kinds.len());
        for c in 0..self.kinds.len() {
            let mut prefix: Vec<u8> = Vec::new();
            if let SKind::OptStr = self.kinds[c] {
                // RLE/bit-packed hybrid, bit width 1: one bit-packed run of ceil(rows/8) groups
                let groups = (rows + 7) / 8;
                let mut rle = Vec::with_capacity(groups + 10);
                varint(&mut rle, ((groups as u64) << 1) | 1);
                rle.extend_from_slice(&self.defs[..groups]);
                prefix.extend_from_slice(&(rle.len() as u32).to_le_bytes());
                prefix.extend_from_slice(&rle);
            }
            let body = prefix.len() + self.vals[c].len();
            let mut h = Tw::new();
            h.i32f(1, 0); // DATA_PAGE
            h.i32f(2, body as i32);
            h.i32f(3, body as i32);
            h.sbegin(5);
            h.i32f(1, rows as i32);
            h.i32f(2, 0); // PLAIN
            h.i32f(3, 3); // RLE
            h.i32f(4, 3); // RLE
            h.send();
            h.buf.push(0);
            let start = self.off;
            self.file.write_all(&h.buf)?;
            self.file.write_all(&prefix)?;
            self.file.write_all(&self.vals[c])?;
            self.off += (h.buf.len() + body) as u64;
            metas.push((start, self.off - start));
            self.vals[c].clear();
        }
        self.defs.clear();
        self.groups.push((rows, metas));
        self.total_rows += rows;
        self.nnull_rows = 0;
        Ok(())
    }
    pub fn finish(mut self) -> std::io::Result<()> {
        use std::io::Write;
        self.flush()?;
        let mut m = Tw::new();
        m.i32f(1, 1);
        m.lbegin(2, T_STRUCT, self.kinds.len() + 1);
        m.elem_sbegin();
        m.binf(4, b"schema");
        m.i32f(5, self.kinds.len() as i32);
        m.send();
        for c in 0..self.kinds.len() {
            let k = self.kinds[c];
            m.elem_sbegin();
            m.i32f(1, sphys(k));
            m.i32f(3, if let SKind::OptStr = k { 1 } else { 0 });
            m.binf(4, self.names[c].as_bytes());
            match k {
                SKind::Str | SKind::OptStr => {
                    m.i32f(6, 0); // UTF8
                    m.sbegin(10);
                    m.sbegin(1); // STRING
                    m.send();
                    m.send();
                }
                SKind::Date => {
                    m.i32f(6, 6); // DATE
                    m.sbegin(10);
                    m.sbegin(6); // DATE
                    m.send();
                    m.send();
                }
                SKind::I32 | SKind::I64 => {}
            }
            m.send();
        }
        m.i64f(3, self.total_rows as i64);
        m.lbegin(4, T_STRUCT, self.groups.len());
        for (rows, metas) in &self.groups {
            m.elem_sbegin();
            m.lbegin(1, T_STRUCT, self.kinds.len());
            let mut total = 0i64;
            for c in 0..self.kinds.len() {
                let (off, size) = metas[c];
                let k = self.kinds[c];
                total += size as i64;
                m.elem_sbegin();
                m.i64f(2, off as i64);
                m.sbegin(3);
                m.i32f(1, sphys(k));
                if let SKind::OptStr = k {
                    m.lbegin(2, T_I32, 2);
                    m.elem_i32(0);
                    m.elem_i32(3);
                } else {
                    m.lbegin(2, T_I32, 1);
                    m.elem_i32(0);
                }
                m.lbegin(3, T_BIN, 1);
                m.elem_bin(self.names[c].as_bytes());
                m.i32f(4, 0); // UNCOMPRESSED
                m.i64f(5, *rows as i64);
                m.i64f(6, size as i64);
                m.i64f(7, size as i64);
                m.i64f(9, off as i64);
                m.send();
                m.send();
            }
            m.i64f(2, total);
            m.i64f(3, *rows as i64);
            m.send();
        }
        m.binf(6, b"ss");
        m.buf.push(0);
        self.file.write_all(&m.buf)?;
        self.file.write_all(&(m.buf.len() as u32).to_le_bytes())?;
        self.file.write_all(b"PAR1")?;
        self.file.flush()
    }
}
