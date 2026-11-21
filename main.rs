//! Shopping Sessions pipeline: std-only Rust, multi-threaded, hand-written CSV and Parquet.
mod pq;

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering as AO};
use std::sync::Mutex;

const DAY_MS: i64 = 86_400_000;
const WEEK_MS: i64 = 7 * DAY_MS;
const GAP_MS: u32 = 1_800_000;
const EXOTIC: u64 = 1 << 63;

// ------------------------------------------------------------------ files
extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut u8;
    fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
}

fn map_file(path: &Path) -> &'static [u8] {
    let f = match File::open(path) {
        Ok(f) => f,
        Err(_) => return &[],
    };
    let len = f.metadata().map(|m| m.len() as usize).unwrap_or(0);
    if len == 0 {
        return &[];
    }
    unsafe {
        let p = mmap(std::ptr::null_mut(), len, 1, 2, f.as_raw_fd(), 0);
        if p as isize == -1 {
            let v = std::fs::read(path).unwrap_or_default();
            return Box::leak(v.into_boxed_slice());
        }
        madvise(p, len, 2); // MADV_SEQUENTIAL
        std::slice::from_raw_parts(p, len)
    }
}

fn csv_files(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "csv").unwrap_or(false) && p.is_file())
            .collect(),
        Err(_) => Vec::new(),
    };
    v.sort();
    v
}

/// Chunks of files: (file index, begin, end). A chunk owns the lines that start in [begin, end).
fn make_chunks(maps: &[&[u8]], target: usize) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    for (i, m) in maps.iter().enumerate() {
        let n = m.len();
        if n == 0 {
            continue;
        }
        let k = ((n + target - 1) / target).max(1);
        let step = (n + k - 1) / k;
        let mut b = 0;
        while b < n {
            let e = (b + step).min(n);
            out.push((i, b, e));
            b = e;
        }
    }
    // biggest first for better load balance
    out.sort_by_key(|&(_, b, e)| std::cmp::Reverse(e - b));
    out
}

/// Iterate the lines starting in [b, e) of `m`, skipping the header line of the file.
#[inline]
fn for_lines(m: &[u8], b: usize, e: usize, mut f: impl FnMut(&[u8])) {
    let mut pos = if b == 0 {
        0
    } else {
        // first line start >= b: after the first '\n' at index >= b-1
        match memchr(b'\n', &m[b - 1..]) {
            Some(i) => b - 1 + i + 1,
            None => return,
        }
    };
    if pos == 0 {
        // skip header
        match memchr(b'\n', m) {
            Some(i) => pos = i + 1,
            None => return,
        }
    }
    while pos < e {
        let rest = &m[pos..];
        let (line, next) = match memchr(b'\n', rest) {
            Some(i) => (&rest[..i], pos + i + 1),
            None => (rest, m.len()),
        };
        f(line);
        pos = next;
    }
}

#[inline]
fn memchr(c: u8, s: &[u8]) -> Option<usize> {
    s.iter().position(|&x| x == c)
}

/// Split a CSV line into exactly N fields, removing whole-field quotes. False if malformed.
#[inline]
fn fields<'a, const N: usize>(line: &'a [u8], out: &mut [&'a [u8]; N]) -> bool {
    let mut n = 0;
    let mut start = 0;
    let len = line.len();
    let mut i = 0;
    loop {
        if i == len || line[i] == b',' {
            if n == N {
                return false;
            }
            let f = &line[start..i];
            let f = if f.contains(&b'"') {
                if f.len() >= 2 && f[0] == b'"' && f[f.len() - 1] == b'"' && !f[1..f.len() - 1].contains(&b'"') {
                    &f[1..f.len() - 1]
                } else {
                    return false;
                }
            } else {
                f
            };
            out[n] = f;
            n += 1;
            if i == len {
                break;
            }
            start = i + 1;
        }
        i += 1;
    }
    n == N
}


/// SIMD line splitter: calls `emit` with the N unquoted fields of every well-formed line
/// starting in [b, e) (skipping the file's header line). Malformed lines are skipped.
#[inline(always)]
fn scan_lines<const N: usize>(m: &[u8], b: usize, e: usize, emit: &mut impl FnMut(&[&[u8]; N])) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("bmi2") {
            unsafe { scan_lines_avx512::<N>(m, b, e, emit) };
            return;
        }
        if is_x86_feature_detected!("avx2") {
            unsafe { scan_lines_avx2::<N>(m, b, e, emit) };
            return;
        }
    }
    for_lines(m, b, e, |line| {
        let mut f: [&[u8]; N] = [&[]; N];
        if fields(line, &mut f) {
            emit(&f);
        }
    });
}

#[inline(always)]
fn first_line_start(m: &[u8], b: usize) -> Option<usize> {
    let mut pos = if b == 0 {
        0
    } else {
        b - 1 + memchr(b'\n', &m[b - 1..])? + 1
    };
    if pos == 0 {
        pos = memchr(b'\n', m)? + 1;
    }
    Some(pos)
}

#[inline(always)]
fn emit_line<'a, const N: usize>(m: &'a [u8], line_start: usize, ends: &[usize; N], quotes: u32, emit: &mut impl FnMut(&[&'a [u8]; N])) {
    let mut f: [&[u8]; N] = [&[]; N];
    let mut s = line_start;
    if quotes == 0 {
        for k in 0..N {
            let t = ends[k];
            f[k] = unsafe { m.get_unchecked(s..t) };
            s = t + 1;
        }
    } else {
        let mut qf = 0u32;
        let mut ok = true;
        for k in 0..N {
            let t = ends[k];
            let l = t - s;
            let q = (unsafe { *m.get_unchecked(s) } == b'"' && l > 0) as usize;
            let last_ok = l >= 2 && unsafe { *m.get_unchecked(t - 1) } == b'"';
            ok &= q == 0 || last_ok;
            qf += q as u32;
            let a = s + q;
            let z = if t >= a + q { t - q } else { a };
            f[k] = unsafe { m.get_unchecked(a..z) };
            s = t + 1;
        }
        if !ok || quotes != 2 * qf {
            return;
        }
    }
    emit(&f);
}

/// Like `emit_line`, with the line's quote mask (bit i = byte line_start+i is a quote).
#[inline(always)]
fn emit_line_q<'a, const N: usize>(m: &'a [u8], ls: usize, ends: &[usize; N], q0: u64, q1: u64, emit: &mut impl FnMut(&[&'a [u8]; N])) {
    let mut f: [&[u8]; N] = [&[]; N];
    let mut s = ls;
    if (q0 | q1) == 0 {
        for k in 0..N {
            let t = ends[k];
            f[k] = unsafe { m.get_unchecked(s..t) };
            s = t + 1;
        }
    } else {
        let qm = (q0 as u128) | ((q1 as u128) << 64);
        let mut expect: u128 = 0;
        for k in 0..N {
            let t = ends[k];
            let rs = s - ls;
            let rt = t - ls;
            let q = (((qm >> rs) & 1) as usize) & ((t > s) as usize);
            // a quoted field needs length >= 2; its closing quote is at rt-1
            let qq = q & ((t >= s + 2) as usize);
            expect |= ((qq as u128) << rs) | ((qq as u128) << (rt.wrapping_sub(1) & 127));
            let a = s + qq;
            let z = t - qq;
            f[k] = unsafe { m.get_unchecked(a..z) };
            s = t + 1;
        }
        if expect != qm {
            return;
        }
    }
    emit(&f);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,popcnt")]
unsafe fn scan_lines_avx2<const N: usize>(m: &[u8], b: usize, e: usize, emit: &mut impl FnMut(&[&[u8]; N])) {
    use std::arch::x86_64::*;
    let len = m.len();
    let mut s = match first_line_start(m, b) {
        Some(p) => p,
        None => return,
    };
    let vc = _mm256_set1_epi8(b',' as i8);
    let vn = _mm256_set1_epi8(b'\n' as i8);
    let vq = _mm256_set1_epi8(b'"' as i8);
    let base = m.as_ptr();
    let mut ends = [0usize; N];
    #[inline(always)]
    unsafe fn mask64(p: *const u8, v: __m256i) -> (u64, u64, u64) {
        let _ = v;
        (0, 0, 0)
    }
    let _ = mask64;
    while s < e && s + 128 <= len {
        let p = base.add(s);
        let a0 = _mm256_loadu_si256(p as *const __m256i);
        let a1 = _mm256_loadu_si256(p.add(32) as *const __m256i);
        let a2 = _mm256_loadu_si256(p.add(64) as *const __m256i);
        let a3 = _mm256_loadu_si256(p.add(96) as *const __m256i);
        let mv = |x: __m256i, v: __m256i| _mm256_movemask_epi8(_mm256_cmpeq_epi8(x, v)) as u32 as u128;
        let nm = mv(a0, vn) | (mv(a1, vn) << 32) | (mv(a2, vn) << 64) | (mv(a3, vn) << 96);
        if nm == 0 {
            // very long line: scalar
            let (line, next) = match memchr(b'\n', &m[s..]) {
                Some(i) => (&m[s..s + i], s + i + 1),
                None => (&m[s..], len),
            };
            let mut f: [&[u8]; N] = [&[]; N];
            if fields(line, &mut f) {
                emit(&f);
            }
            s = next;
            continue;
        }
        let pl = nm.trailing_zeros() as usize;
        let below = (1u128 << pl) - 1;
        let mut cm = (mv(a0, vc) | (mv(a1, vc) << 32) | (mv(a2, vc) << 64) | (mv(a3, vc) << 96)) & below;
        if cm.count_ones() as usize == N - 1 {
            for k in 0..N - 1 {
                ends[k] = s + cm.trailing_zeros() as usize;
                cm &= cm - 1;
            }
            ends[N - 1] = s + pl;
            let qm = (mv(a0, vq) | (mv(a1, vq) << 32) | (mv(a2, vq) << 64) | (mv(a3, vq) << 96)) & below;
            emit_line(m, s, &ends, qm.count_ones(), emit);
        }
        s += pl + 1;
    }
    // tail: scalar
    if s < e && s < len {
        let mut pos = s;
        while pos < e && pos < len {
            let rest = &m[pos..];
            let (line, next) = match memchr(b'\n', rest) {
                Some(i) => (&rest[..i], pos + i + 1),
                None => (rest, len),
            };
            let mut f: [&[u8]; N] = [&[]; N];
            if fields(line, &mut f) {
                emit(&f);
            }
            pos = next;
        }
    }
}


#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx2,bmi1,bmi2,popcnt")]
unsafe fn scan_lines_avx512<const N: usize>(m: &[u8], b: usize, e: usize, emit: &mut impl FnMut(&[&[u8]; N])) {
    use std::arch::x86_64::*;
    let len = m.len();
    let mut s = match first_line_start(m, b) {
        Some(p) => p,
        None => return,
    };
    let vc = _mm512_set1_epi8(b',' as i8);
    let vn = _mm512_set1_epi8(b'\n' as i8);
    let vq = _mm512_set1_epi8(b'"' as i8);
    let base = m.as_ptr();
    let mut ends = [0usize; N];
    while s < e && s + 128 <= len {
        let p = base.add(s);
        let a0 = _mm512_loadu_si512(p as *const _);
        let a1 = _mm512_loadu_si512(p.add(64) as *const _);
        let n0 = _mm512_cmpeq_epi8_mask(a0, vn);
        let n1 = _mm512_cmpeq_epi8_mask(a1, vn);
        let pl = if n0 != 0 {
            n0.trailing_zeros() as usize
        } else if n1 != 0 {
            64 + n1.trailing_zeros() as usize
        } else {
            let (line, next) = match memchr(b'\n', &m[s..]) {
                Some(i) => (&m[s..s + i], s + i + 1),
                None => (&m[s..], len),
            };
            let mut f: [&[u8]; N] = [&[]; N];
            if fields(line, &mut f) {
                emit(&f);
            }
            s = next;
            continue;
        };
        let c0 = _bzhi_u64(_mm512_cmpeq_epi8_mask(a0, vc), pl as u32);
        let c1 = _bzhi_u64(_mm512_cmpeq_epi8_mask(a1, vc), pl.saturating_sub(64) as u32);
        if (c0.count_ones() + c1.count_ones()) as usize == N - 1 {
            let mut x0 = c0;
            let mut x1 = c1;
            for k in 0..N - 1 {
                let z0 = x0 == 0;
                let pos = if z0 { 64 + x1.trailing_zeros() as usize } else { x0.trailing_zeros() as usize };
                ends[k] = s + pos;
                x0 &= x0.wrapping_sub(1);
                x1 = if z0 { x1 & x1.wrapping_sub(1) } else { x1 };
            }
            ends[N - 1] = s + pl;
            let q0 = _bzhi_u64(_mm512_cmpeq_epi8_mask(a0, vq), pl as u32);
            let q1 = _bzhi_u64(_mm512_cmpeq_epi8_mask(a1, vq), pl.saturating_sub(64) as u32);
            emit_line(m, s, &ends, q0.count_ones() + q1.count_ones(), emit);
        }
        s += pl + 1;
    }
    while s < e && s < len {
        let rest = &m[s..];
        let (line, next) = match memchr(b'\n', rest) {
            Some(i) => (&rest[..i], s + i + 1),
            None => (rest, len),
        };
        let mut f: [&[u8]; N] = [&[]; N];
        if fields(line, &mut f) {
            emit(&f);
        }
        s = next;
    }
}


// ------------------------------------------------------------------ memory arenas
/// Anonymous memory, asked for as huge pages and pre-faulted in bulk (cheaper than 4 KB faults).
fn arena(bytes: usize) -> *mut u8 {
    let bytes = bytes.max(4096);
    unsafe {
        let p = mmap(std::ptr::null_mut(), bytes, 3, 0x02 | 0x20 | 0x4000, -1, 0);
        if p as isize == -1 {
            panic!("mmap failed");
        }
        madvise(p, bytes, 14); // MADV_HUGEPAGE
        if std::env::var_os("SS_PREFAULT").is_some() { madvise(p, bytes, 23); }
        p
    }
}

/// P partitions of fixed capacity inside one arena; overflow spills into Vecs.
struct Parts<T: Copy> {
    base: *mut T,
    cap: usize,
    len: Vec<u32>,
    spill: Vec<Vec<T>>,
}
unsafe impl<T: Copy + Send> Send for Parts<T> {}
unsafe impl<T: Copy + Sync> Sync for Parts<T> {}
impl<T: Copy> Parts<T> {
    fn new(mem: *mut u8, bytes: usize, np: usize) -> Self {
        let cap = bytes / std::mem::size_of::<T>() / np;
        Parts { base: mem as *mut T, cap, len: vec![0; np], spill: (0..np).map(|_| Vec::new()).collect() }
    }
    fn alloc(count_per_part: usize, np: usize) -> Self {
        let bytes = count_per_part * np * std::mem::size_of::<T>();
        Parts::new(arena(bytes), bytes, np)
    }
    #[inline(always)]
    fn push(&mut self, p: usize, x: T) {
        unsafe {
            let l = *self.len.get_unchecked(p) as usize;
            if l < self.cap {
                self.base.add(p * self.cap + l).write(x);
                *self.len.get_unchecked_mut(p) = l as u32 + 1;
            } else {
                self.spill.get_unchecked_mut(p).push(x);
            }
        }
    }
    #[inline(always)]
    fn main(&self, p: usize) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.base.add(p * self.cap), self.len[p] as usize) }
    }
    #[inline(always)]
    fn count(&self, p: usize) -> usize {
        self.len[p] as usize + self.spill[p].len()
    }
    #[inline(always)]
    fn each(&self, p: usize, mut f: impl FnMut(&T)) {
        for x in self.main(p) {
            f(x);
        }
        for x in &self.spill[p] {
            f(x);
        }
    }
    fn clear(&mut self) {
        for l in self.len.iter_mut() {
            *l = 0;
        }
        for v in self.spill.iter_mut() {
            v.clear();
        }
    }
    /// Reuse this arena's memory for another record type.
    fn recycle<U: Copy>(self, np: usize) -> Parts<U> {
        let bytes = self.cap * np * std::mem::size_of::<T>();
        Parts::new(self.base as *mut u8, bytes, np)
    }
}

// ------------------------------------------------------------------ keys
const fn build_enc() -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 10 {
        t[b'0' as usize + i] = 1 + i as u8;
        i += 1;
    }
    i = 0;
    while i < 26 {
        t[b'A' as usize + i] = 11 + i as u8;
        t[b'a' as usize + i] = 37 + i as u8;
        i += 1;
    }
    t
}
static ENC: [u8; 256] = build_enc();
const DEC: &[u8; 63] = b"\x000123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

#[inline]
fn enc_key(s: &[u8]) -> Option<u64> {
    if s.len() > 10 || s.is_empty() {
        return None;
    }
    let mut v = 0u64;
    for &c in s {
        let e = ENC[c as usize];
        if e == 0 {
            return None;
        }
        v = (v << 6) | e as u64;
    }
    Some(v << (6 * (10 - s.len())))
}

fn dec_key(k: u64, out: &mut Vec<u8>) {
    for i in 0..10 {
        let c = ((k >> (54 - 6 * i)) & 63) as usize;
        if c == 0 {
            break;
        }
        out.push(DEC[c]);
    }
}

/// Interner for IDs that do not fit the compact encoding (rare).
struct Exotic {
    inner: Mutex<(HashMap<Vec<u8>, u64>, Vec<Vec<u8>>)>,
}
impl Exotic {
    fn new() -> Self {
        Exotic { inner: Mutex::new((HashMap::new(), Vec::new())) }
    }
    #[inline]
    fn key(&self, s: &[u8]) -> u64 {
        match enc_key(s) {
            Some(k) => k,
            None => self.intern(s),
        }
    }
    #[cold]
    fn intern(&self, s: &[u8]) -> u64 {
        let mut g = self.inner.lock().unwrap();
        if let Some(&k) = g.0.get(s) {
            return k;
        }
        let k = EXOTIC | g.1.len() as u64;
        g.1.push(s.to_vec());
        g.0.insert(s.to_vec(), k);
        k
    }
    fn bytes(&self, k: u64, out: &mut Vec<u8>) {
        if k & EXOTIC != 0 {
            let g = self.inner.lock().unwrap();
            out.extend_from_slice(&g.1[(k & !EXOTIC) as usize]);
        } else {
            dec_key(k, out);
        }
    }
    #[inline]
    fn cmp(&self, a: u64, b: u64) -> Ordering {
        if (a | b) & EXOTIC == 0 {
            a.cmp(&b)
        } else {
            self.cmp_slow(a, b)
        }
    }
    #[cold]
    fn cmp_slow(&self, a: u64, b: u64) -> Ordering {
        let mut x = Vec::new();
        let mut y = Vec::new();
        self.bytes(a, &mut x);
        self.bytes(b, &mut y);
        x.cmp(&y)
    }
}


/// Fast compact key for a field known to have 16 readable bytes from its start.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn enc_key16(s: &[u8]) -> Option<u64> {
    use std::arch::x86_64::*;
    let l = s.len();
    if l == 0 || l > 10 {
        return None;
    }
    let c = _mm_loadu_si128(s.as_ptr() as *const __m128i);
    let idx = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    let lm = _mm_cmpgt_epi8(_mm_set1_epi8(l as i8), idx);
    let d = _mm_and_si128(_mm_cmpgt_epi8(c, _mm_set1_epi8(b'0' as i8 - 1)), _mm_cmpgt_epi8(_mm_set1_epi8(b'9' as i8 + 1), c));
    let u = _mm_and_si128(_mm_cmpgt_epi8(c, _mm_set1_epi8(b'A' as i8 - 1)), _mm_cmpgt_epi8(_mm_set1_epi8(b'Z' as i8 + 1), c));
    let lo = _mm_and_si128(_mm_cmpgt_epi8(c, _mm_set1_epi8(b'a' as i8 - 1)), _mm_cmpgt_epi8(_mm_set1_epi8(b'z' as i8 + 1), c));
    let valid = _mm_or_si128(_mm_or_si128(d, u), lo);
    let need = _mm_movemask_epi8(lm);
    if _mm_movemask_epi8(_mm_and_si128(valid, lm)) != need {
        return None;
    }
    let code = _mm_or_si128(
        _mm_or_si128(_mm_and_si128(_mm_sub_epi8(c, _mm_set1_epi8(47)), d), _mm_and_si128(_mm_sub_epi8(c, _mm_set1_epi8(54)), u)),
        _mm_and_si128(_mm_sub_epi8(c, _mm_set1_epi8(60)), lo),
    );
    let code = _mm_and_si128(code, lm);
    let w = _mm_maddubs_epi16(code, _mm_set1_epi16(0x0140)); // bytes (64, 1)
    let dw = _mm_madd_epi16(w, _mm_set1_epi32(0x0001_1000)); // words (4096, 1)
    let a = _mm_cvtsi128_si32(dw) as u32 as u64;
    let b = _mm_extract_epi32(dw, 1) as u32 as u64;
    let cc = _mm_extract_epi32(dw, 2) as u32 as u64;
    Some((a << 36) | (b << 12) | (cc >> 12))
}

impl Exotic {
    // Event keys need equality only. Nine ASCII bytes fit in 63 bits, and a
    // first byte >= 16 keeps these keys above the 60-bit compact alphabet.
    #[inline(always)]
    fn event_key(&self, id: &[u8], limit: usize) -> u64 {
        #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
        if id.len() == 9 && (16..128).contains(&id[0]) {
            let tail = u64::from_le_bytes(id[1..].try_into().unwrap());
            const ASCII: u64 = 0x7f7f_7f7f_7f7f_7f7f;
            if tail & !ASCII == 0 {
                let suffix = unsafe { std::arch::x86_64::_pext_u64(tail, ASCII) };
                return (u64::from(id[0]) << 56) | suffix;
            }
        }
        self.key_in(id, limit)
    }

    /// Key for a field inside a buffer ending at `lim`.
    #[inline(always)]
    fn key_in(&self, s: &[u8], lim: usize) -> u64 {
        #[cfg(target_arch = "x86_64")]
        {
            if (s.as_ptr() as usize) + 16 <= lim {
                if let Some(k) = unsafe { enc_key16(s) } {
                    return k;
                }
                return self.intern(s);
            }
        }
        self.key(s)
    }
}

#[inline]
fn hmix(k: u64) -> u64 {
    let h = k.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^ (h >> 31)
}

// ------------------------------------------------------------------ time
#[inline]
fn d2(s: &[u8], i: usize) -> Option<i64> {
    let a = s[i].wrapping_sub(b'0');
    let b = s[i + 1].wrapping_sub(b'0');
    if a < 10 && b < 10 {
        Some(a as i64 * 10 + b as i64)
    } else {
        None
    }
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

#[inline]
fn dim(y: i64, m: i64) -> i64 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 {
                29
            } else {
                28
            }
        }
    }
}

/// Parse a timestamp (F1, F2 or F3) to epoch ms.
#[inline]
fn parse_ts(s: &[u8]) -> Option<i64> {
    let n = s.len();
    if n == 0 {
        return None;
    }
    if n == 20 || n == 25 {
        if s[4] == b'-' {
            return parse_dt(s);
        }
    }
    // F3: digits only
    let mut v: i64 = 0;
    for &c in s {
        let d = c.wrapping_sub(b'0');
        if d >= 10 {
            return None;
        }
        v = v.checked_mul(10)?.checked_add(d as i64)?;
    }
    Some(v)
}

#[inline]
fn parse_dt(s: &[u8]) -> Option<i64> {
    let n = s.len();
    let y = d2(s, 0)? * 100 + d2(s, 2)?;
    if s[4] != b'-' || s[7] != b'-' || s[10] != b'T' || s[13] != b':' || s[16] != b':' {
        return None;
    }
    let mo = d2(s, 5)?;
    let d = d2(s, 8)?;
    let h = d2(s, 11)?;
    let mi = d2(s, 14)?;
    let se = d2(s, 17)?;
    let off = if n == 20 {
        if s[19] != b'Z' {
            return None;
        }
        0
    } else {
        let sign = match s[19] {
            b'+' => 1,
            b'-' => -1,
            _ => return None,
        };
        if s[22] != b':' {
            return None;
        }
        let oh = d2(s, 20)?;
        let om = d2(s, 23)?;
        if oh > 23 || om > 59 {
            return None;
        }
        sign * (oh * 60 + om)
    };
    if !(1..=12).contains(&mo) || d < 1 || d > dim(y, mo) || h > 23 || mi > 59 || se > 59 {
        return None;
    }
    let days = days_from_civil(y, mo, d);
    Some(((days * 86400 + h * 3600 + mi * 60 + se) - off * 60) * 1000)
}


/// Per-thread cache of validated dates ("YYYY-MM-DD" -> epoch ms of midnight).
struct DateCache {
    k1: [u64; 64],
    k2: [u16; 64],
    v: [i64; 64],
}
const BAD_DATE: i64 = i64::MIN;
impl DateCache {
    fn new() -> Self {
        DateCache { k1: [0; 64], k2: [0; 64], v: [BAD_DATE; 64] }
    }
    #[cold]
    fn fill(&mut self, s: &[u8], h: usize, k1: u64, k2: u16) -> i64 {
        let v = (|| {
            let y = d2(s, 0)? * 100 + d2(s, 2)?;
            if s[4] != b'-' || s[7] != b'-' {
                return None;
            }
            let mo = d2(s, 5)?;
            let d = d2(s, 8)?;
            if !(1..=12).contains(&mo) || d < 1 || d > dim(y, mo) {
                return None;
            }
            Some(days_from_civil(y, mo, d) * DAY_MS)
        })()
        .unwrap_or(BAD_DATE);
        self.k1[h] = k1;
        self.k2[h] = k2;
        self.v[h] = v;
        v
    }
}

const TM: u64 = 0xFFFF_00FF_FF00_FFFF;
const TCOLON: u64 = 0x0000_3A00_003A_0000;
const D30: u64 = 0x3030_3030_3030_3030;
const D06: u64 = 0x0606_0606_0606_0606;
const DF0: u64 = 0xF0F0_F0F0_F0F0_F0F0;

/// Fast timestamp parser (same results as `parse_ts`).
// Reduce eight little-endian digit lanes to one decimal integer.
#[inline(always)]
fn decimal8(digits: u64) -> u32 {
    let pairs = (digits.wrapping_mul(10) + (digits >> 8)) & 0x00ff_00ff_00ff_00ff;
    let quads = (pairs * 100 + (pairs >> 16)) & 0x0000_ffff_0000_ffff;
    (quads as u32) * 10_000 + (quads >> 32) as u32
}

#[inline(always)]
fn epoch13(s: &[u8]) -> Option<i64> {
    // Two overlapping loads stay within the field. Left-pad the five trailing
    // digits with three zero lanes so both groups use the same reduction.
    let a = u64::from_le_bytes(s[..8].try_into().unwrap()) ^ D30;
    let b = (u64::from_le_bytes(s[5..13].try_into().unwrap()) ^ D30) & 0xffff_ffff_ff00_0000;
    if (a | b | a.wrapping_add(D06) | b.wrapping_add(D06)) & DF0 != 0 {
        return None;
    }
    Some(i64::from(decimal8(a)) * 100_000 + i64::from(decimal8(b)))
}

#[cfg(test)]
mod epoch_tests {
    use super::epoch13;

    #[test]
    fn decimal_timestamp_matches_scalar_parsing() {
        for sample in [b"0000000000000", b"0123456789012", b"9999999999999"] {
            for position in 0..sample.len() {
                for byte in 0..=255 {
                    let mut input = *sample;
                    input[position] = byte;
                    let expected = input.iter().all(u8::is_ascii_digit).then(|| {
                        input.iter().fold(0i64, |n, &c| n * 10 + i64::from(c - b'0'))
                    });
                    assert_eq!(epoch13(&input), expected, "{input:?}");
                }
            }
        }
    }
}

#[inline(always)]
fn parse_ts_fast(s: &[u8], dc: &mut DateCache) -> Option<i64> {
    let n = s.len();
    if n == 13 {
        return epoch13(s);
    }
    if (n == 20 || n == 25) && s[4] == b'-' {
        let k1 = u64::from_le_bytes(s[0..8].try_into().unwrap());
        let k2 = u16::from_le_bytes(s[8..10].try_into().unwrap());
        let h = ((k1 ^ ((k2 as u64) << 21)).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 58) as usize;
        let mut dms = dc.v[h];
        if dc.k1[h] != k1 || dc.k2[h] != k2 || dms == BAD_DATE {
            if dc.k1[h] == k1 && dc.k2[h] == k2 {
                return None;
            }
            dms = dc.fill(s, h, k1, k2);
            if dms == BAD_DATE {
                return None;
            }
        }
        if s[10] != b'T' {
            return None;
        }
        let t = u64::from_le_bytes(s[11..19].try_into().unwrap());
        if t & !TM != TCOLON {
            return None;
        }
        let x = t & TM;
        if x & (DF0 & TM) != (D30 & TM) || (x.wrapping_add(D06 & TM)) & (DF0 & TM) != (D30 & TM) {
            return None;
        }
        let v = x.wrapping_sub(D30 & TM);
        let b = |i: u32| ((v >> (8 * i)) & 0xFF) as i64;
        let hh = b(0) * 10 + b(1);
        let mi = b(3) * 10 + b(4);
        let se = b(6) * 10 + b(7);
        if hh > 23 || mi > 59 || se > 59 {
            return None;
        }
        let mut ms = dms + (hh * 3600 + mi * 60 + se) * 1000;
        if n == 20 {
            if s[19] != b'Z' {
                return None;
            }
        } else {
            let sign = match s[19] {
                b'+' => 1,
                b'-' => -1,
                _ => return None,
            };
            if s[22] != b':' {
                return None;
            }
            let oh = d2(s, 20)?;
            let om = d2(s, 23)?;
            if oh > 23 || om > 59 {
                return None;
            }
            ms -= sign * (oh * 60 + om) * 60_000;
        }
        return Some(ms);
    }
    if n == 0 {
        return None;
    }
    if n <= 19 {
        let mut v: u64 = 0;
        for &c in s {
            let d = c.wrapping_sub(b'0');
            if d >= 10 {
                return None;
            }
            v = v * 10 + d as u64;
        }
        if v > i64::MAX as u64 {
            return None;
        }
        return Some(v as i64);
    }
    parse_ts(s)
}

#[inline]
fn floor_div(a: i64, b: i64) -> i64 {
    let q = a / b;
    if (a % b != 0) && ((a < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

// ------------------------------------------------------------------ lookups
struct Lookup {
    p0: u8,
    p1: u8,
    plen: usize,
    arr: Vec<u16>,
    map: HashMap<Vec<u8>, u16>,
}
const NONE16: u16 = u16::MAX;
const ARR_LIMIT: usize = 10_000_000;
impl Lookup {
    fn new(prefix: &'static [u8]) -> Self {
        Lookup { p0: prefix[0], p1: if prefix.len() > 1 { prefix[1] } else { 0 }, plen: prefix.len(), arr: Vec::new(), map: HashMap::new() }
    }
    #[inline(always)]
    fn canon(&self, s: &[u8]) -> Option<usize> {
        let p = self.plen;
        let n = s.len();
        if n <= p || n > p + 7 || s[0] != self.p0 || (p == 2 && s[1] != self.p1) {
            return None;
        }
        let r = &s[p..];
        if r[0] == b'0' && r.len() > 1 {
            return None;
        }
        let mut v = 0usize;
        for &c in r {
            let d = c.wrapping_sub(b'0');
            if d >= 10 {
                return None;
            }
            v = v * 10 + d as usize;
        }
        Some(v)
    }
    fn insert(&mut self, s: &[u8], val: u16) {
        match self.canon(s) {
            Some(n) if n < ARR_LIMIT => {
                if n >= self.arr.len() {
                    self.arr.resize(n + 1, NONE16);
                }
                self.arr[n] = val;
            }
            _ => {
                self.map.insert(s.to_vec(), val);
            }
        }
    }
    #[inline(always)]
    fn get(&self, s: &[u8]) -> Option<u16> {
        match self.canon(s) {
            Some(n) if n < ARR_LIMIT => match self.arr.get(n) {
                Some(&v) if v != NONE16 => Some(v),
                _ => None,
            },
            _ => self.get_slow(s),
        }
    }
    #[cold]
    fn get_slow(&self, s: &[u8]) -> Option<u16> {
        if self.map.is_empty() {
            None
        } else {
            self.map.get(s).copied()
        }
    }
}

/// Read a small clean 2-column CSV.
fn read_pairs(path: &Path) -> Vec<(Vec<u8>, Vec<u8>)> {
    let m = map_file(path);
    let mut v = Vec::new();
    for_lines(m, 0, m.len(), |line| {
        let mut f: [&[u8]; 2] = [&[]; 2];
        if fields(line, &mut f) {
            v.push((f[0].to_vec(), f[1].to_vec()));
        }
    });
    v
}

fn manifest_value(text: &str, key: &str) -> i64 {
    let i = text.find(&format!("\"{}\"", key)).expect("manifest key");
    let rest = &text[i + key.len() + 2..];
    let c = rest.find(':').unwrap();
    let rest = &rest[c + 1..];
    let q1 = rest.find('"').unwrap();
    let rest = &rest[q1 + 1..];
    let q2 = rest.find('"').unwrap();
    parse_ts(rest[..q2].as_bytes()).expect("manifest timestamp")
}

// ------------------------------------------------------------------ link map
#[inline(always)]
fn hu(k: u64) -> u64 {
    k.wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
}

struct LinkMap {
    slots: Vec<[u64; 2]>,
    bits: u32,
}
impl LinkMap {
    fn empty() -> Self {
        LinkMap { slots: Vec::new(), bits: 0 }
    }
    #[inline(always)]
    fn get(&self, k: u64) -> u64 {
        if self.slots.is_empty() {
            return k;
        }
        let mask = self.slots.len() - 1;
        let mut i = (hu(k) >> (64 - self.bits)) as usize;
        loop {
            let s = unsafe { self.slots.get_unchecked(i) };
            if s[0] == k {
                return s[1];
            }
            if s[0] == 0 {
                return k;
            }
            i = (i + 1) & mask;
        }
    }
}

fn build_linkmap(pairs: &[Vec<(u64, u64)>], ex: &Exotic) -> LinkMap {
    let total: usize = pairs.iter().map(|v| v.len()).sum();
    if total == 0 {
        return LinkMap::empty();
    }
    let cap = (total * 4).next_power_of_two().max(16);
    let bits = cap.trailing_zeros();
    let mask = cap - 1;
    let mut slots = vec![[0u64; 2]; cap];
    let mut ukeys: Vec<u64> = Vec::with_capacity(total * 2);
    let mut parent: Vec<u32> = Vec::with_capacity(total * 2);
    #[inline(always)]
    fn intern(k: u64, slots: &mut [[u64; 2]], bits: u32, mask: usize, ukeys: &mut Vec<u64>, parent: &mut Vec<u32>) -> u32 {
        let mut i = (hu(k) >> (64 - bits)) as usize;
        loop {
            let s = &mut slots[i];
            if s[0] == k {
                return s[1] as u32;
            }
            if s[0] == 0 {
                let id = ukeys.len() as u32;
                s[0] = k;
                s[1] = id as u64;
                ukeys.push(k);
                parent.push(id);
                return id;
            }
            i = (i + 1) & mask;
        }
    }
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            let g = p[p[x as usize] as usize];
            p[x as usize] = g;
            x = g;
        }
        x
    }
    for v in pairs {
        for &(a, b) in v {
            let ia = intern(a, &mut slots, bits, mask, &mut ukeys, &mut parent);
            let ib = intern(b, &mut slots, bits, mask, &mut ukeys, &mut parent);
            let ra = find(&mut parent, ia);
            let rb = find(&mut parent, ib);
            if ra != rb {
                if ra < rb {
                    parent[rb as usize] = ra;
                } else {
                    parent[ra as usize] = rb;
                }
            }
        }
    }
    let n = ukeys.len();
    let mut minkey: Vec<u64> = ukeys.clone();
    for i in 0..n {
        let r = find(&mut parent, i as u32) as usize;
        if r != i && ex.cmp(ukeys[i], minkey[r]) == Ordering::Less {
            minkey[r] = ukeys[i];
        }
    }
    for s in slots.iter_mut() {
        if s[0] != 0 {
            let r = find(&mut parent, s[1] as u32) as usize;
            s[1] = minkey[r];
        }
    }
    LinkMap { slots, bits }
}

// ------------------------------------------------------------------ records
#[derive(Clone, Copy)]
#[repr(C)]
struct EvRec {
    eid: u64,
    user: u64,
    ts: u32,     // ms since base (window_start - 7 days)
    ingest: u32, // epoch seconds (saturated)
    pc: u32,     // price (24 bits) | cat << 24
    ok: u32,
}

/// An event (user or person key, ts, price|cat).
#[derive(Clone, Copy)]
#[repr(C)]
struct PRec {
    person: u64,
    ts: u32,
    pc: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct CRec {
    person: u64,
    cid: u64,
    ts: u32,
    camp: u32,
}

#[derive(Clone, Copy)]
struct URow {
    person: u64,
    day: i32,
    sessions: u32,
    events: u32,
    revenue: u64,
    top: i32,
}

struct Ctx {
    base: i64,
    ws: i64,
    we: i64,
    day0: i64,
    nd: usize,
    ncat: usize,
    ncamp: usize,
    prods: Lookup,
    camps: Lookup,
    unknown_cat: u32,
}

const P1_BITS: u32 = 10;
const P2_BITS: u32 = 10;
const P1: usize = 1 << P1_BITS;
const P2: usize = 1 << P2_BITS;

#[inline(always)]
fn part1(eid: u64) -> usize {
    (eid.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - P1_BITS)) as usize
}
/// User and person partitions share `hu`, so a user partition touches one region of the link map.
#[inline(always)]
fn part2(k: u64) -> usize {
    (hu(k) >> (64 - P2_BITS)) as usize
}

#[inline]
fn parse_price(s: &[u8]) -> u32 {
    // 0 = invalid
    if s.is_empty() {
        return 0;
    }
    let mut v: u64 = 0;
    for &c in s {
        let d = c.wrapping_sub(b'0');
        if d >= 10 {
            return 0;
        }
        v = v * 10 + d as u64;
        if v > 10_000_000 {
            // keep scanning digits? any further char only matters if non-digit -> still invalid
            return 0;
        }
    }
    v as u32
}

fn parse_events(ctx: &Ctx, ex: &Exotic, dc: &mut DateCache, m: &[u8], b: usize, e: usize, out: &mut Parts<EvRec>) {
    let base = ctx.base;
    let lo = ctx.ws;
    let hi = ctx.we;
    let lim = m.as_ptr() as usize + m.len();
    scan_lines::<7>(m, b, e, &mut |f: &[&[u8]; 7]| {
        if f[0].is_empty() {
            return;
        }
        let eid = ex.event_key(f[0], lim);
        let ingest_ms = parse_ts_fast(f[3], dc).unwrap_or(i64::MAX / 4);
        let ingest = (ingest_ms.div_euclid(1000)).clamp(0, u32::MAX as i64) as u32;
        let mut ok = !f[1].is_empty() && !f[5].is_empty();
        let mut ts = 0i64;
        if ok {
            match parse_ts_fast(f[2], dc) {
                Some(t) if t >= lo && t < hi => ts = t,
                _ => ok = false,
            }
        }
        let mut price = 0u32;
        if ok {
            let ty = f[4];
            if ty.len() == 4 {
                let w = u32::from_le_bytes([ty[0], ty[1], ty[2], ty[3]]);
                if w != u32::from_le_bytes(*b"view") && w != u32::from_le_bytes(*b"cart") {
                    ok = false;
                }
            } else if ty.len() == 8 && u64::from_le_bytes(ty.try_into().unwrap()) == u64::from_le_bytes(*b"purchase") {
                price = parse_price(f[6]);
                if price == 0 {
                    ok = false;
                }
            } else {
                ok = false;
            }
        }
        if ok && ingest_ms - ts > DAY_MS {
            ok = false;
        }
        let rec = if ok {
            let user = ex.key_in(f[1], lim);
            // Only purchases contribute to category revenue.
            let pc = if price == 0 {
                0
            } else {
                let category = ctx.prods.get(f[5]).map(u32::from).unwrap_or(ctx.unknown_cat);
                price | (category << 24)
            };
            EvRec { eid, user, ts: (ts - base) as u32, ingest, pc, ok: 1 }
        } else {
            EvRec { eid, user: 0, ts: 0, ingest, pc: 0, ok: 0 }
        };
        out.push(part1(eid), rec);
    });
}


fn parse_links(ex: &Exotic, m: &[u8], b: usize, e: usize, out: &mut Vec<(u64, u64)>) {
    let lim = m.as_ptr() as usize + m.len();
    scan_lines::<2>(m, b, e, &mut |f: &[&[u8]; 2]| {
        if f[0].is_empty() || f[1].is_empty() {
            return;
        }
        out.push((ex.key_in(f[0], lim), ex.key_in(f[1], lim)));
    });
}

// ------------------------------------------------------------------ stage 2: dedup
fn parse_clicks_direct(ctx: &Ctx, ex: &Exotic, lm: &LinkMap, dc: &mut DateCache, m: &[u8], b: usize, e: usize, out: &mut Parts<CRec>) {
    let base = ctx.base;
    let hi = ctx.we;
    let lim = m.as_ptr() as usize + m.len();
    scan_lines::<4>(m, b, e, &mut |f: &[&[u8]; 4]| {
        if f[0].is_empty() || f[1].is_empty() {
            return;
        }
        let ts = match parse_ts_fast(f[3], dc) {
            Some(t) if t >= base && t < hi => t,
            _ => return,
        };
        let camp = match ctx.camps.get(f[2]) {
            Some(c) => c as u32,
            None => return,
        };
        let cid = ex.key_in(f[0], lim);
        let person = lm.get(ex.key_in(f[1], lim));
        out.push(part2(person), CRec { person, cid, ts: (ts - base) as u32, camp });
    });
}

#[derive(Default)]
struct DedupScratch<'a> {
    // Each word tracks 64 hash buckets. The second mask flags buckets visited twice.
    buckets: Vec<[u64; 2]>,
    winners: Vec<Option<&'a EvRec>>,
}

fn dedup_direct<'a>(lm: &LinkMap, parts: &'a [Parts<EvRec>], p: usize, scratch: &mut DedupScratch<'a>, out: &mut Parts<PRec>) {
    let n: usize = parts.iter().map(|part| part.count(p)).sum();
    if n == 0 {
        return;
    }
    let buckets = (n * 8).next_power_of_two().max(64);
    scratch.buckets.clear();
    scratch.buckets.resize(buckets / 64, [0, 0]);
    let hash = |eid: u64| eid.wrapping_mul(0x9E37_79B9_7F4A_7C15) << P1_BITS;
    let shift = 64 - buckets.trailing_zeros();
    let records = || parts.iter().flat_map(|part| part.main(p).iter().chain(&part.spill[p]));
    let mut collisions = 0usize;
    for record in records() {
        let index = (hash(record.eid) >> shift) as usize;
        let bit = 1u64 << (index % 64);
        let state = &mut scratch.buckets[index / 64];
        let already_seen = state[0] & bit;
        state[1] |= already_seen;
        state[0] |= bit;
        collisions += usize::from(already_seen != 0);
    }

    // A bucket with k >= 2 records contributes k-1 collisions, so at most
    // 2 * collisions records need exact deduplication. Leave the table half empty.
    let capacity = (4 * collisions).next_power_of_two().max(2);
    scratch.winners.clear();
    scratch.winners.resize(capacity, None);
    let table_shift = 64 - capacity.trailing_zeros();
    let mut emit = |r: &EvRec| {
        if r.ok != 0 {
            let person = lm.get(r.user);
            out.push(part2(person), PRec { person, ts: r.ts, pc: r.pc });
        }
    };
    for record in records() {
        let h = hash(record.eid);
        let index = (h >> shift) as usize;
        if scratch.buckets[index / 64][1] & (1u64 << (index % 64)) == 0 {
            emit(record);
            continue;
        }
        let mut slot = (h >> table_shift) as usize;
        loop {
            match scratch.winners[slot] {
                None => {
                    scratch.winners[slot] = Some(record);
                    break;
                }
                Some(previous) if previous.eid == record.eid => {
                    if record.ingest < previous.ingest {
                        scratch.winners[slot] = Some(record);
                    }
                    break;
                }
                Some(_) => slot = (slot + 1) & (capacity - 1),
            }
        }
    }
    for winner in scratch.winners.iter().flatten() {
        emit(winner);
    }
}

// ------------------------------------------------------------------ stage 4: sessions
struct Agg {
    rows: Vec<URow>,
    camp: Vec<[u64; 4]>, // (ncamp+1) * nd: sessions, purchasing, revenue, assisted
}

struct Ses {
    start: u32,
    day: u32,
}

#[derive(Clone, Copy)]
struct CItem {
    ts: u32,
    camp: u32,
    cid: u64,
}

#[derive(Default)]
struct Scratch {
    table: Vec<u32>,
    persons: Vec<u64>,
    lidx: Vec<u32>,
    cnt: Vec<u32>,
    items: Vec<u64>,
    ccnt: Vec<u32>,
    citems: Vec<CItem>,
    ses: Vec<Ses>,
    cbuf: Vec<(u32, u32)>,
    ctmp: Vec<(u32, u32)>,
    catrev: Vec<u64>,
    day_end: Vec<usize>,
}

fn process_partition(ctx: &Ctx, ex: &Exotic, evs: &[Parts<PRec>], cls: &[Parts<CRec>], p: usize, sc: &mut Scratch, agg: &mut Agg) {
    let n: usize = evs.iter().map(|t| t.count(p)).sum();
    if n == 0 {
        return;
    }
    // local person index
    let cap = (n * 2).next_power_of_two().max(64);
    let mask = cap - 1;
    sc.table.clear();
    sc.table.resize(cap, 0);
    sc.persons.clear();
    sc.lidx.clear();
    sc.cnt.clear();
    for t in evs {
        t.each(p, |r| {
            let k = r.person;
            let mut i = (k.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as usize & mask;
            let l = loop {
                let v = sc.table[i];
                if v == 0 {
                    let l = sc.persons.len() as u32;
                    sc.persons.push(k);
                    sc.cnt.push(0);
                    sc.table[i] = l + 1;
                    break l;
                }
                if sc.persons[(v - 1) as usize] == k {
                    break v - 1;
                }
                i = (i + 1) & mask;
            };
            sc.cnt[l as usize] += 1;
            sc.lidx.push(l);
        });
    }
    let np = sc.persons.len();
    // offsets (cnt becomes start offsets; scatter with a moving cursor)
    let mut acc = 0u32;
    for c in sc.cnt.iter_mut() {
        let x = *c;
        *c = acc;
        acc += x;
    }
    sc.cnt.push(acc);
    sc.items.clear();
    sc.items.resize(n, 0);
    {
        let mut cur: Vec<u32> = sc.cnt[..np].to_vec();
        let mut q = 0usize;
        for t in evs {
            t.each(p, |r| {
                let l = sc.lidx[q] as usize;
                q += 1;
                sc.items[cur[l] as usize] = ((r.ts as u64) << 32) | r.pc as u64;
                cur[l] += 1;
            });
        }
    }
    for l in 0..np {
        let (a, b) = (sc.cnt[l] as usize, sc.cnt[l + 1] as usize);
        match b - a {
            0 | 1 => {}
            2 => {
                if sc.items[a] > sc.items[a + 1] {
                    sc.items.swap(a, a + 1);
                }
            }
            3 => {
                // sorting network for 3 elements (ascending u64 == ascending ts)
                if sc.items[a] > sc.items[a + 1] {
                    sc.items.swap(a, a + 1);
                }
                if sc.items[a + 1] > sc.items[a + 2] {
                    sc.items.swap(a + 1, a + 2);
                }
                if sc.items[a] > sc.items[a + 1] {
                    sc.items.swap(a, a + 1);
                }
            }
            _ => {
                sc.items[a..b].sort_unstable();
            }
        }
    }
    // clicks grouped by the same local index (clicks of persons without events are useless)
    sc.ccnt.clear();
    sc.ccnt.resize(np + 1, 0);
    sc.lidx.clear();
    let mut nc = 0usize;
    for t in cls {
        t.each(p, |r| {
            let k = r.person;
            let mut i = (k.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as usize & mask;
            let l = loop {
                let v = sc.table[i];
                if v == 0 {
                    break u32::MAX;
                }
                if sc.persons[(v - 1) as usize] == k {
                    break v - 1;
                }
                i = (i + 1) & mask;
            };
            sc.lidx.push(l);
            if l != u32::MAX {
                sc.ccnt[l as usize] += 1;
                nc += 1;
            }
        });
    }
    let mut acc = 0u32;
    for c in sc.ccnt.iter_mut() {
        let x = *c;
        *c = acc;
        acc += x;
    }
    sc.citems.clear();
    sc.citems.resize(nc, CItem { ts: 0, camp: 0, cid: 0 });
    {
        let mut cur: Vec<u32> = sc.ccnt[..np].to_vec();
        let mut q = 0usize;
        for t in cls {
            t.each(p, |r| {
                let l = sc.lidx[q];
                q += 1;
                if l != u32::MAX {
                    let l = l as usize;
                    sc.citems[cur[l] as usize] = CItem { ts: r.ts, camp: r.camp, cid: r.cid };
                    cur[l] += 1;
                }
            });
        }
    }
    for l in 0..np {
        let (a, b) = (sc.ccnt[l] as usize, sc.ccnt[l + 1] as usize);
        if b - a > 1 {
            let group = &mut sc.citems[a..b];
            group.sort_unstable_by_key(|x| x.ts);
            let mut i = 0;
            while i < group.len() {
                let ts = group[i].ts;
                let mut end = i + 1;
                let mut best = i;
                while end < group.len() && group[end].ts == ts {
                    if ex.cmp(group[end].cid, group[best].cid) == Ordering::Less { best = end; }
                    end += 1;
                }
                group.swap(i, best);
                i = end;
            }
        }
    }

    let nd = ctx.nd;
    let organic = ctx.ncamp;
    let base = ctx.base;
    if sc.catrev.len() != ctx.ncat {
        sc.catrev = vec![0; ctx.ncat];
    }
    if sc.day_end.len() != nd + 1 {
        sc.day_end = vec![0; nd + 1];
    }
    let day_of = |start: u32| -> u32 { (floor_div(base + start as i64, DAY_MS) - ctx.day0) as u32 };
    for l in 0..np {
        let person = sc.persons[l];
        let ev = &sc.items[sc.cnt[l] as usize..sc.cnt[l + 1] as usize];
        let clicks = &sc.citems[sc.ccnt[l] as usize..sc.ccnt[l + 1] as usize];
        let ses = &mut sc.ses;
        let catrev = &mut sc.catrev;
        ses.clear();
        let mut cp = 0usize;
        let mut k = 0usize;
        let j = ev.len();
        let mut cur_day: u32 = u32::MAX;
        let mut row = URow { person, day: 0, sessions: 0, events: 0, revenue: 0, top: -1 };
        while k < j {
            let start = (ev[k] >> 32) as u32;
            let mut last = start;
            let mut nev = 0u32;
            let mut rev = 0u64;
            let day = day_of(start);
            if day != cur_day {
                if cur_day != u32::MAX {
                    finish_row(&mut row, catrev, &mut agg.rows);
                }
                cur_day = day;
                row = URow { person, day: (ctx.day0 + day as i64) as i32, sessions: 0, events: 0, revenue: 0, top: -1 };
            }
            while k < j {
                let it = ev[k];
                let ts = (it >> 32) as u32;
                if ts - last > GAP_MS {
                    break;
                }
                last = ts;
                nev += 1;
                let pc = it as u32;
                let price = (pc & 0x00FF_FFFF) as u64;
                if price > 0 {
                    rev += price;
                    catrev[(pc >> 24) as usize] += price;
                }
                k += 1;
            }
            row.sessions += 1;
            row.events += nev;
            row.revenue += rev;
            while cp < clicks.len() && clicks[cp].ts <= start {
                cp += 1;
            }
            let mut camp = organic;
            if cp > 0 {
                let hi = cp - 1;
                let t = clicks[hi].ts;
                if (start - t) as i64 <= WEEK_MS {
                    let mut q = hi;
                    while q > 0 && clicks[q - 1].ts == t {
                        q -= 1;
                    }
                    camp = clicks[q].camp as usize;
                }
            }
            let st = &mut agg.camp[camp * nd + day as usize];
            st[0] += 1;
            if rev > 0 {
                st[1] += 1;
            }
            st[2] += rev;
            ses.push(Ses { start, day });
        }
        finish_row(&mut row, catrev, &mut agg.rows);
        if !clicks.is_empty() {
            let day_end = &mut sc.day_end;
            let mut s = 0usize;
            for d in 0..nd {
                while s < ses.len() && ses[s].day as usize <= d {
                    s += 1;
                }
                day_end[d] = s;
            }
            let cbuf = &mut sc.cbuf;
            cbuf.clear();
            cbuf.extend(clicks.iter().map(|c| (c.camp, c.ts)));
            if cbuf.len() <= 32 { cbuf.sort_unstable(); }
            else { stable_campaign_group(cbuf, &mut sc.ctmp); }
            let mut a = 0usize;
            while a < cbuf.len() {
                let camp = cbuf[a].0 as usize;
                let mut b = a;
                while b < cbuf.len() && cbuf[b].0 as usize == camp {
                    b += 1;
                }
                let st = &mut agg.camp[camp * nd..(camp + 1) * nd];
                let mut lo = cbuf[a].1 as u64;
                let mut hi = lo + WEEK_MS as u64;
                for q in a + 1..b {
                    let t = cbuf[q].1 as u64;
                    if t <= hi + 1 {
                        hi = hi.max(t + WEEK_MS as u64);
                    } else {
                        add_assists(ses, day_end, lo, hi, st);
                        lo = t;
                        hi = t + WEEK_MS as u64;
                    }
                }
                add_assists(ses, day_end, lo, hi, st);
                a = b;
            }
        }
    }
}

/// Stable two-pass LSD radix grouping by 16-bit campaign number.
/// Clicks are already timestamp-ordered; stable grouping preserves that ordering.
fn stable_campaign_group(buf: &mut Vec<(u32,u32)>, tmp: &mut Vec<(u32,u32)>) {
    let n = buf.len();
    tmp.resize(n, (0, 0));
    let mut count = [0usize; 256];
    for &(c,_) in buf.iter() { count[(c & 255) as usize] += 1; }
    let mut pos = 0;
    for k in count.iter_mut() { let c=*k; *k=pos; pos+=c; }
    for &r in buf.iter() {
        let k=(r.0 & 255) as usize;
        tmp[count[k]]=r; count[k]+=1;
    }
    count.fill(0);
    for &(c,_) in tmp.iter() { count[((c>>8)&255) as usize]+=1; }
    pos=0;
    for k in count.iter_mut() { let c=*k; *k=pos; pos+=c; }
    for &r in tmp.iter() {
        let k=((r.0>>8)&255) as usize;
        buf[count[k]]=r; count[k]+=1;
    }
}

#[inline]
fn add_assists(ses: &[Ses], day_end: &[usize], l: u64, r: u64, st: &mut [[u64; 4]]) {
    let lo = ses.partition_point(|s| (s.start as u64) < l);
    let hi = ses.partition_point(|s| (s.start as u64) <= r);
    let mut s = lo;
    while s < hi {
        let d = ses[s].day as usize;
        let e = day_end[d].min(hi);
        st[d][3] += (e - s) as u64;
        s = e;
    }
}

#[inline]
fn finish_row(row: &mut URow, catrev: &mut [u64], rows: &mut Vec<URow>) {
    if row.sessions == 0 {
        return;
    }
    let mut top = -1i32;
    if row.revenue > 0 {
        let mut best = 0u64;
        for (c, v) in catrev.iter_mut().enumerate() {
            if *v > best {
                best = *v;
                top = c as i32;
            }
            *v = 0;
        }
    }
    row.top = top;
    rows.push(*row);
}


fn write_user_daily(path: &Path, rows: &[URow], ex: &Exotic, cats: &[Vec<u8>]) {
    let n = rows.len();
    let mut uid = pq::StrCol::with_capacity(n * 14);
    let mut day = Vec::with_capacity(n);
    let mut sessions = Vec::with_capacity(n);
    let mut events = Vec::with_capacity(n);
    let mut revenue = Vec::with_capacity(n);
    let mut top = pq::OptStrCol::with_capacity(n);
    let mut sbuf = Vec::with_capacity(16);
    for r in rows {
        sbuf.clear();
        ex.bytes(r.person, &mut sbuf);
        uid.push(&sbuf);
        day.push(r.day);
        sessions.push(r.sessions as i64);
        events.push(r.events as i64);
        revenue.push(r.revenue as i64);
        top.push(if r.top >= 0 { Some(&cats[r.top as usize][..]) } else { None });
    }
    let cols = vec![
        pq::Column { name: "user_id", kind: pq::Kind::Str, optional: false, data: uid.buf },
        pq::Column { name: "day", kind: pq::Kind::Date, optional: false, data: pq::i32_col(&day) },
        pq::Column { name: "sessions", kind: pq::Kind::I64, optional: false, data: pq::i64_col(&sessions) },
        pq::Column { name: "events", kind: pq::Kind::I64, optional: false, data: pq::i64_col(&events) },
        pq::Column { name: "revenue_cents", kind: pq::Kind::I64, optional: false, data: pq::i64_col(&revenue) },
        pq::Column { name: "top_category", kind: pq::Kind::Str, optional: true, data: top.finish() },
    ];
    pq::write_file(path, n, cols).expect("write user_daily");
}

// ------------------------------------------------------------------ main
fn main() {
    let data = PathBuf::from(std::env::var("DATA_DIR").unwrap_or_else(|_| "/data".into()));
    let outdir = PathBuf::from(std::env::var("OUTPUT_DIR").unwrap_or_else(|_| "/output".into()));
    let nthreads = std::env::var("NCPU").ok().and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)).max(1);
    let timing = std::env::var("SS_TIMING").is_ok();
    let t0 = std::time::Instant::now();
    let tick = |label: &str| {
        if timing { eprintln!("{:8.1} ms  {}", t0.elapsed().as_secs_f64()*1000.0, label); }
    };
    if let Ok(mode) = std::env::var("BENCH") { bench(&data, &mode); return; }
    let manifest = std::fs::read_to_string(data.join("manifest.json")).expect("manifest");
    let ws = manifest_value(&manifest, "window_start");
    let we = manifest_value(&manifest, "window_end");
    let base = ws - WEEK_MS;
    assert!(we - base < u32::MAX as i64 && we > ws);
    let day0 = floor_div(ws, DAY_MS);
    let nd = (floor_div(we - 1, DAY_MS) - day0 + 1) as usize;
    let prod_rows = read_pairs(&data.join("products.csv"));
    let mut cats: Vec<Vec<u8>> = prod_rows.iter().map(|(_, c)| c.clone()).collect();
    cats.push(b"unknown".to_vec());
    cats.sort(); cats.dedup();
    assert!(cats.len() < 256);
    let cat_idx: HashMap<Vec<u8>,u16> = cats.iter().enumerate().map(|(i,c)|(c.clone(),i as u16)).collect();
    let mut prods = Lookup::new(b"p");
    for (p,c) in &prod_rows { prods.insert(p,cat_idx[c]); }
    let unknown_cat = cat_idx[&b"unknown".to_vec()] as u32;
    let camp_rows = read_pairs(&data.join("campaigns.csv"));
    let mut camps = Lookup::new(b"c_");
    for (i,(c,_)) in camp_rows.iter().enumerate() { camps.insert(c,i as u16); }
    let ncamp = camp_rows.len();
    let ctx = Ctx { base, ws, we, day0, nd, ncat: cats.len(), ncamp, prods, camps, unknown_cat };
    let ex = Exotic::new();
    let ev_maps: Vec<&[u8]> = csv_files(&data.join("events")).iter().map(|p| map_file(p)).collect();
    let cl_maps: Vec<&[u8]> = csv_files(&data.join("clicks")).iter().map(|p| map_file(p)).collect();
    let ln_maps: Vec<&[u8]> = csv_files(&data.join("links")).iter().map(|p| map_file(p)).collect();
    let ev_chunks = make_chunks(&ev_maps, 4 << 20);
    let cl_chunks = make_chunks(&cl_maps, 4 << 20);
    let ln_chunks = make_chunks(&ln_maps, 1 << 20);
    let ev_bytes: usize = ev_maps.iter().map(|m|m.len()).sum();
    let cl_bytes: usize = cl_maps.iter().map(|m|m.len()).sum();
    tick("setup");
    let ln_next = AtomicUsize::new(0);
    let link_pairs: Vec<Vec<(u64,u64)>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..nthreads).map(|_| s.spawn(|| {
            let mut links = Vec::new();
            loop {
                let k = ln_next.fetch_add(1, AO::Relaxed);
                if k >= ln_chunks.len() { break; }
                let (fi,b,e) = ln_chunks[k];
                parse_links(&ex, ln_maps[fi], b, e, &mut links);
            }
            links
        })).collect();
        hs.into_iter().map(|h|h.join().unwrap()).collect()
    });
    let lm = std::sync::LazyLock::new({
        let ex = &ex;
        move || build_linkmap(&link_pairs, ex)
    });
    tick("link rows parsed");
    let ev_next = AtomicUsize::new(0);
    let cl_next = AtomicUsize::new(0);
    let est_ev = ev_bytes / 80 + 1024;
    let est_cl = cl_bytes / 40 + 1024;
    let ev_cap = est_ev * 5 / 4 / nthreads / P1 + 32;
    let cl_cap = est_cl * 5 / 4 / nthreads / P2 + 32;
    let pr_cap = est_ev * 5 / 4 / nthreads / P2 + 32;
    let parse = || {
        let mut evp = Parts::<EvRec>::alloc(ev_cap, P1);
        let mut clp = Parts::<CRec>::alloc(cl_cap, P2);
        let mut dc = DateCache::new();
        loop {
            let k = ev_next.fetch_add(1, AO::Relaxed);
            if k >= ev_chunks.len() { break; }
            let (fi,b,e) = ev_chunks[k];
            parse_events(&ctx,&ex,&mut dc,ev_maps[fi],b,e,&mut evp);
        }
        let ids = std::sync::LazyLock::force(&lm);
        loop {
            let k = cl_next.fetch_add(1, AO::Relaxed);
            if k >= cl_chunks.len() { break; }
            let (fi,b,e) = cl_chunks[k];
            parse_clicks_direct(&ctx,&ex,ids,&mut dc,cl_maps[fi],b,e,&mut clp);
        }
        (evp,clp)
    };
    let parsed: Vec<(Parts<EvRec>, Parts<CRec>)> = std::thread::scope(|scope| {
        let workers: Vec<_> = (1..nthreads).map(|_| scope.spawn(&parse)).collect();
        // Other workers can parse events while the calling thread resolves identities.
        std::sync::LazyLock::force(&lm);
        std::iter::once(parse()).chain(workers.into_iter().map(|h| h.join().unwrap())).collect()
    });
    let lm = std::sync::LazyLock::force(&lm);
    let (ev_parts,pc_parts):(Vec<_>,Vec<_>)=parsed.into_iter().unzip();
    tick("parse, identity-aware clicks");
    let p_next = AtomicUsize::new(0);
    let pe_parts: Vec<Parts<PRec>> = std::thread::scope(|s| {
        let hs:Vec<_> = (0..nthreads).map(|_|s.spawn(|| {
            let mut out = Parts::<PRec>::alloc(pr_cap,P2);
            let mut scratch = DedupScratch::default();
            loop {
                let p=p_next.fetch_add(1,AO::Relaxed);
                if p >= P1 {break;}
                dedup_direct(lm,&ev_parts,p,&mut scratch,&mut out);
            }
            out
        })).collect();
        hs.into_iter().map(|h|h.join().unwrap()).collect()
    });
    drop(ev_parts);
    tick("dedup => final person shards");

    // ---------------- stage 4: sessions, attribution, aggregation, user output
    let ud_dir = outdir.join("user_daily");
    let cd_dir = outdir.join("campaign_daily");
    std::fs::create_dir_all(&ud_dir).expect("mkdir");
    std::fs::create_dir_all(&cd_dir).expect("mkdir");
    // biggest partitions first
    let mut order: Vec<(usize, usize)> = (0..P2).map(|p| (pe_parts.iter().map(|t| t.count(p)).sum::<usize>(), p)).collect();
    order.sort_unstable_by(|a, b| b.cmp(a));
    let q_next = AtomicUsize::new(0);
    let aggs: Vec<Vec<[u64; 4]>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..nthreads)
            .map(|t| {
                let ctx = &ctx;
                let ex = &ex;
                let pe_parts = &pe_parts;
                let pc_parts = &pc_parts;
                let q_next = &q_next;
                let ud_dir = &ud_dir;
                let cats = &cats;
                let order = &order;
                s.spawn(move || {
                    let mut agg = Agg { rows: Vec::new(), camp: vec![[0u64; 4]; (ctx.ncamp + 1) * ctx.nd] };
                    let mut sc = Scratch::default();
                    loop {
                        let k = q_next.fetch_add(1, AO::Relaxed);
                        if k >= P2 {
                            break;
                        }
                        process_partition(ctx, ex, pe_parts, pc_parts, order[k].1, &mut sc, &mut agg);
                    }
                    if t == 0 || !agg.rows.is_empty() {
                        write_user_daily(&ud_dir.join(format!("part-{:03}.parquet", t)), &agg.rows, ex, cats);
                    }
                    agg.camp
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    tick("stage4 sessions");

    // ---------------- campaign_daily
    let mut tot = vec![[0u64; 4]; (ncamp + 1) * nd];
    for a in &aggs {
        for (x, y) in tot.iter_mut().zip(a.iter()) {
            for q in 0..4 {
                x[q] += y[q];
            }
        }
    }
    let mut cid = pq::StrCol::with_capacity(1 << 16);
    let mut chan = pq::StrCol::with_capacity(1 << 16);
    let mut day = Vec::new();
    let (mut s0, mut s1, mut s2, mut s3) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for c in 0..=ncamp {
        for d in 0..nd {
            let st = tot[c * nd + d];
            if st[0] == 0 && st[3] == 0 {
                continue;
            }
            if c == ncamp {
                cid.push(b"organic");
                chan.push(b"organic");
            } else {
                cid.push(&camp_rows[c].0);
                chan.push(&camp_rows[c].1);
            }
            day.push((day0 + d as i64) as i32);
            s0.push(st[0] as i64);
            s1.push(st[1] as i64);
            s2.push(st[2] as i64);
            s3.push(if c == ncamp { 0 } else { st[3] as i64 });
        }
    }
    let n = day.len();
    let cols = vec![
        pq::Column { name: "campaign_id", kind: pq::Kind::Str, optional: false, data: cid.buf },
        pq::Column { name: "channel", kind: pq::Kind::Str, optional: false, data: chan.buf },
        pq::Column { name: "day", kind: pq::Kind::Date, optional: false, data: pq::i32_col(&day) },
        pq::Column { name: "sessions", kind: pq::Kind::I64, optional: false, data: pq::i64_col(&s0) },
        pq::Column { name: "purchasing_sessions", kind: pq::Kind::I64, optional: false, data: pq::i64_col(&s1) },
        pq::Column { name: "revenue_cents", kind: pq::Kind::I64, optional: false, data: pq::i64_col(&s2) },
        pq::Column { name: "assisted_sessions", kind: pq::Kind::I64, optional: false, data: pq::i64_col(&s3) },
    ];
    pq::write_file(&cd_dir.join("part-000.parquet"), n, cols).expect("write campaign_daily");
    tick("done");
    std::process::exit(0);
}

fn bench(data: &Path, mode: &str) {
    if mode == "parse" || mode == "parse_np" {
        let manifest = std::fs::read_to_string(data.join("manifest.json")).unwrap();
        let ws = manifest_value(&manifest, "window_start");
        let we = manifest_value(&manifest, "window_end");
        let prod_rows = read_pairs(&data.join("products.csv"));
        let mut prods = Lookup::new(b"p");
        for (i, (p, _)) in prod_rows.iter().enumerate() { prods.insert(p, (i % 20) as u16); }
        let ctx = Ctx { base: ws - WEEK_MS, ws, we, day0: 0, nd: 1, ncat: 21, ncamp: 0, prods, camps: Lookup::new(b"c_"), unknown_cat: 20 };
        let ev_maps: Vec<&[u8]> = csv_files(&data.join("events")).iter().map(|p| map_file(p)).collect();
        let ex = Exotic::new();
        let mut dc = DateCache::new();
        let total: usize = ev_maps.iter().map(|m| m.len()).sum();
        let t = std::time::Instant::now();
        let mut evp: Parts<EvRec> = Parts::alloc(total / 80 * 5 / 4 / P1 + 32, P1);
        eprintln!("alloc {:?}", t.elapsed());
        for m in &ev_maps { parse_events(&ctx, &ex, &mut dc, m, 0, m.len(), &mut evp); if mode == "parse_np" { evp.clear(); } }
        eprintln!("{} {:?} n={}", mode, t.elapsed(), (0..P1).map(|p| evp.count(p)).sum::<usize>());
        return;
    }
    let ev_maps: Vec<&[u8]> = csv_files(&data.join("events")).iter().map(|p| map_file(p)).collect();
    let ex = Exotic::new();
    let mut dc = DateCache::new();
    let t = std::time::Instant::now();
    let mut cnt = 0usize;
    let mut acc = 0u64;
    for m in &ev_maps {
        match mode {
            "fields" => scan_lines::<7>(m, 0, m.len(), &mut |f: &[&[u8]; 7]| {
                cnt += 1;
                acc += f[2].len() as u64;
            }),
            "ts" => scan_lines::<7>(m, 0, m.len(), &mut |f: &[&[u8]; 7]| {
                cnt += 1;
                acc = acc.wrapping_add(parse_ts_fast(f[2], &mut dc).unwrap_or(0) as u64).wrapping_add(parse_ts_fast(f[3], &mut dc).unwrap_or(0) as u64);
            }),
            "keycheck" => scan_lines::<7>(m, 0, m.len(), &mut |f: &[&[u8]; 7]| {
                let lim = m.as_ptr() as usize + m.len();
                for k in 0..7 {
                    let a = enc_key(f[k]);
                    let b = if (f[k].as_ptr() as usize) + 16 <= lim { unsafe { enc_key16(f[k]) } } else { a };
                    if a != b { cnt += 1; if cnt < 5 { eprintln!("mismatch {:?} {:?} {:?}", String::from_utf8_lossy(f[k]), a, b); } }
                    acc += 1;
                }
            }),
            "keys2" => scan_lines::<7>(m, 0, m.len(), &mut |f: &[&[u8]; 7]| {
                let lim = m.as_ptr() as usize + m.len();
                cnt += 1;
                acc = acc.wrapping_add(ex.key_in(f[0], lim)).wrapping_add(ex.key_in(f[1], lim));
            }),
            "keys" => scan_lines::<7>(m, 0, m.len(), &mut |f: &[&[u8]; 7]| {
                cnt += 1;
                acc = acc.wrapping_add(ex.key(f[0])).wrapping_add(ex.key(f[1]));
            }),
            "masks" => unsafe {
                use std::arch::x86_64::*;
                let vn = _mm256_set1_epi8(b'\n' as i8);
                let mut o = 0;
                while o + 64 <= m.len() {
                    let a = _mm256_loadu_si256(m.as_ptr().add(o) as *const __m256i);
                    let b = _mm256_loadu_si256(m.as_ptr().add(o + 32) as *const __m256i);
                    let x = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, vn)) as u32).count_ones() + (_mm256_movemask_epi8(_mm256_cmpeq_epi8(b, vn)) as u32).count_ones();
                    acc += x as u64;
                    o += 64;
                }
            },
            "memcpy" => { let mut v = vec![0u8; m.len()]; v.copy_from_slice(m); acc += v[v.len()/2] as u64; }
            _ => {}
        }
    }
    eprintln!("{} {:?} cnt={} acc={}", mode, t.elapsed(), cnt, acc);
}

#[cfg(test)]
mod event_dedup_tests {
    use super::*;

    #[test]
    fn event_keys_are_distinct_across_packed_and_fallback_encodings() {
        let keys = Exotic::new();
        let mut seen = HashMap::new();
        let mut check = |id: &[u8]| {
            let mut input = [0u8; 32];
            input[..id.len()].copy_from_slice(id);
            let field = &input[..id.len()];
            let start = input.as_ptr() as usize;
            let key = keys.event_key(field, start + input.len());
            assert_eq!(key, keys.event_key(field, start + field.len()));
            if let Some(previous) = seen.insert(key, id.to_vec()) {
                assert_eq!(previous, id);
            }
        };
        for len in 1..=12 {
            check(&b"abcdefghijkl"[..len]);
        }
        for at in 0..9 {
            for byte in 0..=255 {
                let mut id = *b"e12345678";
                id[at] = byte;
                check(&id);
            }
        }
    }

    #[test]
    fn colliding_event_ids_and_spilled_duplicates_keep_the_earliest_copy() {
        // Put three distinct IDs in one 64-bucket filter slot and one in another.
        let bucket = |key: u64| key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - P1_BITS - 6);
        let ids: Vec<_> = (1u64..).filter(|&key| bucket(key) == 0).take(3).collect();
        let singleton = (1u64..).find(|&key| bucket(key) == 1).unwrap();
        let event = |eid, user, ingest, ok| EvRec { eid, user, ts: 1, ingest, pc: 0, ok };
        let mut a = Parts::alloc(1, 1);
        let mut b = Parts::alloc(1, 1);
        a.push(0, event(ids[0], 11, 2, 1));
        a.push(0, event(ids[1], 22, 5, 1));
        a.push(0, event(singleton, 33, 1, 1));
        b.push(0, event(ids[0], 44, 1, 1));
        b.push(0, event(ids[1], 55, 3, 0));
        b.push(0, event(ids[2], 66, 1, 1));
        let inputs = [a, b];
        let mut scratch = DedupScratch::default();
        let mut out = Parts::alloc(1, P2);
        dedup_direct(&LinkMap::empty(), &inputs, 0, &mut scratch, &mut out);
        let mut users = Vec::new();
        for p in 0..P2 {
            out.each(p, |r| users.push(r.person));
        }
        users.sort_unstable();
        assert_eq!(users, [33, 44, 66]);
    }
}
