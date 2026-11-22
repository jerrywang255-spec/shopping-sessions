//! Shopping Sessions pipeline: std-only Rust, multi-threaded, hand-written CSV and Parquet.
mod pq;

use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::{BuildHasher, RandomState};
use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering as AO};
use std::sync::{Mutex, OnceLock};

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
fn for_lines<'a>(m: &'a [u8], b: usize, e: usize, mut f: impl FnMut(&'a [u8])) {
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
static NO512: OnceLock<bool> = OnceLock::new();
#[inline(always)]
fn scan_lines<'a, const N: usize>(m: &'a [u8], b: usize, e: usize, emit: &mut impl FnMut(&[&'a [u8]; N])) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("bmi2") && is_x86_feature_detected!("pclmulqdq") && !*NO512.get_or_init(|| std::env::var_os("SS_NO512").is_some()) {
            if is_x86_feature_detected!("avx512vbmi2") && std::env::var_os("SS_NOVBMI").is_none() {
                unsafe { scan_lines_avx512_vbmi2::<N>(m, b, e, emit) };
                return;
            }
            unsafe { scan_lines_avx512::<N>(m, b, e, emit) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("pclmulqdq") {
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

#[inline(always)]
fn pxor64(x: u64) -> u64 {
    // carry-less multiply by all-ones = prefix xor (callers run with pclmulqdq enabled)
    unsafe {
        use std::arch::x86_64::*;
        let r = _mm_clmulepi64_si128(_mm_set_epi64x(0, x as i64), _mm_set1_epi8(-1), 0);
        _mm_cvtsi128_si64(r) as u64
    }
}
#[allow(dead_code)]
#[inline(always)]
fn pxor64_shift(mut x: u64) -> u64 {
    x ^= x << 1;
    x ^= x << 2;
    x ^= x << 4;
    x ^= x << 8;
    x ^= x << 16;
    x ^= x << 32;
    x
}

/// One line from its masks (bits relative to the line start `s`, already cut at the line end `pl`):
/// commas c0|c1<<64 (exactly N-1 of them), quotes q0|q1<<64. Validates the quoting without branches
/// on the (common) quoted case and strips whole-field quotes.
#[inline(always)]
unsafe fn emit_line_masks<'a, const N: usize>(m: &'a [u8], s: usize, pl: usize, c0: u64, c1: u64, q0: u64, q1: u64, emit: &mut impl FnMut(&[&'a [u8]; N])) {
    // field boundaries
    let mut ends = [0usize; N];
    let mut x0 = c0;
    let mut x1 = c1;
    for k in 0..N - 1 {
        let z0 = x0 == 0;
        let pos = if z0 { 64 + x1.trailing_zeros() as usize } else { x0.trailing_zeros() as usize };
        ends[k] = pos;
        x0 &= x0.wrapping_sub(1);
        x1 = if z0 { x1 & x1.wrapping_sub(1) } else { x1 };
    }
    ends[N - 1] = pl;
    emit_line_ends(m, s, pl, &ends, c0, c1, q0, q1, emit)
}

#[inline(always)]
unsafe fn emit_line_ends<'a, const N: usize>(m: &'a [u8], s: usize, pl: usize, ends: &[usize; N], c0: u64, c1: u64, q0: u64, q1: u64, emit: &mut impl FnMut(&[&'a [u8]; N])) {
    let base = m.as_ptr();
    // quoting: opening quotes must start a field, closing quotes must end one, no comma inside
    // quotes, and every quote closed. Starts S = (c << 1) | 1, last bytes E = (c >> 1) | bit(pl - 1).
    let px0 = pxor64(q0);
    let px1 = pxor64(q1) ^ (0u64.wrapping_sub((q0.count_ones() & 1) as u64));
    let s0 = (c0 << 1) | 1;
    let s1 = (c1 << 1) | (c0 >> 63);
    let lp = pl.wrapping_sub(1);
    let e0 = (c0 >> 1) | (c1 << 63) | if lp < 64 { 1u64 << lp } else { 0 };
    let e1 = (c1 >> 1) | if lp >= 64 && lp < 128 { 1u64 << (lp - 64) } else { 0 };
    let bad = ((q0 & px0) & !s0) | ((q1 & px1) & !s1) | ((q0 & !px0) & !e0) | ((q1 & !px1) & !e1) | (c0 & px0) | (c1 & px1);
    let open_end = ((q0.count_ones() + q1.count_ones()) & 1) as u64;
    if (bad | open_end) != 0 {
        return;
    }
    let mut f: [&[u8]; N] = [&[]; N];
    let mut a = s;
    for k in 0..N {
        let t = s + ends[k];
        let q = (*base.add(a) == b'"') as usize & ((t > a) as usize);
        f[k] = std::slice::from_raw_parts(base.add(a + q), t - a - 2 * q);
        a = t + 1;
    }
    emit(&f);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,popcnt,pclmulqdq")]
unsafe fn scan_lines_avx2<'a, const N: usize>(m: &'a [u8], b: usize, e: usize, emit: &mut impl FnMut(&[&'a [u8]; N])) {
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
        if s + 2048 < len {
            _mm_prefetch(p.add(2048) as *const i8, _MM_HINT_T0);
        }
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
        let cm = (mv(a0, vc) | (mv(a1, vc) << 32) | (mv(a2, vc) << 64) | (mv(a3, vc) << 96)) & below;
        if cm.count_ones() as usize == N - 1 {
            let qm = (mv(a0, vq) | (mv(a1, vq) << 32) | (mv(a2, vq) << 64) | (mv(a3, vq) << 96)) & below;
            emit_line_masks(m, s, pl, cm as u64, (cm >> 64) as u64, qm as u64, (qm >> 64) as u64, emit);
        }
        let _ = &mut ends;
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
#[target_feature(enable = "avx512f,avx512bw,avx2,bmi1,bmi2,popcnt,pclmulqdq")]
unsafe fn scan_lines_avx512<'a, const N: usize>(m: &'a [u8], b: usize, e: usize, emit: &mut impl FnMut(&[&'a [u8]; N])) {
    scan_lines_512g::<N, false>(m, b, e, emit)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi2,avx2,bmi1,bmi2,popcnt,pclmulqdq")]
unsafe fn scan_lines_avx512_vbmi2<'a, const N: usize>(m: &'a [u8], b: usize, e: usize, emit: &mut impl FnMut(&[&'a [u8]; N])) {
    scan_lines_512g::<N, true>(m, b, e, emit)
}
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn scan_lines_512g<'a, const N: usize, const V: bool>(m: &'a [u8], b: usize, e: usize, emit: &mut impl FnMut(&[&'a [u8]; N])) {
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
        if s + 2048 < len {
            _mm_prefetch(p.add(2048) as *const i8, _MM_HINT_T0);
        }
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
            let q0 = _bzhi_u64(_mm512_cmpeq_epi8_mask(a0, vq), pl as u32);
            let q1 = _bzhi_u64(_mm512_cmpeq_epi8_mask(a1, vq), pl.saturating_sub(64) as u32);
            if V {
                // comma offsets as bytes: compress an index vector by the comma masks
                let iota = _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
                let p0 = _mm512_maskz_compress_epi8(c0, iota);
                let p1 = _mm512_maskz_compress_epi8(c1, _mm512_add_epi8(iota, _mm512_set1_epi8(64)));
                let mut buf = [0u8; 192];
                _mm512_storeu_si512(buf.as_mut_ptr() as *mut _, p0);
                _mm512_storeu_si512(buf.as_mut_ptr().add(c0.count_ones() as usize) as *mut _, p1);
                let mut ends = [0usize; N];
                for k in 0..N - 1 {
                    ends[k] = buf[k] as usize;
                }
                ends[N - 1] = pl;
                emit_line_ends(m, s, pl, &ends, c0, c1, q0, q1, emit);
            } else {
                emit_line_masks(m, s, pl, c0, c1, q0, q1, emit);
            }
        }
        let _ = &mut ends;
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

/// MADV_HUGEPAGE on the whole 4 KB pages inside an untouched heap buffer.
fn advise_huge(p: *mut u8, len: usize) {
    let start = (p as usize + 4095) & !4095;
    let end = (p as usize + len) & !4095;
    if end > start {
        unsafe { madvise(start as *mut u8, end - start, 14) };
    }
}

/// Bytes per block. Partitions are chains of blocks, so memory is touched only as it fills,
/// and a consumed partition's blocks can be recycled for the next stage's output.
const BLK: usize = 8192;
const RESERVE: usize = 1 << 30;

/// Block allocator: a free list, then a bump pointer through reserved huge-page memory.
struct Pool {
    bump: *mut u8,
    end: *mut u8,
    free: Vec<*mut u8>,
}
unsafe impl Send for Pool {}
impl Pool {
    fn new() -> Self {
        Pool { bump: std::ptr::null_mut(), end: std::ptr::null_mut(), free: Vec::new() }
    }
    #[cold]
    fn refill(&mut self) {
        unsafe {
            let p = mmap(std::ptr::null_mut(), RESERVE, 3, 0x02 | 0x20 | 0x4000, -1, 0);
            if p as isize == -1 {
                panic!("mmap failed");
            }
            madvise(p, RESERVE, 14); // MADV_HUGEPAGE
            self.bump = p;
            self.end = p.add(RESERVE);
        }
    }
    #[inline(always)]
    fn get(&mut self) -> *mut u8 {
        if let Some(b) = self.free.pop() {
            return b;
        }
        if self.bump == self.end {
            self.refill();
        }
        let b = self.bump;
        self.bump = unsafe { b.add(BLK) };
        b
    }
}

/// P partitions, each a chain of blocks taken from one pool (owned by this set, never freed).
struct Parts<T: Copy> {
    cur: Vec<*mut T>,
    lim: Vec<*mut T>,
    blocks: Vec<Vec<*mut T>>,
    pool: *mut Pool,
}
unsafe impl<T: Copy + Send> Send for Parts<T> {}
unsafe impl<T: Copy + Sync> Sync for Parts<T> {}
impl<T: Copy> Parts<T> {
    const PER: usize = BLK / std::mem::size_of::<T>();
    /// `_count_per_part` is a sizing hint kept for the callers; memory grows on demand.
    fn alloc(_count_per_part: usize, np: usize) -> Self {
        let pool = Box::into_raw(Box::new(Pool::new()));
        Parts { cur: vec![std::ptr::null_mut(); np], lim: vec![std::ptr::null_mut(); np], blocks: (0..np).map(|_| Vec::new()).collect(), pool }
    }
    #[cold]
    #[inline(never)]
    fn grow(&mut self, p: usize) {
        let b = unsafe { (*self.pool).get() } as *mut T;
        self.blocks[p].push(b);
        self.cur[p] = b;
        self.lim[p] = unsafe { b.add(Self::PER) };
    }
    #[inline(always)]
    fn push(&mut self, p: usize, x: T) {
        unsafe {
            if *self.cur.get_unchecked(p) == *self.lim.get_unchecked(p) {
                self.grow(p);
            }
            let c = *self.cur.get_unchecked(p);
            c.write(x);
            *self.cur.get_unchecked_mut(p) = c.add(1);
        }
    }
    /// Room for `r` more records in the current block (allocating one if needed); returns the write position.
    #[inline(always)]
    fn reserve(&mut self, p: usize, r: usize) -> *mut T {
        unsafe {
            if (*self.lim.get_unchecked(p) as usize) - (*self.cur.get_unchecked(p) as usize) < r * std::mem::size_of::<T>() {
                self.grow(p);
            }
            let c = *self.cur.get_unchecked(p);
            *self.cur.get_unchecked_mut(p) = c.add(r);
            c
        }
    }
    #[inline(always)]
    fn count(&self, p: usize) -> usize {
        let n = self.blocks[p].len();
        if n == 0 {
            return 0;
        }
        (n - 1) * Self::PER + (self.cur[p] as usize - self.blocks[p][n - 1] as usize) / std::mem::size_of::<T>()
    }
    /// The partition as slices, one per block.
    #[inline(always)]
    fn runs(&self, p: usize) -> impl Iterator<Item = &[T]> + '_ {
        let bl = &self.blocks[p];
        let n = bl.len();
        let last = self.cur[p];
        bl.iter().enumerate().map(move |(i, &b)| {
            let len = if i + 1 == n { (last as usize - b as usize) / std::mem::size_of::<T>() } else { Self::PER };
            unsafe { std::slice::from_raw_parts(b as *const T, len) }
        })
    }
    #[inline(always)]
    fn each(&self, p: usize, mut f: impl FnMut(&T)) {
        for sl in self.runs(p) {
            for x in sl {
                f(x);
            }
        }
    }
    /// Hand partition p's blocks to `pool` (the partition is empty afterwards).
    fn release(&mut self, p: usize, pool: *mut Pool) {
        for b in self.blocks[p].drain(..) {
            unsafe { (*pool).free.push(b as *mut u8) };
        }
        self.cur[p] = std::ptr::null_mut();
        self.lim[p] = std::ptr::null_mut();
    }
    fn clear(&mut self) {
        let pool = self.pool;
        for p in 0..self.blocks.len() {
            self.release(p, pool);
        }
    }
}

/// Software write-combining in front of `Parts`: records gather in one 64-byte line per partition
/// and go to memory with non-temporal stores (no read-for-ownership, no cache pollution).
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Line([u8; 64]);
struct Wc {
    stage: Vec<Line>,
    fill: Vec<u8>,
}
impl Wc {
    fn new(np: usize) -> Self {
        Wc { stage: vec![Line([0; 64]); np], fill: vec![0; np] }
    }
    #[inline(always)]
    fn push<T: Copy>(&mut self, parts: &mut Parts<T>, p: usize, x: T) {
        let sz = std::mem::size_of::<T>();
        let r = 64 / sz;
        unsafe {
            let f = *self.fill.get_unchecked(p) as usize;
            let line = self.stage.get_unchecked_mut(p).0.as_mut_ptr();
            (line.add(f * sz) as *mut T).write(x);
            if f + 1 < r {
                *self.fill.get_unchecked_mut(p) = (f + 1) as u8;
                return;
            }
            *self.fill.get_unchecked_mut(p) = 0;
            let dst = parts.reserve(p, r) as *mut u8;
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                let a = _mm256_load_si256(line as *const __m256i);
                let b = _mm256_load_si256(line.add(32) as *const __m256i);
                _mm256_stream_si256(dst as *mut __m256i, a);
                _mm256_stream_si256(dst.add(32) as *mut __m256i, b);
            }
        }
    }
    fn finish<T: Copy>(&mut self, parts: &mut Parts<T>) {
        let sz = std::mem::size_of::<T>();
        for p in 0..self.fill.len() {
            let f = self.fill[p] as usize;
            for k in 0..f {
                let x = unsafe { (self.stage[p].0.as_ptr().add(k * sz) as *const T).read() };
                parts.push(p, x);
            }
            self.fill[p] = 0;
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_sfence();
        }
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

const INTERN_SHARD_BITS: u32 = 8;
const INTERN_SHARDS: usize = 1 << INTERN_SHARD_BITS;

struct InternedId {
    bytes: Box<[u8]>,
    next: Option<std::num::NonZeroUsize>,
}

#[derive(Default)]
struct InternShard {
    heads: HashMap<u64, std::num::NonZeroUsize>,
    entries: Vec<InternedId>,
}

impl InternShard {
    fn intern(&mut self, hash: u64, s: &[u8]) -> usize {
        let head = self.heads.get(&hash).copied();
        let mut current = head;
        loop {
            let Some(entry_index) = current else {
                let index = self.entries.len();
                self.entries.push(InternedId { bytes: s.into(), next: head });
                self.heads.insert(hash, std::num::NonZeroUsize::new(index + 1).unwrap());
                break index;
            };
            let index = entry_index.get() - 1;
            let entry = &self.entries[index];
            // The full byte string decides equality, even when hashes collide.
            if entry.bytes.as_ref() == s {
                break index;
            }
            current = entry.next;
        }
    }
}

#[cfg(test)]
mod intern_tests {
    use super::InternShard;

    #[test]
    fn shared_hash_preserves_distinct_opaque_ids() {
        let mut shard = InternShard::default();
        let ids = [
            &b"e00112233445566778899aabbccddeeff"[..],
            &b"e00112233445566778899aabbccddeefF"[..],
            "id-ÃŽÂ»".as_bytes(),
        ];
        for (index, id) in ids.iter().enumerate() {
            assert_eq!(shard.intern(42, id), index);
        }
        for (index, id) in ids.iter().enumerate().rev() {
            assert_eq!(shard.intern(42, id), index);
            assert_eq!(shard.entries[index].bytes.as_ref(), *id);
        }
    }
}

/// Lock-free table of long IDs that live in the (never unmapped) input files: a slot holds a
/// nonzero 64-bit hash and a pointer|length; equality is decided by the bytes.
struct LongIds {
    slots: Vec<[std::sync::atomic::AtomicU64; 2]>,
    mask: usize,
    count: AtomicUsize,
    full: std::sync::atomic::AtomicBool,
}
const LONG_TAG: u64 = 1 << 62;
const HEX_TAG: u64 = 1 << 61;

/// IDs of one byte + 32 lowercase hex digits, stored exactly as (prefix, 128-bit value): equality
/// never touches the input bytes again. Slot: [meta = 1 | prefix << 8 (0 empty, 2 busy), lo, hi, ptr].
struct HexIds {
    slots: Vec<[std::sync::atomic::AtomicU64; 4]>,
    mask: usize,
    count: AtomicUsize,
    full: std::sync::atomic::AtomicBool,
}
impl HexIds {
    fn new(cap: usize) -> Self {
        let cap = cap.next_power_of_two().max(1024);
        let mut slots = Vec::with_capacity(cap);
        slots.resize_with(cap, || Default::default());
        HexIds { slots, mask: cap - 1, count: AtomicUsize::new(0), full: std::sync::atomic::AtomicBool::new(false) }
    }
    #[inline(always)]
    fn home(&self, s: &[u8], lo: u64, hi: u64) -> usize {
        let meta = 1 | ((s[0] as u64) << 8);
        let h = fold_mul(lo ^ meta.wrapping_mul(0x9E37_79B9_7F4A_7C15), hi ^ 0x243F_6A88_85A3_08D3);
        (h >> 7) as usize & self.mask
    }
    #[inline(always)]
    fn prefetch(&self, i: usize) {
        unsafe { std::arch::x86_64::_mm_prefetch(self.slots.as_ptr().add(i) as *const i8, std::arch::x86_64::_MM_HINT_T0) };
    }
    #[inline]
    fn slot(&self, s: &[u8], lo: u64, hi: u64) -> Option<usize> {
        self.slot_at(s, lo, hi, self.home(s, lo, hi))
    }
    #[inline]
    fn slot_at(&self, s: &[u8], lo: u64, hi: u64, home: usize) -> Option<usize> {
        let meta = 1 | ((s[0] as u64) << 8);
        let mut i = home;
        let mut probes = 0;
        loop {
            let cell = &self.slots[i];
            let mut m = cell[0].load(AO::Acquire);
            if m == 0 {
                if self.full.load(AO::Relaxed) {
                    return None;
                }
                match cell[0].compare_exchange(0, 2, AO::AcqRel, AO::Acquire) {
                    Ok(_) => {
                        cell[1].store(lo, AO::Relaxed);
                        cell[2].store(hi, AO::Relaxed);
                        cell[3].store(s.as_ptr() as u64, AO::Relaxed);
                        cell[0].store(meta, AO::Release);
                        if self.count.fetch_add(1, AO::Relaxed) * 10 > self.slots.len() * 6 {
                            self.full.store(true, AO::Relaxed);
                        }
                        return Some(i);
                    }
                    Err(x) => m = x,
                }
            }
            while m == 2 {
                std::hint::spin_loop();
                m = cell[0].load(AO::Acquire);
            }
            if m == meta && cell[1].load(AO::Relaxed) == lo && cell[2].load(AO::Relaxed) == hi {
                return Some(i);
            }
            i = (i + 1) & self.mask;
            probes += 1;
            if probes > 1024 {
                return None;
            }
        }
    }
    fn bytes(&self, i: usize) -> &'static [u8] {
        let p = self.slots[i][3].load(AO::Acquire);
        unsafe { std::slice::from_raw_parts(p as *const u8, 33) }
    }
    fn len(&self) -> usize {
        self.count.load(AO::Relaxed)
    }
}

#[inline(always)]
fn fold_mul(a: u64, b: u64) -> u64 {
    let m = (a as u128) * (b as u128);
    (m as u64) ^ ((m >> 64) as u64)
}
/// Hash of an ID of 16+ bytes (first and last 16 bytes, and the length).
#[inline(always)]
fn long_hash(s: &[u8]) -> u64 {
    let n = s.len();
    unsafe {
        let p = s.as_ptr();
        let a = (p as *const u64).read_unaligned();
        let b = (p.add(8) as *const u64).read_unaligned();
        let c = (p.add(n - 16) as *const u64).read_unaligned();
        let d = (p.add(n - 8) as *const u64).read_unaligned();
        let h = fold_mul(a ^ 0x243F_6A88_85A3_08D3, b ^ 0x1319_8A2E_0370_7344) ^ fold_mul(c ^ 0xA409_3822_299F_31D0, d ^ (n as u64).wrapping_mul(0x082E_FA98_EC4E_6C89));
        fold_mul(h, 0x9E37_79B9_7F4A_7C15) | 1
    }
}

/// Byte equality of two strings of the same length n >= 16 (16-byte chunks, last one overlapping).
#[inline(always)]
unsafe fn eq_long(a: *const u8, b: *const u8, n: usize) -> bool {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 16 < n {
        let x = _mm_loadu_si128(a.add(i) as *const __m128i);
        let y = _mm_loadu_si128(b.add(i) as *const __m128i);
        if _mm_movemask_epi8(_mm_cmpeq_epi8(x, y)) != 0xFFFF {
            return false;
        }
        i += 16;
    }
    let x = _mm_loadu_si128(a.add(n - 16) as *const __m128i);
    let y = _mm_loadu_si128(b.add(n - 16) as *const __m128i);
    _mm_movemask_epi8(_mm_cmpeq_epi8(x, y)) == 0xFFFF
}

/// Exact 128-bit value of 32 lowercase hex digits at p, or None.
#[inline(always)]
unsafe fn hex128(p: *const u8) -> Option<(u64, u64)> {
    use std::arch::x86_64::*;
    let v = _mm256_loadu_si256(p as *const __m256i);
    let d = _mm256_sub_epi8(v, _mm256_set1_epi8(b'0' as i8));
    let a = _mm256_sub_epi8(v, _mm256_set1_epi8((b'a' - 10) as i8));
    // digit: (v - '0') as unsigned < 10; letter: (v - 'a' + 10) in 10..16
    let is_d = _mm256_cmpeq_epi8(_mm256_min_epu8(d, _mm256_set1_epi8(9)), d);
    let a10 = _mm256_sub_epi8(a, _mm256_set1_epi8(10));
    let is_a = _mm256_cmpeq_epi8(_mm256_min_epu8(a10, _mm256_set1_epi8(5)), a10);
    if _mm256_movemask_epi8(_mm256_or_si256(is_d, is_a)) != -1 {
        return None;
    }
    let nib = _mm256_blendv_epi8(a, d, is_d);
    // pairs (hi nibble first) -> bytes
    let w = _mm256_maddubs_epi16(nib, _mm256_set1_epi16(0x0110));
    let packed = _mm256_packus_epi16(w, w); // per 128-bit lane: 8 bytes in the low half
    let lo = _mm256_extract_epi64(packed, 0) as u64;
    let hi = _mm256_extract_epi64(packed, 2) as u64;
    Some((lo, hi))
}

impl LongIds {
    fn new(cap: usize) -> Self {
        let cap = cap.next_power_of_two().max(1024);
        let mut slots = Vec::with_capacity(cap);
        slots.resize_with(cap, || [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)]);
        LongIds { slots, mask: cap - 1, count: AtomicUsize::new(0), full: std::sync::atomic::AtomicBool::new(false) }
    }
    /// Slot of `s` (len >= 16, inside a static mapping), or None when the table is too full.
    #[inline]
    fn slot(&self, s: &[u8]) -> Option<usize> {
        if s.len() > 0xFFFF {
            return None;
        }
        let h = long_hash(s);
        let packed = (s.as_ptr() as u64) | ((s.len() as u64) << 48);
        let mut i = (h >> 20) as usize & self.mask;
        let mut probes = 0;
        loop {
            let cell = &self.slots[i];
            let mut cur = cell[0].load(AO::Acquire);
            if cur == 0 {
                if self.full.load(AO::Relaxed) {
                    return None;
                }
                match cell[0].compare_exchange(0, h, AO::AcqRel, AO::Acquire) {
                    Ok(_) => {
                        cell[1].store(packed, AO::Release);
                        if self.count.fetch_add(1, AO::Relaxed) * 10 > self.slots.len() * 6 {
                            self.full.store(true, AO::Relaxed);
                        }
                        return Some(i);
                    }
                    Err(x) => cur = x,
                }
            }
            if cur == h {
                let mut v = cell[1].load(AO::Acquire);
                while v == 0 {
                    std::hint::spin_loop();
                    v = cell[1].load(AO::Acquire);
                }
                if (v >> 48) as usize == s.len() && unsafe { eq_long((v & ((1 << 48) - 1)) as *const u8, s.as_ptr(), s.len()) } {
                    return Some(i);
                }
            }
            i = (i + 1) & self.mask;
            probes += 1;
            if probes > 1024 {
                return None;
            }
        }
    }
    fn len(&self) -> usize {
        self.count.load(AO::Relaxed)
    }
    fn bytes(&self, i: usize) -> &'static [u8] {
        let v = self.slots[i][1].load(AO::Acquire);
        unsafe { std::slice::from_raw_parts((v & ((1 << 48) - 1)) as *const u8, (v >> 48) as usize) }
    }
}

/// Intern opaque IDs using exact byte equality and independent shard locks.
struct Exotic {
    hash: RandomState,
    shards: Box<[Mutex<InternShard>]>,
    long: OnceLock<LongIds>,
    hex: OnceLock<HexIds>,
}
impl Exotic {
    fn new() -> Self {
        Exotic {
            hash: RandomState::new(),
            shards: (0..INTERN_SHARDS).map(|_| Mutex::new(InternShard::default())).collect(),
            long: OnceLock::new(),
            hex: OnceLock::new(),
        }
    }
    /// Enable the lock-free long-ID table (IDs must point into static input mappings from now on).
    fn init_long(&self, cap: usize) {
        let _ = self.long.set(LongIds::new(cap));
        let _ = self.hex.set(HexIds::new(cap));
    }
    #[inline]
    fn key(&self, s: &[u8]) -> u64 {
        match enc_key(s) {
            Some(k) => k,
            None => self.intern(s),
        }
    }
    #[inline]
    fn intern(&self, s: &[u8]) -> u64 {
        if s.len() == 33 {
            if let Some(t) = self.hex.get() {
                if let Some((lo, hi)) = unsafe { hex128(s.as_ptr().add(1)) } {
                    if let Some(i) = t.slot(s, lo, hi) {
                        return EXOTIC | LONG_TAG | HEX_TAG | i as u64;
                    }
                }
            }
        }
        if s.len() >= 16 {
            if let Some(t) = self.long.get() {
                if let Some(i) = t.slot(s) {
                    return EXOTIC | LONG_TAG | i as u64;
                }
            }
        }
        self.intern_locked(s)
    }
    #[cold]
    fn intern_locked(&self, s: &[u8]) -> u64 {
        let hash = self.hash.hash_one(s);
        let shard = hash as usize & (INTERN_SHARDS - 1);
        let mut shard_data = self.shards[shard].lock().unwrap();
        let index = shard_data.intern(hash, s);
        EXOTIC | ((index as u64) << INTERN_SHARD_BITS) | shard as u64
    }
    fn bytes(&self, k: u64, out: &mut Vec<u8>) {
        if k & (EXOTIC | LONG_TAG | HEX_TAG) == EXOTIC | LONG_TAG | HEX_TAG {
            out.extend_from_slice(self.hex.get().unwrap().bytes((k & !(EXOTIC | LONG_TAG | HEX_TAG)) as usize));
        } else if k & (EXOTIC | LONG_TAG) == EXOTIC | LONG_TAG {
            out.extend_from_slice(self.long.get().unwrap().bytes((k & !(EXOTIC | LONG_TAG)) as usize));
        } else if k & EXOTIC != 0 {
            let shard = k as usize & (INTERN_SHARDS - 1);
            let index = ((k & !EXOTIC) >> INTERN_SHARD_BITS) as usize;
            let shard_data = self.shards[shard].lock().unwrap();
            out.extend_from_slice(&shard_data.entries[index].bytes);
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
                // pack eight 7-bit bytes into 56 bits without PEXT (microcoded on some AMD CPUs)
                let y = (tail & 0x007f_007f_007f_007f) | ((tail & 0x7f00_7f00_7f00_7f00) >> 1);
                let z = (y & 0x0000_3fff_0000_3fff) | ((y & 0x3fff_0000_3fff_0000) >> 2);
                let suffix = (z & 0x0fff_ffff) | ((z >> 32) << 28);
                return (u64::from(id[0]) << 56) | suffix;
            }
        }
        self.key_in(id, limit)
    }

    /// Start a key lookup early: for a hex-suffixed ID, its packed value and prefetched home slot.
    #[inline(always)]
    fn pre(&self, s: &[u8]) -> Option<(u64, u64, usize)> {
        if s.len() == 33 {
            if let Some(t) = self.hex.get() {
                if let Some((lo, hi)) = unsafe { hex128(s.as_ptr().add(1)) } {
                    let h = t.home(s, lo, hi);
                    t.prefetch(h);
                    return Some((lo, hi, h));
                }
            }
        }
        None
    }
    /// Finish a lookup started with `pre`.
    #[inline(always)]
    fn key_pre(&self, s: &[u8], pre: Option<(u64, u64, usize)>, lim: usize) -> u64 {
        if let Some((lo, hi, h)) = pre {
            if let Some(i) = self.hex.get().unwrap().slot_at(s, lo, hi, h) {
                return EXOTIC | LONG_TAG | HEX_TAG | i as u64;
            }
        }
        self.key_in(s, lim)
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
    /// Start loading the first probe slot for `k` (dedup knows the next users ahead of time).
    #[inline(always)]
    fn prefetch(&self, k: u64) {
        #[cfg(target_arch = "x86_64")]
        if !self.slots.is_empty() {
            let i = (hu(k) >> (64 - self.bits)) as usize;
            unsafe {
                use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                _mm_prefetch::<_MM_HINT_T0>(self.slots.as_ptr().add(i) as *const i8);
            }
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
    // The zeroed buffer is still untouched: back it with huge pages like the arenas, because
    // every event and click probes this table at random.
    advise_huge(slots.as_mut_ptr() as *mut u8, cap * std::mem::size_of::<[u64; 2]>());
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
    // Zero for compact IDs. Long IDs use a one-based index in this worker's input slices.
    opaque_index: u32,
}

struct EventBatch<'a> {
    records: Parts<EvRec>,
    opaque_ids: Vec<&'a [u8]>,
    wc: Wc,
}

impl<'a> EventBatch<'a> {
    fn alloc(count_per_part: usize, partitions: usize) -> Self {
        Self { records: Parts::alloc(count_per_part, partitions), opaque_ids: Vec::new(), wc: Wc::new(partitions) }
    }

    fn key(&mut self, ex: &Exotic, id: &'a [u8], limit: usize) -> (u64, u32) {
        if id.len() <= 10 {
            return (ex.event_key(id, limit), 0);
        }
        self.opaque_ids.push(id);
        let index = u32::try_from(self.opaque_ids.len()).expect("too many long IDs in one worker");
        (if id.len() >= 16 { long_hash(id) } else { ex.hash.hash_one(id) }, index)
    }

    fn opaque_id(&self, index: u32) -> Option<&[u8]> {
        index.checked_sub(1).map(|i| self.opaque_ids[i as usize])
    }
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
struct CRec<'a> {
    person: u64,
    cid: &'a [u8],
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

fn parse_events<'a>(ctx: &Ctx, ex: &Exotic, dc: &mut DateCache, m: &'a [u8], b: usize, e: usize, out: &mut EventBatch<'a>) {
    let base = ctx.base;
    let lo = ctx.ws;
    let hi = ctx.we;
    let lim = m.as_ptr() as usize + m.len();
    scan_lines::<7>(m, b, e, &mut |f: &[&[u8]; 7]| {
        if f[0].is_empty() {
            return;
        }
        let upre = ex.pre(f[1]);
        let (eid, opaque_index) = out.key(ex, f[0], lim);
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
            let user = ex.key_pre(f[1], upre, lim);
            // Only purchases contribute to category revenue.
            let pc = if price == 0 {
                0
            } else {
                let category = ctx.prods.get(f[5]).map(u32::from).unwrap_or(ctx.unknown_cat);
                price | (category << 24)
            };
            EvRec { eid, user, ts: (ts - base) as u32, ingest, pc, opaque_index }
        } else {
            EvRec { eid, user: 0, ts: 0, ingest, pc: 0, opaque_index }
        };
        out.wc.push(&mut out.records, part1(eid), rec);
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
fn parse_clicks_direct<'a>(ctx: &Ctx, ex: &Exotic, lm: &LinkMap, dc: &mut DateCache, m: &'a [u8], b: usize, e: usize, out: &mut Parts<CRec<'a>>) {
    let base = ctx.base;
    let hi = ctx.we;
    let lim = m.as_ptr() as usize + m.len();
    // Identity lookups wait in a short FIFO so their table slots can be prefetched first.
    const AHEAD: usize = 16;
    let mut ring = [CRec { person: 0, cid: &[], ts: 0, camp: 0 }; AHEAD];
    let mut queued = 0usize;
    let resolve = |r: CRec<'a>, out: &mut Parts<CRec<'a>>| {
        let person = lm.get(r.person);
        out.push(part2(person), CRec { person, ..r });
    };
    scan_lines::<4>(m, b, e, &mut |f: &[&[u8]; 4]| {
        if f[0].is_empty() || f[1].is_empty() {
            return;
        }
        let upre = ex.pre(f[1]);
        let ts = match parse_ts_fast(f[3], dc) {
            Some(t) if t >= base && t < hi => t,
            _ => return,
        };
        let camp = match ctx.camps.get(f[2]) {
            Some(c) => c as u32,
            None => return,
        };
        // Click IDs only break timestamp ties. Keep the original unquoted field.
        let cid = f[0];
        let user = ex.key_pre(f[1], upre, lim);
        lm.prefetch(user);
        let slot = queued % AHEAD;
        if queued >= AHEAD {
            resolve(ring[slot], out);
        }
        ring[slot] = CRec { person: user, cid, ts: (ts - base) as u32, camp };
        queued += 1;
    });
    for k in queued.saturating_sub(AHEAD)..queued {
        resolve(ring[k % AHEAD], out);
    }
}

#[derive(Default)]
struct DedupScratch<'a> {
    // Each word tracks 64 hash buckets. The second mask flags buckets visited twice.
    buckets: Vec<[u64; 2]>,
    winners: Vec<Option<(&'a EvRec, usize)>>,
}

fn dedup_direct<'a>(lm: &LinkMap, parts: &'a [EventBatch<'_>], p: usize, scratch: &mut DedupScratch<'a>, out: &mut Parts<PRec>, wc: &mut Wc) {
    let n: usize = parts.iter().map(|part| part.records.count(p)).sum();
    if n == 0 {
        return;
    }
    let buckets = (n * 8).next_power_of_two().max(64);
    scratch.buckets.clear();
    scratch.buckets.resize(buckets / 64, [0, 0]);
    let hash = |eid: u64| eid.wrapping_mul(0x9E37_79B9_7F4A_7C15) << P1_BITS;
    let shift = 64 - buckets.trailing_zeros();
    let records = || parts.iter().flat_map(|part| part.records.runs(p)).flatten();
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
        // All nonempty user IDs encode to a nonzero key. Invalid rows carry zero.
        if r.user != 0 {
            let person = lm.get(r.user);
            let _ = &wc; out.push(part2(person), PRec { person, ts: r.ts, pc: r.pc });
        }
    };
    // Identity lookups are random reads of a large table; prefetch a fixed distance ahead.
    const AHEAD: usize = 16;
    for (batch_index, batch) in parts.iter().enumerate() {
        for run in batch.records.runs(p) {
            for (i, record) in run.iter().enumerate() {
                if let Some(next) = run.get(i + AHEAD) {
                    lm.prefetch(next.user);
                }
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
                            scratch.winners[slot] = Some((record, batch_index));
                            break;
                        }
                        Some((previous, owner)) if previous.eid == record.eid
                            && parts[owner].opaque_id(previous.opaque_index) == batch.opaque_id(record.opaque_index) => {
                            if record.ingest < previous.ingest {
                                scratch.winners[slot] = Some((record, batch_index));
                            }
                            break;
                        }
                        Some(_) => slot = (slot + 1) & (capacity - 1),
                    }
                }
            }
        }
    }
    for (winner, _) in scratch.winners.iter().flatten() {
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
struct CItem<'a> {
    ts: u32,
    camp: u32,
    cid: &'a [u8],
}

#[derive(Default)]
struct Scratch<'a> {
    table: Vec<u32>,
    persons: Vec<u64>,
    lidx: Vec<u32>,
    cnt: Vec<u32>,
    items: Vec<u64>,
    ccnt: Vec<u32>,
    citems: Vec<CItem<'a>>,
    ses: Vec<Ses>,
    cbuf: Vec<(u32, u32)>,
    ctmp: Vec<(u32, u32)>,
    catrev: Vec<u64>,
    day_end: Vec<usize>,
    stamp: Vec<u32>,
    sid: u32,
}

fn process_partition<'a>(ctx: &Ctx, evs: &[Parts<PRec>], cls: &[Parts<CRec<'a>>], p: usize, sc: &mut Scratch<'a>, agg: &mut Agg) {
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
    sc.citems.resize(nc, CItem { ts: 0, camp: 0, cid: &[] });
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
                    if group[end].cid < group[best].cid { best = end; }
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
    if sc.stamp.len() != ctx.ncamp + 1 {
        sc.stamp = vec![0; ctx.ncamp + 1];
        sc.sid = 0;
    }
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
        // few clicks: count assists directly from the click window of each session
        let direct = clicks.len() <= 48;
        let mut clo = 0usize;
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
            if direct {
                if cp > 0 {
                    let from = start.saturating_sub(WEEK_MS as u32);
                    while clo < cp && clicks[clo].ts < from {
                        clo += 1;
                    }
                    if clo < cp {
                        sc.sid = sc.sid.wrapping_add(1);
                        if sc.sid == 0 {
                            sc.stamp.iter_mut().for_each(|x| *x = 0);
                            sc.sid = 1;
                        }
                        let sid = sc.sid;
                        for q in clo..cp {
                            let c = clicks[q].camp as usize;
                            if sc.stamp[c] != sid {
                                sc.stamp[c] = sid;
                                agg.camp[c * nd + day as usize][3] += 1;
                            }
                        }
                    }
                }
            } else {
                ses.push(Ses { start, day });
            }
        }
        finish_row(&mut row, catrev, &mut agg.rows);
        if !direct && !clicks.is_empty() {
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


fn stream_rows(ud: &mut Option<pq::Stream>, dir: &Path, t: usize, rows: &[URow], ex: &Exotic, cats: &[Vec<u8>], sbuf: &mut Vec<u8>) {
    let w = ud.get_or_insert_with(|| {
        pq::Stream::create(
            &dir.join(format!("part-{:03}.parquet", t)),
            &[
                ("user_id", pq::SKind::Str),
                ("day", pq::SKind::Date),
                ("sessions", pq::SKind::I32),
                ("events", pq::SKind::I32),
                ("revenue_cents", pq::SKind::I64),
                ("top_category", pq::SKind::OptStr),
            ],
        )
        .expect("create user_daily")
    });
    for r in rows {
        sbuf.clear();
        ex.bytes(r.person, sbuf);
        w.str(0, sbuf);
        w.i32(1, r.day);
        w.i32(2, r.sessions as i32);
        w.i32(3, r.events as i32);
        w.i64(4, r.revenue as i64);
        w.opt_str(5, if r.top >= 0 { Some(&cats[r.top as usize][..]) } else { None });
        w.end_row().expect("write user_daily");
    }
}

#[allow(dead_code)]
fn write_user_daily(path: &Path, rows: &[URow], ex: &Exotic, cats: &[Vec<u8>]) {
    let n = rows.len();
    let mut uid = pq::StrCol::with_capacity(n * 14);
    // PLAIN integer columns are little-endian values back to back: encode them in place.
    let mut day = Vec::with_capacity(n * 4);
    let mut sessions = Vec::with_capacity(n * 8);
    let mut events = Vec::with_capacity(n * 8);
    let mut revenue = Vec::with_capacity(n * 8);
    let mut top = pq::OptStrCol::with_capacity(n);
    let mut sbuf = Vec::with_capacity(16);
    for r in rows {
        sbuf.clear();
        ex.bytes(r.person, &mut sbuf);
        uid.push(&sbuf);
        day.extend_from_slice(&r.day.to_le_bytes());
        sessions.extend_from_slice(&(r.sessions as i64).to_le_bytes());
        events.extend_from_slice(&(r.events as i64).to_le_bytes());
        revenue.extend_from_slice(&(r.revenue as i64).to_le_bytes());
        top.push(if r.top >= 0 { Some(&cats[r.top as usize][..]) } else { None });
    }
    let cols = vec![
        pq::Column { name: "user_id", kind: pq::Kind::Str, optional: false, data: uid.buf },
        pq::Column { name: "day", kind: pq::Kind::Date, optional: false, data: day },
        pq::Column { name: "sessions", kind: pq::Kind::I64, optional: false, data: sessions },
        pq::Column { name: "events", kind: pq::Kind::I64, optional: false, data: events },
        pq::Column { name: "revenue_cents", kind: pq::Kind::I64, optional: false, data: revenue },
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
    {
        // distinct long user IDs: well under one per 2 KB of input; overflow falls back to locked interning
        let total: usize = ev_maps.iter().chain(&cl_maps).chain(&ln_maps).map(|m| m.len()).sum();
        ex.init_long(total / 4000 + 4096);
    }
    let ev_chunks = make_chunks(&ev_maps, 4 << 20);
    let cl_chunks = make_chunks(&cl_maps, 4 << 20);
    let ln_chunks = make_chunks(&ln_maps, 1 << 20);
    // Ask the kernel to read every input chunk ahead, in the order the workers consume them,
    // so a cold page cache streams from disk at full speed instead of faulting 4 KB at a time.
    if std::env::var_os("SS_NOWILL").is_none() {
        let order: Vec<(usize, usize)> = ln_chunks.iter().map(|&(fi, b, e)| (ln_maps[fi].as_ptr() as usize + b, e - b))
            .chain(ev_chunks.iter().map(|&(fi, b, e)| (ev_maps[fi].as_ptr() as usize + b, e - b)))
            .chain(cl_chunks.iter().map(|&(fi, b, e)| (cl_maps[fi].as_ptr() as usize + b, e - b)))
            .collect();
        let nwill: usize = std::env::var("SS_WILLT").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let order = std::sync::Arc::new(order);
        for w in 0..nwill {
            let order = order.clone();
            std::thread::spawn(move || {
                for (i, &(p, len)) in order.iter().enumerate() {
                    if i % nwill != w {
                        continue;
                    }
                    let start = p & !4095;
                    let end = p + len;
                    unsafe { madvise(start as *mut u8, end - start, 3) }; // MADV_WILLNEED
                }
            });
        }
    }
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
        let mut evp = EventBatch::alloc(ev_cap, P1);
        let mut clp = Parts::<CRec>::alloc(cl_cap, P2);
        let mut dc = DateCache::new();
        loop {
            let k = ev_next.fetch_add(1, AO::Relaxed);
            if k >= ev_chunks.len() { break; }
            let (fi,b,e) = ev_chunks[k];
            parse_events(&ctx,&ex,&mut dc,ev_maps[fi],b,e,&mut evp);
        }
        evp.wc.finish(&mut evp.records);
        let ids = std::sync::LazyLock::force(&lm);
        loop {
            let k = cl_next.fetch_add(1, AO::Relaxed);
            if k >= cl_chunks.len() { break; }
            let (fi,b,e) = cl_chunks[k];
            parse_clicks_direct(&ctx,&ex,ids,&mut dc,cl_maps[fi],b,e,&mut clp);
        }
        (evp,clp)
    };
    let parsed: Vec<(EventBatch<'_>, Parts<CRec>)> = std::thread::scope(|scope| {
        let workers: Vec<_> = (1..nthreads).map(|_| scope.spawn(&parse)).collect();
        // Other workers can parse events while the calling thread resolves identities.
        let tl = std::time::Instant::now();
        std::sync::LazyLock::force(&lm);
        if timing { eprintln!("  linkmap build {:?} (done at {:?})", tl.elapsed(), t0.elapsed()); }
        std::iter::once(parse()).chain(workers.into_iter().map(|h| h.join().unwrap())).collect()
    });
    let lm = std::sync::LazyLock::force(&lm);
    let (ev_parts,pc_parts):(Vec<_>,Vec<_>)=parsed.into_iter().unzip();
    tick("parse, identity-aware clicks");
    let p_next = AtomicUsize::new(0);
    let pe_parts: Vec<Parts<PRec>> = std::thread::scope(|s| {
        let hs:Vec<_> = (0..nthreads).map(|_|s.spawn(|| {
            let mut out = Parts::<PRec>::alloc(pr_cap,P2);
            let mut wc = Wc::new(P2);
            let mut scratch = DedupScratch::default();
            loop {
                let p=p_next.fetch_add(1,AO::Relaxed);
                if p >= P1 {break;}
                dedup_direct(lm,&ev_parts,p,&mut scratch,&mut out,&mut wc);
                // this partition's event blocks are free now: recycle them for the output
                for b in ev_parts.iter() {
                    unsafe { (*(&b.records as *const Parts<EvRec> as *mut Parts<EvRec>)).release(p, out.pool) };
                }
            }
            wc.finish(&mut out);
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
                    let mut agg = Agg { rows: Vec::with_capacity(70000), camp: vec![[0u64; 4]; (ctx.ncamp + 1) * ctx.nd] };
                    let mut sc = Scratch::default();
                    let mut ud: Option<pq::Stream> = None;
                    let mut sbuf = Vec::with_capacity(64);
                    loop {
                        let k = q_next.fetch_add(1, AO::Relaxed);
                        if k >= P2 {
                            break;
                        }
                        process_partition(ctx, pe_parts, pc_parts, order[k].1, &mut sc, &mut agg);
                        if agg.rows.len() >= 65536 {
                            stream_rows(&mut ud, ud_dir, t, &agg.rows, ex, cats, &mut sbuf);
                            agg.rows.clear();
                        }
                    }
                    if t == 0 || !agg.rows.is_empty() || ud.is_some() {
                        stream_rows(&mut ud, ud_dir, t, &agg.rows, ex, cats, &mut sbuf);
                        ud.unwrap().finish().expect("write user_daily");
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
    if timing { eprintln!("  long ids: {}", ex.long.get().map(|t| t.len()).unwrap_or(0) + ex.hex.get().map(|t| t.len()).unwrap_or(0)); }
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
        let mut evp = EventBatch::alloc(total / 80 * 5 / 4 / P1 + 32, P1);
        eprintln!("alloc {:?}", t.elapsed());
        for m in &ev_maps { parse_events(&ctx, &ex, &mut dc, m, 0, m.len(), &mut evp); if mode == "parse_np" { evp.records.clear(); evp.opaque_ids.clear(); } }
        eprintln!("{} {:?} n={}", mode, t.elapsed(), (0..P1).map(|p| evp.records.count(p)).sum::<usize>());
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
            "fall" => scan_lines::<7>(m, 0, m.len(), &mut |f: &[&[u8]; 7]| {
                cnt += 1;
                for x in f.iter() { acc = acc.wrapping_add(x.len() as u64 ^ (x.as_ptr() as u64)); }
            }),
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
        let event = |eid, user, ingest, opaque_index| EvRec { eid, user, ts: 1, ingest, pc: 0, opaque_index };
        let mut a = EventBatch::alloc(1, 1);
        let mut b = EventBatch::alloc(1, 1);
        a.opaque_ids.push(b"same-long-event");
        b.opaque_ids.extend([b"distinct-long-event".as_slice(), b"same-long-event".as_slice()]);
        a.records.push(0, event(ids[0], 11, 2, 1));
        a.records.push(0, event(ids[1], 22, 5, 0));
        a.records.push(0, event(singleton, 33, 1, 0));
        b.records.push(0, event(ids[0], 44, 1, 2));
        // Force the same fingerprint for two distinct long IDs and a compact ID.
        b.records.push(0, event(ids[0], 77, 1, 1));
        b.records.push(0, event(ids[0], 88, 1, 0));
        b.records.push(0, event(ids[1], 0, 3, 0));
        b.records.push(0, event(ids[2], 66, 1, 0));
        let inputs = [a, b];
        let mut scratch = DedupScratch::default();
        let mut out = Parts::alloc(1, P2);
        let mut wc = Wc::new(P2);
        dedup_direct(&LinkMap::empty(), &inputs, 0, &mut scratch, &mut out, &mut wc);
        wc.finish(&mut out);
        let mut users = Vec::new();
        for p in 0..P2 {
            out.each(p, |r| users.push(r.person));
        }
        users.sort_unstable();
        assert_eq!(users, [33, 44, 66, 77, 88]);
    }

    #[test]
    fn queued_click_lookups_keep_every_click_at_chunk_ends() {
        // The identity prefetch queue holds 16 clicks; check sizes around it and a linked user.
        let mut camps = Lookup::new(b"c_");
        camps.insert(b"c_1", 0);
        let ctx = Ctx { base: 0, ws: 0, we: i64::MAX / 4, day0: 0, nd: 1, ncat: 1, ncamp: 1,
                        prods: Lookup::new(b"p"), camps, unknown_cat: 0 };
        let keys = Exotic::new();
        let lm = build_linkmap(&[vec![(enc_key(b"ua").unwrap(), enc_key(b"ub").unwrap())]], &keys);
        for n in [0usize, 1, 15, 16, 17, 33] {
            let mut csv = b"click_id,user_id,campaign_id,ts\n".to_vec();
            for i in 0..n {
                let user = if i % 3 == 0 { "ub" } else { "uc" };
                csv.extend_from_slice(format!("k{i},{user},c_1,{}\n", 1000 + i).as_bytes());
            }
            let mut out = Parts::<CRec>::alloc(4, P2);
            let mut dc = DateCache::new();
            parse_clicks_direct(&ctx, &keys, &lm, &mut dc, &csv, 0, csv.len(), &mut out);
            let mut got = Vec::new();
            for p in 0..P2 {
                out.each(p, |r| {
                    assert_eq!(part2(r.person), p);
                    got.push((r.ts, r.person));
                });
            }
            got.sort_unstable();
            let want: Vec<_> = (0..n)
                .map(|i| (1000 + i as u32, enc_key(if i % 3 == 0 { b"ua" } else { b"uc" }).unwrap()))
                .collect();
            assert_eq!(got, want, "{n} clicks");
        }
    }
}
