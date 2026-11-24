//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// The copy's bytes before the submission ran, from an offset on,
    /// covering the ranges its shaders write: what the GPU wrote differs.
    before: Mutex<Option<(usize, Vec<u8>)>>,
    /// What keeps the mapping alive when `buffer` is a sparse one.
    pub(crate) _backing: Option<ID3D12Resource>,
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one.
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool (at most 8 kept, the largest).
fn give_back(v: Vec<u8>) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    if pool.len() > 8 {
        pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
        pool.truncate(8);
    }
}

const NOT_WATCHED: u8 = 2;

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    /// The resource-relative offset of a title range contained by this copy.
    pub(crate) fn offset_for_range(&self, address: u64, len: u64) -> Option<u64> {
        let offset = address.checked_sub(self.start)?;
        (offset.checked_add(len)? <= self.len).then_some(offset)
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            before: Mutex::default(),
            _backing: None,
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (
            from - start,
            address.saturating_add(len).clamp(from, end) - start,
        )
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// one snapshot per copy and submission, grown to cover every range the
    /// submission declares (the world declares overlapping views of its
    /// 174 MB heap dozens of times), in reused memory (fresh snapshots cost
    /// ~2.3 million page faults per world frame).
    pub(crate) fn writing(&self, address: usize, len: usize) {
        let (from, to) = self.inside(address, len);
        if from >= to {
            return;
        }
        let end = address.saturating_add(len);
        {
            let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
            if pending.contains(&(address, end)) {
                return;
            }
            pending.push((address, end));
        }
        // SAFETY: inside the mapping.
        let host =
            |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let (start, finish) = match &*before {
            Some((at, bytes)) => ((*at).min(from), (at + bytes.len()).max(to)),
            None => (from, to),
        };
        // Parts newly covered from the copy, the rest as it was taken.
        let mut grown = pooled(finish - start);
        match before.take() {
            Some((at, bytes)) => {
                grown.extend_from_slice(host(start, at));
                grown.extend_from_slice(&bytes);
                grown.extend_from_slice(host(at + bytes.len(), finish));
                give_back(bytes);
            }
            None => grown.extend_from_slice(host(start, finish)),
        }
        // The range just declared is snapshotted as it is now: a refresh
        // since the surrounding snapshot was taken must not leave a stale
        // baseline (a gap turned into a write would replay the old bytes).
        grown[from - start..to - start].copy_from_slice(host(from, to));
        *before = Some((start, grown));
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        if let Some((taken, before)) = &*snapshot
            && *taken <= from
            && to <= taken + before.len()
            && from < to
        {
            let before = &before[from - taken..to - taken];
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote, a run
            // of such pieces with one kernel call (a call costs tens to
            // hundreds of microseconds however small; the world puts back
            // ~650 pieces a frame).
            let end_of = |at: usize| (at + PIECE - (start + at) % PIECE).min(to);
            // SAFETY: inside the mapping; only the submitting thread writes it.
            let host = |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
            let differs = |a: usize, b: usize| host(a, b) != &before[a - from..b - from];
            let mut title = Vec::new();
            let mut synced = Vec::new();
            let mut at = from;
            while at < to {
                let mut end = end_of(at);
                if differs(at, end) {
                    while end < to && differs(end, end_of(end)) {
                        end = end_of(end);
                    }
                    written += self.put_back(
                        at,
                        end,
                        &before[at - from..end - from],
                        &mut title,
                        &mut synced,
                    );
                }
                at = end;
            }
            // Pages made equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh).
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            if first < last
                && !synced.is_empty()
                && self.watched.load(Ordering::Relaxed) != NOT_WATCHED
                && let Some((now, _)) = crate::watch::writes(first * PAGE, last * PAGE, true)
            {
                let base = start / PAGE;
                let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                for page in synced.into_iter().filter(|p| (first..last).contains(p)) {
                    copied[page - base] = now;
                }
            }
        }
        // The copy-back leaves the copy equal to title memory over the range;
        // the snapshot takes those bytes so an overlapping copy-back does not
        // replay a title edit the GPU did not make.
        if let Some((taken, before)) = snapshot.as_mut()
            && *taken <= from
            && to <= *taken + before.len()
            && from < to
        {
            let (s, e) = (from - *taken, to - *taken);
            // SAFETY: inside the mapping.
            let host = unsafe { std::slice::from_raw_parts(self.data.add(from), to - from) };
            before[s..e].copy_from_slice(host);
        }
        // Every submitted step that declared the range is done with it (one
        // submission runs at a time), and the snapshot with the last range.
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.retain(|&r| r != (address, address.saturating_add(len)));
        if pending.is_empty()
            && let Some((_, bytes)) = snapshot.take()
        {
            give_back(bytes);
        }
        written
    }

    /// [`Shadow::copy_back`] for `[from, to)` of the copy, whole pieces but
    /// for the range's ends, each changed by the GPU (`old` is the range
    /// before the submission): title memory with the GPU's bytes put in,
    /// the copy taking the title's bytes in turn. Pieces that cannot be
    /// read are left alone; pieces read are noted in `synced`. Returns the
    /// bytes written.
    fn put_back(
        &self,
        from: usize,
        to: usize,
        old: &[u8],
        title: &mut Vec<u8>,
        synced: &mut Vec<usize>,
    ) -> usize {
        let start = self.start as usize;
        title.resize(to - from, 0);
        if !crate::host::read_into((start + from) as *const u8, title) {
            // Not all of it is mapped: piece by piece.
            if to - from <= PIECE {
                return 0;
            }
            let mut written = 0;
            let mut at = from;
            while at < to {
                let end = (at + PIECE - (start + at) % PIECE).min(to);
                written += self.put_back(at, end, &old[at - from..end - from], title, synced);
                at = end;
            }
            return written;
        }
        // SAFETY: inside the mapping; only the submitting thread writes it.
        let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(from), to - from) };
        let first = (PIECE - (start + from) % PIECE).min(to - from);
        synced.extend((start + from) / PAGE..=(start + to - 1) / PAGE);
        let mut written = 0;
        for (a, b) in merge(title, host, old, first) {
            crate::watch::writing(start + from + a, b - a);
            crate::host::write_mapped((start + from + a) as *mut u8, title[a..].as_ptr(), b - a);
            written += b - a;
        }
        written
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), address.saturating_add(len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            return bytes;
        }
        self.update_free(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        self.copy(from, to);
        to - from
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let base = self.start as usize / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        let (mut bytes, mut run) = (0, None);
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = &mut copied[page - base];
            if *at == 0 || write > *at {
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                let (start, end) = (self.start as usize, (self.start + self.len) as usize);
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                run.get_or_insert(page);
            } else if let Some(first) = run.take() {
                bytes += self.copy((first * PAGE).max(from), (page * PAGE).min(to));
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some(first) = run {
            bytes += self.copy((first * PAGE).max(from), ((last + 1) * PAGE).min(to));
        }
        bytes
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        to - from
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    let Some(end) = start.checked_add(len) else {
        return;
    };
    buffers().insert((start, resource), Entry { end, shadow: None });
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    buffers().remove(&(start, resource));
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    find_or(address, len, loose)
}

fn find_or(
    address: u64,
    len: u64,
    loose: impl FnOnce(u64, u64) -> Option<Arc<Shadow>>,
) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    // The innermost buffer holding the range: the largest start at or below
    // `address` whose end reaches `end`; of equal starts, the smallest end.
    // Same-start entries are contiguous, so the first covering block found
    // in start-descending order is the innermost.
    let mut key: Option<((u64, usize), u64)> = None;
    for (&k, e) in map.range(..=(address, usize::MAX)).rev() {
        if let Some((best, bend)) = &key {
            if best.0 != k.0 {
                break;
            }
            if e.end >= end && e.end < *bend {
                key = Some((k, e.end));
            }
        } else if e.end >= end {
            key = Some((k, e.end));
        }
    }
    let Some((key, _)) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(1 << 16))?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Arc::new(Shadow::new(buffer, data, gpu, key.0, len)));
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

/// Copies of just one view's bytes, by (start, length): for a view whose
/// start is not on an element of its host copy (`aligned`). The heap slot
/// holds the copy; this map only finds it again, so it holds weak references
/// and the copy goes when the slot is remade.
static ALIGNED: Mutex<BTreeMap<(u64, u64), std::sync::Weak<Shadow>>> = Mutex::new(BTreeMap::new());

/// A copy of `[address, address + len)` that starts at `address` itself, so a
/// structured view of it can have its elements where the title put them.
///
/// A title aligns a resource; the host copy of a buffer starts where the
/// buffer does, so a view of memory the title placed inside another object
/// (`View::Placed`) starts at an offset that need not be a multiple of the
/// view's stride. This copy starts at the view, and the slot that made it
/// keeps it alive.
///
/// Its own refresh copies the 64 KiB pages the title wrote since they were
/// last copied, straight from title memory: a submission's copy-back runs
/// before the next submission prepares (`Host::execute` waits for the GPU
/// before copying back), so by then title memory holds what the GPU wrote in
/// any copy, and what shaders wrote in this one has gone back there. Copying
/// from the buffer's copy instead would copy the view's whole range again on
/// every submission.
pub(crate) fn aligned(address: u64, len: u64) -> Option<Arc<Shadow>> {
    if len == 0 {
        return None;
    }
    let key = (address, len);
    if let Some(copy) = ALIGNED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&key)
        .and_then(std::sync::Weak::upgrade)
    {
        return Some(copy);
    }
    // Views hold hundreds of megabytes in the world; made as a buffer's copy
    // is, so that shaders may write it (unordered access).
    let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(64))?;
    crate::host::name(&buffer, || format!("aligned view {address:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let copy = Arc::new(Shadow::new(buffer, data, gpu, address, len));
    ALIGNED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(key, Arc::downgrade(&copy));
    tracing::debug!(
        address = format_args!("{address:#x}"),
        len,
        "aligned view copy made"
    );
    Some(copy)
}

/// The copy `aligned` made for this range, if one is still alive.
pub(crate) fn aligned_existing(address: u64, len: u64) -> Option<Arc<Shadow>> {
    ALIGNED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&(address, len))
        .and_then(std::sync::Weak::upgrade)
}

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.checked_next_multiple_of(WINDOW)?.checked_sub(start)?;
    loose_window(address, end, start, len)
}

/// Wide windows of title memory by start: the copy and where its window
/// ends (the copy itself holds only committed memory from where root SRVs
/// in the window point on).
static WIDE: Mutex<BTreeMap<u64, (Arc<Shadow>, u64)>> = Mutex::new(BTreeMap::new());

/// When a wide window last looked for title memory committed past its copy.
static WIDE_CHECKED: Mutex<BTreeMap<u64, std::time::Instant>> = Mutex::new(BTreeMap::new());

/// Like [`find`], for a range a root SRV may read: a title buffer holding
/// it, else a window of title memory twice the range's size rounded up to a
/// power of two, on a grid of that size, so any range fits one and each
/// title byte is in at most two. The window is a sparse buffer: the title's
/// committed memory from the lowest root SRV in it on is copied, the rest
/// are unmapped tiles that read as zeros. The world's mesh shaders read
/// there, past the committed end of streaming pools (Aftermath: 741 KB past
/// it), which the console answers with zeros too; backing whole windows
/// cost 11.5 GB.
pub(crate) fn find_wide(address: u64, len: u64) -> Option<Arc<Shadow>> {
    find_or(address, len, |address, end| {
        const TILE: u64 = 64 << 10;
        let mut wide = WIDE.lock().unwrap_or_else(|e| e.into_inner());
        let grain = (end - address).next_power_of_two().max(WINDOW);
        let start = address & !(grain - 1);
        let limit = start + 2 * grain;
        // A window of a lower cell may hold the range already.
        if let Some(shadow) = wide
            .range(..start)
            .rev()
            .take(2)
            .find(|(_, (s, window_end))| s.start <= address && *window_end >= end)
            .map(|(_, (s, _))| s.clone())
        {
            return Some(shadow);
        }
        let mut lo = address & !(TILE - 1);
        let mut hi = 0;
        if let Some((shadow, _)) = wide.get(&start) {
            let has = shadow.start + shadow.len;
            if shadow.start <= address {
                if has >= end {
                    return Some(shadow.clone());
                }
                // The title commits more of a pool as it streams: look
                // again now and then whether the copy stops short of it.
                let mut checked = WIDE_CHECKED.lock().unwrap_or_else(|e| e.into_inner());
                let now = std::time::Instant::now();
                if checked
                    .get(&start)
                    .is_some_and(|t| now - *t < std::time::Duration::from_millis(100))
                {
                    return Some(shadow.clone());
                }
                checked.insert(start, now);
                if committed_end(address, limit) <= has {
                    return Some(shadow.clone());
                }
            }
            lo = lo.min(shadow.start);
            hi = has;
        }
        let committed = committed_end(address, limit).max(hi);
        let backed = (committed - lo).next_multiple_of(TILE).min(limit - lo);
        let (buffer, backing, data) = host()?.sparse_buffer(2 * grain, lo - start, backed)?;
        crate::host::name(&buffer, || format!("title memory {lo:#x}+{backed:#x} (sparse)"));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() } + (lo - start);
        tracing::info!(
            start = format_args!("{start:#x}"),
            len = backed,
            full = 2 * grain,
            "wide loose memory copied to the host"
        );
        let mut shadow = Shadow::new(buffer, data, gpu, lo, backed);
        shadow._backing = Some(backing);
        let shadow = Arc::new(shadow);
        wide.insert(start, (shadow.clone(), limit));
        Some(shadow)
    })
}

/// Where title memory committed from `address` on ends, at most `limit`
/// (adjacent allocations count: the title sees one address space).
fn committed_end(address: u64, limit: u64) -> u64 {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut p = address;
    while p < limit {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.State != MEM_COMMIT {
            break;
        }
        p = info.BaseAddress as u64 + info.RegionSize as u64;
    }
    p.min(limit).max(address + 1)
}

fn loose_window(address: u64, end: u64, start: u64, len: u64) -> Option<Arc<Shadow>> {
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    // Any window copied already that holds the range: a range crossing a
    // window's end used to get a second copy of the same bytes.
    if let Some(shadow) = map
        .range((start, 0)..=(start, u64::MAX))
        .find(|(_, s)| s.start <= address && s.start + s.len >= end)
        .map(|(_, s)| s.clone())
    {
        return Some(shadow);
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    if len >= 32 << 20 {
        tracing::info!(
            start = format_args!("{start:#x}"),
            len,
            "wide loose memory copied to the host"
        );
    } else {
        tracing::debug!(
            start = format_args!("{start:#x}"),
            len,
            "loose memory copied to the host"
        );
    }
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, start, len));
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// Title memory and its copy are compared by piece.
const PIECE: usize = 64 << 10;

/// Puts the GPU's bytes (where `host` differs from `old`, the copy before
/// the submission) into `title` where it lacks them, and `title` into
/// `host`. Returns the runs of pieces (the first `first` bytes long, the
/// rest [`PIECE`]) where `title` lacked a GPU byte, as offsets: only those
/// go back, so title writes elsewhere are kept.
fn merge(title: &mut [u8], host: &mut [u8], old: &[u8], first: usize) -> Vec<(usize, usize)> {
    let len = title.len();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut at = 0;
    while at < len {
        let end = if at == 0 { first } else { at + PIECE }.min(len);
        let mut changed = false;
        for ((t, h), o) in title[at..end].iter_mut().zip(&mut host[at..end]).zip(&old[at..end]) {
            if *h != *o && *t != *h {
                *t = *h;
                changed = true;
            }
            *h = *t;
        }
        match runs.last_mut() {
            Some(run) if changed && run.1 == at => run.1 = end,
            _ if changed => runs.push((at, end)),
            _ => {}
        }
        at = end;
    }
    runs
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overflowing_registrations_do_not_enter_the_map() {
        let start = u64::MAX - 8;
        register(0, start, 9);
        assert!(!buffers().contains_key(&(start, 0)));
    }

    #[test]
    fn overflowing_loose_window_is_rejected() {
        assert!(find(u64::MAX, 1).is_none());
        assert!(find(u64::MAX - 1, 1).is_none());
    }

    /// GPU bytes go into title memory, which keeps its own writes; only
    /// runs of pieces the title lacked a GPU byte in go back, joined.
    #[test]
    fn put_back_writes_joined_runs_of_pieces_the_gpu_changed() {
        let len = 10 + 3 * PIECE;
        let old = vec![0u8; len];
        let mut host = old.clone();
        let mut title = old.clone();
        // The GPU wrote a byte in every piece.
        for at in [2, 10 + 5, 10 + PIECE + 7, 10 + 2 * PIECE] {
            host[at] = 1;
        }
        // The title has the second piece's byte already, and wrote its own
        // byte there and in the third piece while the GPU ran.
        title[10 + 5] = 1;
        title[10 + 9] = 2;
        title[10 + PIECE + 1] = 3;
        let runs = merge(&mut title, &mut host, &old, 10);
        assert_eq!(runs, [(0, 10), (10 + PIECE, len)]);
        let mut want = old.clone();
        for (at, v) in [
            (2, 1),
            (15, 1),
            (19, 2),
            (10 + PIECE + 1, 3),
            (10 + PIECE + 7, 1),
            (10 + 2 * PIECE, 1),
        ] {
            want[at] = v;
        }
        assert!(title == want, "title memory with the GPU's bytes put in");
        assert!(host == want, "the copy takes the title's bytes");
    }
}

#[cfg(all(test, feature = "diagnostics"))]
mod warp_tests {
    use super::*;

    fn warp() -> Option<&'static crate::host::Host> {
        if std::env::var_os("VICTORY_D3D12_WARP").is_none() {
            eprintln!("skipping live D3D12 test: VICTORY_D3D12_WARP is unset");
            return None;
        }
        Some(host().expect("create the requested WARP device"))
    }

    fn shadow(device: &crate::host::Host, title: &mut [u8]) -> Shadow {
        let (buffer, data) = device
            .shared_buffer(PAGE as u64)
            .expect("map a WARP shadow buffer");
        // SAFETY: the buffer is live and mapped for at least title.len() bytes.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        Shadow::new(
            buffer,
            data,
            gpu,
            title.as_mut_ptr() as u64,
            title.len() as u64,
        )
    }

    struct Registration(usize, u64);

    impl Registration {
        fn new(resource: usize, start: u64, len: u64) -> Self {
            register(resource, start, len);
            Self(resource, start)
        }
    }

    impl Drop for Registration {
        fn drop(&mut self) {
            unregister(self.0, self.1);
        }
    }

    #[test]
    fn lookup_checks_past_noncovering_buffers() {
        let Some(_device) = warp() else { return };
        let mut title = vec![0u8; 160];
        let start = title.as_mut_ptr() as u64;
        let _outer = Registration::new(0, start, 160);
        let _short: Vec<_> = (1..=70)
            .map(|i| Registration::new(0, start + i, 1))
            .collect();
        let copy = find(start + 90, 4).unwrap();
        assert_eq!(copy.offset(start + 90), 90);
    }

    #[test]
    fn equal_start_lookup_chooses_the_smallest_covering_buffer() {
        let Some(_device) = warp() else { return };
        let mut title = vec![0u8; 64];
        let start = title.as_mut_ptr() as u64;
        let _short = Registration::new(1, start, 8);
        let _long = Registration::new(2, start, 64);
        assert_eq!(find(start + 4, 4).unwrap().left(start + 4), 4);
        assert_eq!(find(start + 4, 5).unwrap().left(start + 4), 60);
    }

    #[test]
    fn oversized_updates_and_writes_clip_to_the_buffer() {
        let Some(device) = warp() else { return };
        let mut title = [1u8; 16];
        let copy = shadow(device, &mut title);
        let start = copy.start as usize;
        assert_eq!(copy.update(start - 2, usize::MAX), title.len());
        copy.writing(start + 14, usize::MAX);
        // SAFETY: within the live mapping; the title and host copy are distinct.
        unsafe { copy.data.add(15).write(9) };
        assert_eq!(copy.copy_back(start + 14, usize::MAX), 2);
        assert_eq!(title[15], 9);
    }

    #[test]
    fn refresh_past_the_end_does_not_underflow() {
        let Some(device) = warp() else { return };
        let mut title = [1u8; 16];
        let copy = Arc::new(shadow(device, &mut title));
        let list = device.new_list().unwrap();
        copy.refresh(&list, copy.start + copy.len + 1, 1);
    }

    #[test]
    fn a_newly_declared_gap_takes_its_own_baseline() {
        let Some(device) = warp() else { return };
        let mut title = [1u8; 16];
        let copy = shadow(device, &mut title);
        let start = copy.start as usize;
        assert_eq!(copy.update(start, title.len()), title.len());
        copy.writing(start, 2);
        copy.writing(start + 12, 2);
        title[6] = 2;
        assert_eq!(copy.update(start + 6, 1), 1);
        copy.writing(start + 6, 1);
        title[6] = 3;
        assert_eq!(copy.copy_back(start + 6, 1), 0);
        assert_eq!(title[6], 3);
        assert_eq!(copy.copy_back(start, 2), 0);
        assert_eq!(copy.copy_back(start + 12, 2), 0);
    }

    #[test]
    fn overlapping_copy_backs_do_not_replay_an_earlier_cpu_edit() {
        let Some(device) = warp() else { return };
        let mut title = [1u8; 16];
        let copy = shadow(device, &mut title);
        let start = copy.start as usize;
        assert_eq!(copy.update(start, title.len()), title.len());
        copy.writing(start, 8);
        copy.writing(start + 4, 8);
        title[5] = 2;
        // SAFETY: within the live mapping; simulates a GPU write before copy-back.
        unsafe { copy.data.write(9) };
        assert_eq!(copy.copy_back(start, 8), 8);
        assert_eq!(title[0], 9);
        title[5] = 3;
        assert_eq!(copy.copy_back(start + 4, 8), 0);
        assert_eq!(title[5], 3);
    }

    /// How many bytes of `[address, address + len)` lie inside the copy that
    /// starts at `start` of `size` bytes: what a refresh of that copy can bring up
    /// to date. `0` when the range is not inside it at all.
    fn inside(address: u64, len: u64, start: u64, size: u64) -> u64 {
        let end = start.saturating_add(size);
        let from = address.max(start);
        let to = address.saturating_add(len).min(end);
        to.saturating_sub(from)
    }

    /// A refresh of the aligned copy brings up to date the bytes of the view
    /// that lie inside it, cut at both ends, and nothing when the range is
    /// outside it.
    #[test]
    fn a_refresh_covers_the_views_bytes_inside_its_copy() {
        // The hit-detection view: 3136 bytes of 56-byte elements, 48 bytes
        // into its buffer, in a copy that starts at the view.
        assert_eq!(inside(0x3000_0030, 3136, 0x3000_0030, 3136), 3136);
        // A view reaching past its copy is cut to it.
        assert_eq!(inside(0x4000_0100, 900, 0x4000_0100, 0x100), 0x100);
        // A view starting before its copy loses the part before it.
        assert_eq!(inside(0x0ff8, 0x40, 0x1000, 0x1000), 0x38);
        // Outside the copy at either end: nothing to refresh.
        assert_eq!(inside(0x1000, 0x10, 0x2000, 0x100), 0);
        assert_eq!(inside(0x20f0, 0x20, 0x1000, 0x100), 0);
        // A size that would overflow does not.
        assert_eq!(inside(u64::MAX - 8, 16, 0, 16), 0);
    }

    /// What a shader writes in a view's own copy, which starts off a page and
    /// off its buffer's elements, goes back to title memory after the
    /// submission, and a byte the title wrote meanwhile is kept.
    #[test]
    fn writes_in_a_views_own_copy_go_back_to_title_memory() {
        // Title memory, reached only through this pointer.
        let mut memory = vec![1u8; 4096];
        let title = memory.as_mut_ptr();
        // Four 56-byte elements, 48 bytes in.
        let (address, len) = (title as usize + 48, 4 * 56);
        let Some(copy) = aligned(address as u64, len as u64) else {
            return; // no host GPU
        };
        // As the submission's refresh and write declaration leave it.
        copy.copy(address, address + len);
        copy.writing(address, len);
        // The GPU writes the third element; the title writes the view's first
        // byte while it runs.
        // SAFETY: inside the copy's mapping and the title memory.
        unsafe {
            std::ptr::write_bytes(copy.data.add(112), 7, 56);
            title.add(48).write(9);
        }
        assert!(copy.copy_back(address, len) > 0);
        // SAFETY: the title memory, no longer written.
        let after = unsafe { std::slice::from_raw_parts(title, 4096) };
        assert!(after[160..216].iter().all(|&b| b == 7), "the GPU's element");
        assert_eq!(after[48], 9, "the title's own write");
        assert!(after[..48].iter().chain(&after[216..]).all(|&b| b == 1));
        drop(memory);
    }
}
