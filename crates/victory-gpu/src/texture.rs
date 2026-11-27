//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory was written since the last detiling
//! ([`crate::watch`]), in write-watched memory; in physical-page views,
//! which cannot be watched, a content hash of the whole memory at every
//! submission that samples it.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Only textures read through descriptor tables are brought up to date
//! (directly indexed views are not known per draw); plane 0 is uploaded
//! (every mip and array slice), depth and stencil planes are not.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use victory_formats::texture::{Error, Layout};
use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    buffer: ID3D12Resource,
    /// Its bytes.
    size: usize,
    data: *mut u8,
    footprints: Vec<UploadFootprint>,
    layout: Layout,
    /// Mips per array slice.
    mips: u32,
    memory: *const u8,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    last: Mutex<Last>,
}

const NOT_WATCHED: u8 = 2;

struct UploadFootprint {
    placed: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    range: Range<usize>,
    depth_pitch: usize,
}

impl UploadFootprint {
    fn new(
        placed: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
        rows: u32,
        row_size: u64,
        total: usize,
    ) -> Option<Self> {
        let pitch = placed.Footprint.RowPitch as usize;
        let row_size = usize::try_from(row_size).ok()?;
        let (len, depth_pitch) = footprint_size(pitch, rows, placed.Footprint.Depth, row_size)?;
        let start = usize::try_from(placed.Offset).ok()?;
        let end = start.checked_add(len)?;
        (end <= total).then_some(Self {
            placed,
            range: start..end,
            depth_pitch,
        })
    }
}

/// What a texture's staging buffer holds: the content hash of the memory
/// detiled (memory not watched), or the write-watch asking it was detiled
/// after, with room to read the memory into.
#[derive(Default)]
struct Last {
    hash: Option<[u64; 2]>,
    asked: u64,
    /// `watch::remap_epoch()` when a `MEM_MAPPED` range was last verified
    /// committed (0 if not cached).
    committed_epoch: u64,
    read: Vec<u8>,
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

static STAGING: Mutex<Option<HashMap<usize, Arc<Staging>>>> = Mutex::new(None);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HostWrite {
    /// Write-watched memory: the write-watch asking when the host copy happened.
    Watched {
        start: usize,
        end: usize,
        asked: u64,
    },
    /// Non-watched memory (physical-page view): content hash at the host copy.
    Hashed {
        memory: usize,
        size: usize,
        hash: Option<blake3::Hash>,
    },
    /// Layout unknown or unsupported: keep host contents until released.
    Fixed,
}

/// Textures the host GPU wrote by a copy or resolve: their contents stay on
/// the host until the title writes their memory again.
static HOST_WRITTEN: Mutex<Option<HashMap<usize, HostWrite>>> = Mutex::new(None);

fn host_write_state(memory: *const u8, size: usize) -> HostWrite {
    let start = memory as usize;
    if let Some((now, _)) = crate::watch::writes(start, start + size, false) {
        return HostWrite::Watched {
            start,
            end: start + size,
            asked: now,
        };
    }
    let hash = committed(memory, size).then(|| {
        // SAFETY: committed memory of the texture's layout size.
        let tiled = unsafe { std::slice::from_raw_parts(memory, size) };
        blake3::hash(tiled)
    });
    HostWrite::Hashed {
        memory: start,
        size,
        hash,
    }
}

/// Whether `resource`'s host copy is still newer than its title memory.
fn still_host_written(resource: usize) -> bool {
    let state = HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .and_then(|map| map.get(&resource).copied());
    let Some(state) = state else {
        return false;
    };
    let unchanged = match state {
        HostWrite::Fixed => true,
        HostWrite::Watched { start, end, asked } => crate::watch::writes(start, end, false)
            .is_none_or(|(_, written)| written.iter().all(|&(_, w)| w <= asked)),
        HostWrite::Hashed { memory, size, hash } => {
            if !committed(memory as *const u8, size) {
                true
            } else {
                // SAFETY: committed memory of the texture's layout size.
                let tiled = unsafe { std::slice::from_raw_parts(memory as *const u8, size) };
                hash == Some(blake3::hash(tiled))
            }
        }
    };
    if !unchanged {
        let mut map = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(map) = map.as_mut()
            && map.get(&resource) == Some(&state)
        {
            map.remove(&resource);
        }
    }
    unchanged
}

/// A host copy or resolve writes `resource`.
pub(crate) fn host_written(resource: This) {
    if crate::resource::host_resource(resource).is_some_and(|t| !title_written(t)) {
        return;
    }
    let state = crate::resource::texture_memory(resource)
        .and_then(|(memory, desc)| {
            let layout = crate::resource::texture_layout(&desc)?.planes[0]
                .layout
                .clone();
            (!layout.desc.swizzle.xor()).then(|| host_write_state(memory, layout.size as usize))
        })
        .unwrap_or(HostWrite::Fixed);
    let mut map = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    map.get_or_insert_with(HashMap::new)
        .insert(resource as usize, state);
}

/// The title released a resource.
pub(crate) fn forget(resource: usize) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&resource);
    }
    let mut host_written = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(host_written) = host_written.as_mut() {
        host_written.remove(&resource);
    }
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    // SAFETY: a live resource.
    title_written_flags(unsafe { texture.GetDesc() }.Flags)
}

fn title_written_flags(flags: D3D12_RESOURCE_FLAGS) -> bool {
    let host_writable = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET
        | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL
        | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    (flags & host_writable) == D3D12_RESOURCE_FLAG_NONE
}

/// Records, once per list, the copy of `resource`'s staging buffer into
/// its host copy, and the detiling that refreshes the staging buffer at
/// submission if the memory changed. Leaves the texture in the copy
/// destination state.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    if !list.first_use(resource as usize) || still_host_written(resource as usize) {
        return;
    }
    let Some(staging) = staging(resource, texture) else {
        return;
    };
    const COPY_DEST: u32 = 0x400;
    crate::list::ensure_state(list, resource, COPY_DEST);
    // The staging buffer lives as long as `staging`, which the deferred
    // refresh below holds until the list is reset.
    for (i, footprint) in staging.footprints.iter().enumerate() {
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.placed.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (staging.buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
            list.note_read(
                || {
                    format!(
                        "texture upload {resource:?} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.size,
                        footprint.placed.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + footprint.placed.Offset, footprint.range.len() as u64),
            );
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&staging.buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint.placed,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    // Once per submission however many of its lists sample the texture.
    list.defer_texture(staging);
}

impl Staging {
    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all).
    pub(crate) fn refresh(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let read = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
    }

    /// Returns the bytes staged for upload into the host texture (0 if unchanged).
    fn refresh_now(&self) -> usize {
        let Ok(size) = usize::try_from(self.layout.size) else {
            return 0;
        };
        let start = self.memory as usize;
        let Some(end) = start
            .checked_add(size)
            .filter(|_| size <= isize::MAX as usize)
        else {
            return 0;
        };
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        let epoch = crate::watch::remap_epoch();
        if last.committed_epoch != 0 && last.committed_epoch != epoch {
            last.committed_epoch = 0;
            self.watched.store(0, Ordering::Relaxed);
        }
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED
            && let Some((now, written)) = crate::watch::writes(start, end, false)
        {
            self.watched.store(1, Ordering::Relaxed);
            if last.asked != 0 && written.iter().all(|&(_, w)| w <= last.asked) {
                return 0;
            }
            last.asked = now;
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page.
            last.read.resize(size, 0);
            crate::host::copy_mapped(last.read.as_mut_ptr(), self.memory, size, false);
            self.detile(&last.read);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
            return self.size;
        }
        if last.committed_epoch != epoch {
            match committed_kind(self.memory, size) {
                Committed::No => {
                    #[cfg(feature = "diagnostics")]
                    tracing::debug!(
                        memory = ?self.memory,
                        size,
                        desc = ?self.layout.desc,
                        "a sampled texture's memory is not committed"
                    );
                    return 0;
                }
                Committed::Mapped => {
                    // Only physical-page views (MEM_MAPPED) are guaranteed to
                    // notify `title_memory_remapped` on unmap, so only cache
                    // their commit status across submissions.
                    last.committed_epoch = epoch;
                    self.watched.store(NOT_WATCHED, Ordering::Relaxed);
                }
                Committed::Other => {
                    self.watched.store(NOT_WATCHED, Ordering::Relaxed);
                }
            }
        }
        // SAFETY: committed memory of the texture's layout size.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        let hash = content_hash(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if last.hash == Some(hash) {
            return 0;
        }
        last.hash = Some(hash);
        self.detile(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        self.size
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer.
    fn detile(&self, tiled: &[u8]) {
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        // SAFETY: the upload buffer is mapped for exactly this range and
        // refresh holds `last`'s lock while writing it.
        let linear = unsafe { std::slice::from_raw_parts_mut(self.data, self.size) };
        if let Err((e, mip, slice)) =
            detile_plane(&self.layout, self.mips, &self.footprints, tiled, linear)
        {
            tracing::warn!(?e, mip, slice, "texture not detiled");
        }
    }
}

fn detile_plane(
    layout: &Layout,
    mips: u32,
    footprints: &[UploadFootprint],
    tiled: &[u8],
    linear: &mut [u8],
) -> Result<(), (Error, u32, u32)> {
    for (i, footprint) in footprints.iter().enumerate() {
        let (mip, slice) = (i as u32 % mips, i as u32 / mips);
        let target =
            linear
                .get_mut(footprint.range.clone())
                .ok_or((Error::TooSmall, mip, slice))?;
        let f = &footprint.placed.Footprint;
        if contiguous_copy(ContiguousCopy {
            layout,
            tiled,
            linear: target,
            mip,
            slice,
            pitch: f.RowPitch as usize,
            rows: f.Height,
            depth: f.Depth,
        }) {
            continue;
        }
        layout
            .detile(
                tiled,
                mip,
                slice,
                target,
                footprint.placed.Footprint.RowPitch as usize,
                footprint.depth_pitch,
            )
            .map_err(|e| (e, mip, slice))?;
    }
    Ok(())
}

struct ContiguousCopy<'a> {
    layout: &'a Layout,
    tiled: &'a [u8],
    linear: &'a mut [u8],
    mip: u32,
    slice: u32,
    pitch: usize,
    rows: u32,
    depth: u32,
}

fn contiguous_copy(copy: ContiguousCopy<'_>) -> bool {
    let ContiguousCopy {
        layout,
        tiled,
        linear,
        mip,
        slice,
        pitch,
        rows,
        depth,
    } = copy;
    use victory_formats::texture::Swizzle;
    if layout.desc.swizzle != Swizzle::LINEAR {
        return false;
    }
    let Some(m) = layout.mips.get(mip as usize) else {
        return false;
    };
    let bytes = (m.width as usize).saturating_mul(layout.desc.element_bytes as usize);
    let native = (m.pitch as usize).saturating_mul(layout.desc.element_bytes as usize);
    if native != pitch
        || pitch != bytes
        || rows != m.height
        || depth != m.depth.max(1)
        || bytes == 0
    {
        return false;
    }
    let Some(len) = bytes
        .checked_mul(rows as usize)
        .and_then(|len| len.checked_mul(depth as usize))
    else {
        return false;
    };
    let Some(at) = (m.offset as usize)
        .checked_add((slice as usize).saturating_mul(layout.slice_size as usize))
    else {
        return false;
    };
    let Some(end) = at.checked_add(len) else {
        return false;
    };
    if end > tiled.len() || len > linear.len() {
        return false;
    }
    linear[..len].copy_from_slice(&tiled[at..][..len]);
    true
}

/// Bytes read by a copy and the stride between depth slices.
fn footprint_size(pitch: usize, rows: u32, depth: u32, row_size: usize) -> Option<(usize, usize)> {
    if row_size == 0 || row_size > pitch {
        return None;
    }
    let depth_pitch = pitch.checked_mul(rows as usize)?;
    // D3D12 allocates the data bytes of the last row only.
    let preceding_rows = (rows as usize)
        .checked_mul(depth as usize)?
        .checked_sub(1)?;
    let len = pitch.checked_mul(preceding_rows)?.checked_add(row_size)?;
    Some((len, depth_pitch))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Committed {
    No,
    Mapped,
    Other,
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    committed_kind(at, len) != Committed::No
}

fn committed_kind(at: *const u8, len: usize) -> Committed {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    let mut p = at as usize;
    let Some(end) = p.checked_add(len) else {
        return Committed::No;
    };
    let mut all_mapped = true;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.State != MEM_COMMIT {
            return Committed::No;
        }
        if info.Type != MEM_MAPPED {
            all_mapped = false;
        }
        let Some(next) = (info.BaseAddress as usize).checked_add(info.RegionSize) else {
            return Committed::No;
        };
        p = next;
    }
    if all_mapped {
        Committed::Mapped
    } else {
        Committed::Other
    }
}

fn plane_subresources(desc: &D3D12_RESOURCE_DESC) -> u32 {
    let mips = u32::from(desc.MipLevels);
    if desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        mips
    } else {
        mips * u32::from(desc.DepthOrArraySize)
    }
}

/// The staging state of a title texture, made on first use.
fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    if let Some(s) = map.get(&(resource as usize)) {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let count = plane_subresources(&d);
    if count == 0 {
        return None;
    }
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut row_sizes = vec![0u64; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            Some(row_sizes.as_mut_ptr()),
            Some(&mut total),
        );
    }
    let size = usize::try_from(total)
        .ok()
        .filter(|&size| size <= isize::MAX as usize)?;
    let footprints = footprints
        .into_iter()
        .zip(rows)
        .zip(row_sizes)
        .map(|((placed, rows), row_size)| UploadFootprint::new(placed, rows, row_size, size))
        .collect::<Option<Vec<_>>>()?;
    let (buffer, data) = host.upload_buffer(total)?;
    crate::host::name(&buffer, || {
        format!(
            "staging of {resource:?} {}x{}x{} f{}",
            d.Width, d.Height, d.DepthOrArraySize, d.Format.0
        )
    });
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    let s = Arc::new(Staging {
        buffer,
        size,
        data,
        footprints,
        layout,
        mips: u32::from(d.MipLevels),
        memory,
        watched: AtomicU8::new(0),
        last: Mutex::default(),
    });
    map.insert(resource as usize, s.clone());
    Some(s)
}

/// Fast 128-bit non-cryptographic content digest for detecting changes in
/// unwatched texture memory. Processes 32 bytes per iteration across four
/// independent 64-bit accumulators to stay memory-bandwidth-bound.
fn content_hash(bytes: &[u8]) -> [u64; 2] {
    const K0: u64 = 0x9E37_79B1_85EB_CA87;
    const K1: u64 = 0xC2B2_AE3D_27D4_EB4F;
    const K2: u64 = 0x1656_67B1_9E37_79F9;
    const K3: u64 = 0x85EB_CA77_C2B2_AE63;

    #[inline(always)]
    fn mix64(mut x: u64) -> u64 {
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    }

    let mut a0 = K0 ^ (bytes.len() as u64);
    let mut a1 = K1;
    let mut a2 = K2;
    let mut a3 = K3;

    let (chunks, tail32) = bytes.as_chunks::<32>();
    for chunk in chunks {
        let w0 = u64::from_le_bytes(chunk[0..8].try_into().unwrap());
        let w1 = u64::from_le_bytes(chunk[8..16].try_into().unwrap());
        let w2 = u64::from_le_bytes(chunk[16..24].try_into().unwrap());
        let w3 = u64::from_le_bytes(chunk[24..32].try_into().unwrap());
        a0 = a0.wrapping_add(w0).wrapping_mul(K0).rotate_left(31);
        a1 = a1.wrapping_add(w1).wrapping_mul(K1).rotate_left(29);
        a2 = a2.wrapping_add(w2).wrapping_mul(K2).rotate_left(33);
        a3 = a3.wrapping_add(w3).wrapping_mul(K3).rotate_left(27);
    }

    let (words, tail) = tail32.as_chunks::<8>();
    for (i, word) in words.iter().enumerate() {
        let w = u64::from_le_bytes(*word);
        match i {
            0 => a0 = a0.wrapping_add(w).wrapping_mul(K0).rotate_left(31),
            1 => a1 = a1.wrapping_add(w).wrapping_mul(K1).rotate_left(29),
            _ => a2 = a2.wrapping_add(w).wrapping_mul(K2).rotate_left(33),
        }
    }

    if !tail.is_empty() {
        let mut buf = [0u8; 8];
        buf[..tail.len()].copy_from_slice(tail);
        let w = u64::from_le_bytes(buf);
        a3 = a3.wrapping_add(w).wrapping_mul(K3).rotate_left(27);
    }

    let lo = mix64(a0 ^ a2.rotate_left(17) ^ (bytes.len() as u64).wrapping_mul(K1));
    let hi = mix64(a1 ^ a3.rotate_left(23) ^ lo.wrapping_mul(K2));
    [lo, hi]
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_formats::texture::{Dimension, Swizzle, Texture, TextureDesc};
    use windows::Win32::Graphics::Dxgi::Common::{
        DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_BC7_UNORM, DXGI_FORMAT_R8_UNORM,
        DXGI_FORMAT_R8G8B8A8_UNORM,
    };

    #[test]
    fn content_hash_detects_single_bit_tail_and_order_changes() {
        let mut data: Vec<u8> = (0..4099).map(|i| (i * 37 + 13) as u8).collect();
        let base = content_hash(&data);
        assert_eq!(base, content_hash(&data));

        // Single-bit flips across main loop, 8-byte remainder, and 1..7-byte tail.
        for pos in [0, 7, 8, 31, 32, 1024, 4088, 4095, 4096, 4098] {
            data[pos] ^= 1;
            assert_ne!(content_hash(&data), base, "missed bit flip at {pos}");
            data[pos] ^= 1;
        }

        // Swapping two 8-byte lanes must change the digest.
        let mut swapped = data.clone();
        for i in 0..8 {
            swapped.swap(i, i + 8);
        }
        assert_ne!(content_hash(&swapped), base);

        // Appending a zero byte must change the digest.
        let mut extended = data.clone();
        extended.push(0);
        assert_ne!(content_hash(&extended), base);

        // Small sizes (0..=40) are all distinct when filled with zeros.
        let mut seen = std::collections::HashSet::new();
        for len in 0..=40 {
            let h = content_hash(&vec![0u8; len]);
            assert!(seen.insert(h), "collision on zero buffer of length {len}");
        }
    }

    #[test]
    fn committed_kind_distinguishes_mapped_and_private_and_remap_advances_epoch() {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, PAGE_READWRITE, UnmapViewOfFile,
        };

        let private = vec![0u8; 4096];
        assert!(committed_kind(private.as_ptr(), private.len()) == Committed::Other);

        // SAFETY: pagefile-backed section of 1 page, unmapped and closed below.
        unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                4096,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 4096);
            assert!(!view.Value.is_null());
            assert!(committed_kind(view.Value.cast(), 4096) == Committed::Mapped);

            let before = crate::watch::remap_epoch();
            crate::watch::title_memory_remapped(view.Value as usize, 4096);
            assert!(crate::watch::remap_epoch() > before);

            UnmapViewOfFile(view);
            CloseHandle(section);
        }
    }

    #[test]
    #[ignore = "benchmark"]
    fn content_hash_vs_blake3_throughput() {
        let buf: Vec<u8> = (0..(16 << 20)).map(|i| (i * 131) as u8).collect();
        let t0 = std::time::Instant::now();
        let mut b3 = [0u8; 32];
        for _ in 0..10 {
            b3 = *std::hint::black_box(blake3::hash(std::hint::black_box(&buf))).as_bytes();
        }
        let blake3_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = std::time::Instant::now();
        let mut fast = [0u64; 2];
        for _ in 0..10 {
            fast = std::hint::black_box(content_hash(std::hint::black_box(&buf)));
        }
        let fast_ms = t1.elapsed().as_secs_f64() * 1000.0;

        eprintln!(
            "160 MiB: blake3={blake3_ms:.2} ms ({:.2} GB/s), content_hash={fast_ms:.2} ms ({:.2} GB/s), speedup={:.1}x (b3[0]={}, fast={:?})",
            0.160 / (blake3_ms / 1000.0),
            0.160 / (fast_ms / 1000.0),
            blake3_ms / fast_ms,
            b3[0],
            fast,
        );
    }

    #[test]
    fn footprint_ends_after_the_last_rows_data() {
        // Two depth slices of three rows. The sixth row starts at 1280,
        // but only its 20 data bytes belong to the upload range.
        assert_eq!(footprint_size(256, 3, 2, 20), Some((1300, 768)));
    }

    #[test]
    fn footprint_rejects_invalid_geometry_and_out_of_bounds_placement() {
        assert_eq!(footprint_size(256, 0, 1, 20), None);
        assert_eq!(footprint_size(256, 1, 0, 20), None);
        assert_eq!(footprint_size(256, 1, 1, 0), None);
        assert_eq!(footprint_size(256, 1, 1, 257), None);
        assert_eq!(footprint_size(usize::MAX, 1, 2, 1), None);

        let placed = placed(512, [5, 3, 2]);
        let footprint = UploadFootprint::new(placed, 3, 20, 1812).unwrap();
        assert_eq!(footprint.range, 512..1812);
        assert!(UploadFootprint::new(placed, 3, 20, 1811).is_none());
        let overflowing = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: u64::MAX,
            ..placed
        };
        assert!(UploadFootprint::new(overflowing, 3, 20, usize::MAX).is_none());
    }

    #[test]
    fn plane_zero_counts_each_array_mip_and_each_volume_mip_once() {
        let mut desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE1D,
            DepthOrArraySize: 6,
            MipLevels: 4,
            ..Default::default()
        };
        assert_eq!(plane_subresources(&desc), 24);
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        assert_eq!(plane_subresources(&desc), 24);
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        assert_eq!(plane_subresources(&desc), 4);
    }

    fn placed(offset: u64, [width, height, depth]: [u32; 3]) -> D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: offset,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                Width: width,
                Height: height,
                Depth: depth,
                RowPitch: 256,
                ..Default::default()
            },
        }
    }

    // Guest mip and slice, host offset and dimensions, then rows and bytes per row.
    type Shape = ((u32, u32), u64, [u32; 3], u32, u64);

    fn assert_detiles(desc: TextureDesc, element_bytes: usize, shapes: &[Shape], total: usize) {
        let layout = &Texture::new(&desc).unwrap().planes[0].layout;
        let mut tiled = vec![0u8; layout.size as usize];
        let mut linear = vec![0xff; total];
        let mut expected = linear.clone();
        let mut footprints = Vec::new();
        for (i, &((mip, slice), offset, dims, rows, row_size)) in shapes.iter().enumerate() {
            footprints
                .push(UploadFootprint::new(placed(offset, dims), rows, row_size, total).unwrap());
            for z in 0..dims[2] {
                for y in 0..rows {
                    for x in 0..row_size as usize / element_bytes {
                        let guest = layout.offset(mip, x as u32, y, slice + z).unwrap() as usize;
                        let host =
                            offset as usize + (z * rows + y) as usize * 256 + x * element_bytes;
                        let value = (i * 37 + (z * 7 + y * 13) as usize + x * 3) as u8;
                        for b in 0..element_bytes {
                            tiled[guest + b] = value + b as u8;
                            expected[host + b] = value + b as u8;
                        }
                    }
                }
            }
        }
        detile_plane(layout, desc.mips, &footprints, &tiled, &mut linear).unwrap();
        assert_eq!(linear, expected);
    }

    #[test]
    fn array_upload_uses_mip_first_order_and_preserves_padding() {
        let desc = TextureDesc {
            dimension: Dimension::Tex2d,
            width: 7,
            height: 5,
            depth_or_array: 2,
            mips: 2,
            format: DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32,
            samples: 1,
            swizzle: Swizzle::SW_4KB_S,
        };
        assert_detiles(
            desc,
            4,
            &[
                ((0, 0), 0, [7, 5, 1], 5, 28),
                ((1, 0), 1536, [3, 2, 1], 2, 12),
                ((0, 1), 2048, [7, 5, 1], 5, 28),
                ((1, 1), 3584, [3, 2, 1], 2, 12),
            ],
            3852,
        );
    }

    #[test]
    fn volume_upload_uses_pitched_depth_slices_and_shrinks_depth_by_mip() {
        let desc = TextureDesc {
            dimension: Dimension::Tex3d,
            width: 5,
            height: 3,
            depth_or_array: 3,
            mips: 2,
            format: DXGI_FORMAT_R8_UNORM.0 as u32,
            samples: 1,
            swizzle: Swizzle::LINEAR,
        };
        assert_detiles(
            desc,
            1,
            &[
                ((0, 0), 0, [5, 3, 3], 3, 5),
                ((1, 0), 2560, [2, 1, 1], 1, 2),
            ],
            2562,
        );
    }

    #[test]
    fn block_compressed_upload_uses_whole_blocks_for_partial_rows_and_mips() {
        for (format, block_bytes) in [(DXGI_FORMAT_BC1_UNORM, 8), (DXGI_FORMAT_BC7_UNORM, 16)] {
            let desc = TextureDesc {
                dimension: Dimension::Tex2d,
                width: 7,
                height: 5,
                depth_or_array: 1,
                mips: 2,
                format: format.0 as u32,
                samples: 1,
                swizzle: Swizzle::SW_4KB_S,
            };
            assert_detiles(
                desc,
                block_bytes as usize,
                &[
                    ((0, 0), 0, [8, 8, 1], 2, block_bytes * 2),
                    ((1, 0), 512, [4, 4, 1], 1, block_bytes),
                ],
                512 + block_bytes as usize,
            );
        }
    }

    #[test]
    fn host_writable_flags_exclude_texture_uploads() {
        assert!(title_written_flags(D3D12_RESOURCE_FLAG_NONE));
        assert!(title_written_flags(
            D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE
        ));
        for flag in [
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ] {
            assert!(!title_written_flags(flag));
            assert!(!title_written_flags(
                flag | D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE
            ));
        }
    }

    #[test]
    fn contiguous_subresources_take_the_one_copy_path() {
        use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};
        let desc = Desc {
            dimension: Dimension::Tex2d,
            width: 64,
            height: 4,
            depth: 1,
            mips: 1,
            element_bytes: 4,
            block: 1,
            samples: 1,
            swizzle: Swizzle::LINEAR,
        };
        let layout = Layout::new(&desc).unwrap();
        let tiled: Vec<u8> = (0..layout.size as usize).map(|i| i as u8).collect();
        let mut linear = vec![0u8; tiled.len()];
        assert!(contiguous_copy(ContiguousCopy {
            layout: &layout,
            tiled: &tiled,
            linear: &mut linear,
            mip: 0,
            slice: 0,
            pitch: 256,
            rows: 4,
            depth: 1,
        }));
        assert_eq!(linear, tiled);
    }

    #[test]
    fn padded_and_tiled_subresources_do_not_take_the_contiguous_path() {
        use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};
        let desc = Desc {
            dimension: Dimension::Tex2d,
            width: 64,
            height: 4,
            depth: 1,
            mips: 1,
            element_bytes: 4,
            block: 1,
            samples: 1,
            swizzle: Swizzle::LINEAR,
        };
        let layout = Layout::new(&desc).unwrap();
        let tiled: Vec<u8> = (0..layout.size as usize).map(|i| i as u8).collect();
        let mut padded = vec![0u8; tiled.len()];
        assert!(!contiguous_copy(ContiguousCopy {
            layout: &layout,
            tiled: &tiled,
            linear: &mut padded,
            mip: 0,
            slice: 0,
            pitch: 320,
            rows: 4,
            depth: 1,
        }));
        assert!(padded.iter().all(|&b| b == 0));

        let tiled_desc = Desc {
            swizzle: Swizzle::SW_256B_S,
            ..desc
        };
        let tiled_layout = Layout::new(&tiled_desc).unwrap();
        let mut linear = vec![0u8; tiled_layout.size as usize];
        assert!(!contiguous_copy(ContiguousCopy {
            layout: &tiled_layout,
            tiled: &tiled,
            linear: &mut linear,
            mip: 0,
            slice: 0,
            pitch: 256,
            rows: 4,
            depth: 1,
        }));
    }

    #[test]
    fn committed_rejects_an_address_range_that_wraps() {
        assert!(!committed(std::ptr::without_provenance(usize::MAX), 2));
        assert!(committed(std::ptr::without_provenance(usize::MAX), 0));
        assert!(!committed(std::ptr::null(), 1));
    }

    #[test]
    fn committed_checks_every_region_in_the_range() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READONLY, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };

        // SAFETY: reserve two pages, then commit them separately with
        // different protections so VirtualQuery reports distinct regions.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), 8192, MEM_RESERVE, PAGE_READWRITE);
            assert!(!base.is_null());
            assert_eq!(VirtualAlloc(base, 4096, MEM_COMMIT, PAGE_READWRITE), base);
            let bytes = base.cast::<u8>();
            assert!(committed(bytes, 4096));
            assert!(!committed(bytes.add(4095), 2));
            let second = bytes.add(4096).cast();
            assert_eq!(
                VirtualAlloc(second, 4096, MEM_COMMIT, PAGE_READONLY),
                second
            );
            assert!(committed(bytes.add(4095), 2));
            assert!(committed(bytes, 8192));
            assert_ne!(VirtualFree(base, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn host_written_clears_after_write_watched_memory_is_written() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let len = crate::watch::PAGE;
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let ptr = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as *mut u8;
        assert!(!ptr.is_null());
        let key = ptr as usize;
        let state = host_write_state(ptr, len);
        assert!(matches!(state, HostWrite::Watched { .. }));
        {
            let mut map = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
            map.get_or_insert_with(HashMap::new).insert(key, state);
        }
        assert!(
            still_host_written(key),
            "untouched memory stays host-written"
        );
        // SAFETY: committed page of ours.
        unsafe { ptr.write(0x42) };
        assert!(
            !still_host_written(key),
            "writing title memory clears host-written state"
        );
        assert!(!still_host_written(key));
        forget(key);
        // SAFETY: our allocation.
        unsafe { VirtualFree(ptr.cast(), 0, MEM_RELEASE) };
    }

    #[test]
    fn host_written_clears_after_non_watched_memory_changes() {
        let mut buf = vec![0x11u8; 256];
        let key = buf.as_ptr() as usize;
        let state = host_write_state(buf.as_ptr(), buf.len());
        assert!(matches!(state, HostWrite::Hashed { hash: Some(_), .. }));
        {
            let mut map = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
            map.get_or_insert_with(HashMap::new).insert(key, state);
        }
        assert!(still_host_written(key), "unchanged bytes stay host-written");
        buf[0] = 0x22;
        assert!(
            !still_host_written(key),
            "changed title bytes clear host-written state"
        );
        assert!(!still_host_written(key));
        forget(key);
    }

    #[test]
    fn unchanged_content_hash_indicates_zero_staged_bytes() {
        let data = vec![0x33u8; 1024];
        let hash = content_hash(&data);
        let last_hash = Some(hash);
        let changed = last_hash != Some(hash);
        assert!(!changed);
        let mut modified = data.clone();
        modified[0] = 0x44;
        let new_hash = content_hash(&modified);
        assert_ne!(new_hash, hash);
    }
}
