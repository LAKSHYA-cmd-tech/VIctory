//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed.

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::Mutex;

use victory_shader::emit::{RawWindow, StrideEntry};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::{VIEW_TAG, View, view_at};
use crate::host::host;

/// The title's slots past a host heap that could not hold them (the NVIDIA
/// cap): how many times a slot was skipped, the highest index among them,
/// and the size of the heap they were past.
pub(crate) static PAST_HEAP: PastHeap = PastHeap::new();

/// How many times a title slot past the host heap was skipped, the highest
/// index among them, and the size of the heap. The highest index is what
/// says how far past the heap the title went, and the count of the slots
/// past the heap is what a run's statistics name: a warning per slot past
/// the heap floods the log (310 720 lines in one world load).
pub(crate) struct PastHeap {
    skips: std::sync::atomic::AtomicU64,
    highest: std::sync::atomic::AtomicU32,
    count: std::sync::atomic::AtomicU32,
}

impl PastHeap {
    const fn new() -> Self {
        Self {
            skips: std::sync::atomic::AtomicU64::new(0),
            highest: std::sync::atomic::AtomicU32::new(0),
            count: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// Counts a title slot past a heap of `count` slots: the skips so far,
    /// the highest index seen, and the heap's size.
    fn note(&self, index: u32, count: u32) -> (u64, u32, u32) {
        let skips = self.skips.fetch_add(1, Ordering::Relaxed) + 1;
        let highest = self.highest.fetch_max(index, Ordering::Relaxed).max(index);
        self.count.store(count, Ordering::Relaxed);
        (skips, highest, count)
    }

    /// The skips so far, the highest index seen, and the heap's size.
    #[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
    pub(crate) fn get(&self) -> (u64, u32, u32) {
        (
            self.skips.load(Ordering::Relaxed),
            self.highest.load(Ordering::Relaxed),
            self.count.load(Ordering::Relaxed),
        )
    }
}

/// How many of the title's slots are past a heap of `count` slots, when the
/// title's slots reach `highest`: they are numbered from zero, so the ones
/// past the heap are those from `count` to `highest`.
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
pub(crate) fn slots_past(count: u32, highest: u32) -> u32 {
    highest.saturating_sub(count.saturating_sub(1))
}

/// Whether the `count`th skipped slot is worth a warning: the first one, and
/// then every doubling, so the log shows how far past the heap the title went
/// without a line per slot.
fn worth_a_warning(count: u64) -> bool {
    count == 1 || count.is_power_of_two()
}

pub(crate) struct HostHeap {
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
    /// Slots mirroring the title's; view heaps have `dynamic` more, then
    /// their guard slot ([`HostHeap::guard_slot`]).
    count: u32,
    /// Dynamic slots past `count` (none in sampler heaps): [`DYNAMIC`]
    /// where the host takes them, else [`MIN_DYNAMIC`].
    dynamic: u32,
    sampler: bool,
    /// The record (view index and generation) each slot was last made from.
    made: Mutex<Vec<u64>>,
    /// The host resource each slot views, held while the slot does: shaders
    /// index heaps directly, so a slot the title has not rewritten since it
    /// released the resource may still be read, and a freed host resource
    /// read there is a GPU page fault (the console reads stale memory).
    kept: Mutex<Vec<Option<ID3D12Resource>>>,
    /// A host copy of just the view's bytes, for views that start off their
    /// buffer copy's elements (`aligned`), held while the slot views it.
    kept_copy: Mutex<Vec<Option<std::sync::Arc<crate::shadow::Shadow>>>>,
    /// Where the search for free dynamic slots starts, and which are taken
    /// (by lists not yet reset).
    next: Mutex<(u32, Vec<bool>)>,
    /// Per slot, what shaders need to read a buffer whose stride shader
    /// analysis could not recover statically ([`StrideEntry`]).
    strides: Option<StrideTable>,
    /// The first of the [`ALIASES`] slots after the dynamic ones (view
    /// heaps), and which slots have a raw view there.
    alias_base: u32,
    aliases: Mutex<Aliases>,
    range_id: u64,
}

impl Drop for HostHeap {
    fn drop(&mut self) {
        let mut ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
        unregister_gpu_range(&mut ranges, self.range_id);
    }
}

/// The raw views of the bytes structured and typed buffer views read, for
/// shaders that read such a buffer without knowing its stride: they take a
/// raw handle, which reads a raw view only (over the title's own view the
/// host GPUs disagree: an RTX 5070 reads other dwords than a Radeon). The
/// runtime stride table names the slot.
struct Aliases {
    /// The pool slot of each heap slot that has a raw view.
    of: HashMap<u32, u32>,
    /// Pool slots given back.
    free: Vec<u32>,
    /// Pool slots from here on were never taken.
    next: u32,
}

/// Slots past the dynamic ones in a view heap for [`Aliases`]: one per
/// structured or typed buffer view in the heap at a time.
/// shortcut: a heap holding more such views at once leaves the rest without
/// a raw view (they read zeros through the unknown-stride path, with a
/// warning); the count is a guess, not a measured need.
const ALIASES: u32 = 1 << 16;

/// The pool's first slots: null raw views (a shader view, an unordered
/// access one) for views that cannot have a raw view of their own.
const NULL_ALIASES: u32 = 2;

struct StrideTable {
    _resource: ID3D12Resource,
    ptr: usize,
    gpu: u64,
    len: u32,
}

impl StrideTable {
    fn new(device: &ID3D12Device, len: u32) -> Option<Self> {
        let bytes = (u64::from(len) * u64::from(StrideEntry::BYTES)).max(8);
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut resource: Option<ID3D12Resource> = None;
        let mut p = std::ptr::null_mut();
        // SAFETY: plain committed upload buffer creation and persistent CPU map.
        let gpu = unsafe {
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_UPLOAD,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut resource,
                )
                .ok()?;
            let res = resource.as_ref()?;
            res.Map(0, None, Some(&mut p)).ok()?;
            std::ptr::write_bytes(p.cast::<u8>(), 0, bytes as usize);
            res.GetGPUVirtualAddress()
        };
        Some(Self {
            _resource: resource?,
            ptr: p as usize,
            gpu,
            len,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GpuRange {
    id: u64,
    start: u64,
    bytes: u64,
    step: u64,
    sampler: bool,
}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<GpuRange>> = Mutex::new(Vec::new());
static NEXT_RANGE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn register_gpu_range(
    ranges: &mut Vec<GpuRange>,
    id: u64,
    start: u64,
    bytes: u64,
    step: u64,
    sampler: bool,
) {
    ranges.push(GpuRange {
        id,
        start,
        bytes,
        step,
        sampler,
    });
}

fn unregister_gpu_range(ranges: &mut Vec<GpuRange>, id: u64) {
    ranges.retain(|range| range.id != id);
}

fn near_heaps_in(ranges: &[GpuRange], va: u64) -> String {
    let mut text = String::new();
    for &GpuRange {
        start,
        bytes,
        step,
        sampler,
        ..
    } in ranges
    {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!(
                "{:#x} past its end (slot {})",
                va - start - bytes,
                (va - start) / step
            )
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    near_heaps_in(&ranges, va)
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more.
const PORTABLE_VIEWS: u32 = 1_000_000;

/// The maximum shader-visible sampler count on D3D12 (all tiers).
const PORTABLE_SAMPLERS: u32 = 2048;

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until it is reset (its submissions have run) or released.
/// 2^16 ran out seconds into the world (every world run on an RX 9070 XT,
/// each a device removal shortly after; #59).
const DYNAMIC: u32 = 1 << 18;

/// The dynamic slots of a heap whose host would not take [`DYNAMIC`] more
/// (a capped heap, NVIDIA): the title's slots keep their room first.
const MIN_DYNAMIC: u32 = 1 << 18;

impl HostHeap {
    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        let aliases = if sampler { 0 } else { ALIASES };
        let guard = if sampler { 0 } else { 1 };
        // The larger dynamic range only where the host takes it on top of
        // every title slot; else the smaller one, and the cap after that.
        let (heap, count, extra) = if sampler {
            let requested = count;
            let target_count = if count > PORTABLE_SAMPLERS {
                tracing::warn!(count, "host sampler heap capped at {PORTABLE_SAMPLERS}");
                PORTABLE_SAMPLERS
            } else {
                count
            };
            match make(target_count) {
                Ok(heap) => (heap, target_count, 0),
                Err(e) => {
                    tracing::warn!(error = %e, count = requested, sampler, "no host descriptor heap");
                    return None;
                }
            }
        } else if count + DYNAMIC + aliases + guard <= PORTABLE_VIEWS
            && let Ok(heap) = make(count + DYNAMIC + aliases + guard)
        {
            (heap, count, DYNAMIC)
        } else if count + MIN_DYNAMIC + aliases + guard <= PORTABLE_VIEWS
            && let Ok(heap) = make(count + MIN_DYNAMIC + aliases + guard)
        {
            (heap, count, MIN_DYNAMIC)
        } else {
            // shortcut: descriptors past the host's limit are left
            // out; a title that reaches them needs remapping.
            tracing::warn!(count, "host view heap capped at {PORTABLE_VIEWS}");
            let (heap, total) = [PORTABLE_VIEWS, 500_000, 250_000, MIN_DYNAMIC]
                .into_iter()
                .find_map(|cap| make(cap).ok().map(|h| (h, cap)))?;
            let extra = if total >= PORTABLE_VIEWS {
                (1 << 17).min(total / 4)
            } else {
                MIN_DYNAMIC.min(total / 2)
            };
            (heap, total.saturating_sub(extra + aliases + guard), extra)
        };
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        tracing::debug!(count, sampler, "host descriptor heap");
        // SAFETY: a live shader-visible heap.
        let gpu = unsafe { heap.GetGPUDescriptorHandleForHeapStart().ptr };
        let range_id = NEXT_RANGE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        register_gpu_range(
            &mut GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()),
            range_id,
            gpu,
            u64::from(count + extra + aliases + guard) * u64::from(step),
            u64::from(step),
            sampler,
        );
        let strides = if sampler {
            None
        } else {
            StrideTable::new(&host.device, count + extra)
        };
        let h = HostHeap {
            heap,
            cpu,
            step,
            count,
            dynamic: extra,
            sampler,
            made: Mutex::new(vec![0; count as usize]),
            kept: Mutex::new(vec![None; (count + extra) as usize]),
            kept_copy: Mutex::new(vec![None; (count + extra) as usize]),
            next: Mutex::new((0, vec![false; extra as usize])),
            strides,
            alias_base: count + extra,
            aliases: Mutex::new(Aliases {
                of: HashMap::new(),
                free: Vec::new(),
                next: NULL_ALIASES,
            }),
            range_id,
        };
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, count + extra + aliases + guard) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
        }
        if !sampler {
            null_buffer_view(h.cpu(h.alias_base));
            null_raw_uav(h.cpu(h.alias_base + 1));
        }
        tracing::debug!(
            ms = started.elapsed().as_millis(),
            "host descriptor heap made null"
        );
        Some(h)
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        const BLOCK: u32 = 1 << 16;
        let block_size = BLOCK.min(total);
        if block_size == 0 {
            return Ok(());
        }
        let Some(host) = host() else { return Ok(()) };
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: block_size,
                        ..Default::default()
                    })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            if self.sampler {
                null_sampler(start);
            } else {
                null_view(start);
            }
            let mut filled = 1;
            while filled < block_size {
                let n = filled.min(block_size - filled);
                host.device
                    .CopyDescriptorsSimple(n, at(filled), start, kind);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = block_size.min(total - done);
                host.device
                    .CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                done += n;
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.cpu(index));
    }
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
}

impl HostHeap {
    /// `n` consecutive free dynamic slots (view heaps only), taken until
    /// [`HostHeap::release`]: the first's index.
    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        if self.sampler || n == 0 || n > self.dynamic {
            return None;
        }
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let (next, taken) = &mut *guard;
        let n = n as usize;
        let cursor = *next as usize;
        let end = self.dynamic as usize;
        let second_end = cursor.saturating_add(n - 1).min(end);
        // Search from `cursor` on, then from the start up to `second_end`
        // so candidate spans that cross `cursor` are not missed.
        let starts = (cursor..end).chain(0..second_end);
        let mut run = 0;
        let mut found = None;
        let mut last = usize::MAX;
        for at in starts {
            // A run cannot wrap past the end.
            if at != last.wrapping_add(1) {
                run = 0;
            }
            last = at;
            run = if taken[at] { 0 } else { run + 1 };
            if run == n {
                found = Some(at + 1 - n);
                break;
            }
        }
        let Some(first) = found else {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(n, "no free dynamic descriptor slots: a table is not made")
            });
            return None;
        };
        taken[first..first + n].fill(true);
        *next = ((first + n) % self.dynamic as usize) as u32;
        Some(self.count + first as u32)
    }

    /// Recycles `n` dynamic slots in ring-buffer fashion when the dynamic pool
    /// is otherwise exhausted, preventing draw/dispatch drops.
    pub(crate) fn recycle_dynamic(&self, n: u32) -> Option<u32> {
        if self.sampler || n == 0 || n > self.dynamic {
            return None;
        }
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let (next, taken) = &mut *guard;
        let n = n as usize;
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                n,
                dynamic = self.dynamic,
                "dynamic descriptor pool exhausted: recycling slots in ring-buffer fashion"
            );
        });
        let mut candidate = *next as usize;
        if candidate + n > self.dynamic as usize {
            candidate = 0;
        }
        taken[candidate..candidate + n].fill(true);
        *next = ((candidate + n) % self.dynamic as usize) as u32;
        Some(self.count + candidate as u32)
    }

    /// The title slots this heap has host slots for; fewer than the title's
    /// when the host caps its size.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// The slot translated shaders read in place of a computed heap index
    /// at or past it (a stale buffer can give any index, and one past the
    /// heap hangs some GPUs): in a view heap its last slot, after the
    /// dynamic ones, null from the start and never made; in a sampler heap
    /// its last slot (any sampler reads safely).
    pub(crate) fn guard_slot(&self) -> u32 {
        if self.sampler {
            self.count.saturating_sub(1)
        } else {
            self.count + self.dynamic + ALIASES
        }
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned).
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, first: u32, n: u32) {
        self.release_with(first, n, Self::forget_view);
    }

    pub(super) fn release_with(&self, first: u32, n: u32, mut clear: impl FnMut(&Self, u32)) {
        let Some(at) = first.checked_sub(self.count) else {
            return;
        };
        // Clear the retiring list's aliases before another list can lease
        // these slots and publish its own stride entries.
        for index in first..first.saturating_add(n).min(self.alias_base) {
            clear(self, index);
        }
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let end = (at + n).min(self.dynamic) as usize;
        if let Some(slots) = guard.1.get_mut(at as usize..end) {
            slots.fill(false);
        }
    }

    /// Makes slot `index` (a dynamic one) from `view`; a null view if the
    /// host cannot.
    pub(crate) fn make_at(&self, index: u32, view: &View) {
        let kept = make(view, self.cpu(index));
        match &kept {
            Some(Kept::Nothing) => self.keep(index, None, None),
            Some(Kept::Resource { resource, copy }) => {
                self.keep(index, Some(resource.clone()), copy.clone())
            }
            None => {
                null_for(view, self.cpu(index));
                self.keep(index, None, None);
            }
        }
        self.note_view(index, view, kept.as_ref());
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.cpu(index));
        }
        self.keep(index, None, None);
        self.forget_view(index);
    }

    /// Slot `index` now views `resource` (or nothing of the host's) and, if
    /// it reads a copy of just its view's bytes, keeps `copy` alive.
    fn keep(
        &self,
        index: u32,
        resource: Option<ID3D12Resource>,
        copy: Option<std::sync::Arc<crate::shadow::Shadow>>,
    ) {
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = kept.get_mut(index as usize) {
            // A running submission may still read the old resource through
            // this slot.
            if let Some(old) = std::mem::replace(slot, resource) {
                crate::host::retire(old);
            }
        }
        drop(kept);
        let mut held = self.kept_copy.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = held.get_mut(index as usize) {
            *slot = copy;
        }
    }

    fn set_entry(&self, index: u32, entry: StrideEntry) {
        if let Some(t) = &self.strides
            && index < t.len
        {
            // SAFETY: `t.ptr` points to a live mapped UPLOAD buffer of
            // `t.len` entries of two dwords.
            unsafe {
                (t.ptr as *mut [u32; 2])
                    .add(index as usize)
                    .write(entry.words());
            }
        }
    }

    /// GPU virtual address of the runtime stride table: a [`StrideEntry`]
    /// per slot.
    pub(crate) fn stride_table_gpu(&self) -> Option<u64> {
        self.strides.as_ref().map(|t| t.gpu)
    }

    /// Makes slot `index` (a dynamic one) a constant buffer view of host
    /// memory (the list's own, which it holds).
    pub(crate) fn constant_at(&self, index: u32, gpu: u64, size: u32) {
        self.keep(index, None, None);
        self.forget_view(index);
        let Some(host) = host() else { return };
        let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: gpu,
            SizeInBytes: size.next_multiple_of(256),
        };
        // SAFETY: a slot of our heap; the memory is the caller's.
        unsafe {
            host.device
                .CreateConstantBufferView(Some(&desc), self.cpu(index))
        };
    }

    /// Makes slot `index` (a dynamic one) a raw buffer view of `len` bytes
    /// of `buffer` from byte `offset` (a multiple of 16): a root SRV with a
    /// size, whose reads past it return zeros instead of reaching whatever
    /// the host placed after the buffer.
    pub(crate) fn raw_at(&self, index: u32, buffer: &ID3D12Resource, offset: u64, len: u64) {
        let Some(host) = host() else { return };
        let elements = (len / 4).clamp(1, u64::from(u32::MAX)) as u32;
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: offset / 4,
                    NumElements: elements,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        // SAFETY: a slot of our heap and a live buffer covering the range.
        unsafe {
            host.device
                .CreateShaderResourceView(buffer, Some(&desc), self.cpu(index))
        };
        self.keep(index, Some(buffer.clone()), None);
    }

    /// Makes slot `index` a raw buffer view that reads zeros.
    pub(crate) fn null_raw_at(&self, index: u32) {
        null_buffer_view(self.cpu(index));
        self.keep(index, None, None);
    }

    /// Whether this is a sampler heap.
    pub(crate) fn is_sampler(&self) -> bool {
        self.sampler
    }

    /// The shader-visible handle of slot `index`.
    pub(crate) fn gpu_handle(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: a live shader-visible heap.
        let start = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart().ptr };
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: start + u64::from(index) * u64::from(self.step),
        }
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    /// Makes host slot `index` from the record the title has at `handle`
    /// if it changed since, and returns the view recorded there. Buffer
    /// views are made only for draws (`for_draw`), which also copy their
    /// data: a host copy of a buffer is made on first use.
    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<View> {
        if index >= self.count {
            // Diagnostic (NVIDIA heap cap): the title's slots past the host
            // heap. The count of them is in the run's statistics; a warning
            // per slot floods the log.
            let (skips, highest, _) = PAST_HEAP.note(index, self.count);
            if worth_a_warning(skips) {
                tracing::warn!(
                    skips,
                    index,
                    highest,
                    count = self.count,
                    for_draw,
                    "title slot past the host heap"
                );
            }
            // Graceful degradation: return the title's view so callers (e.g. table_entry)
            // can still transition and prepare resources rather than dropping state.
            // SAFETY: the handle is a descriptor slot in title memory.
            let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
            return (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        // Generations start at 1, so a made slot's key is never 0.
        let key = u64::from(generation) << 32 | u64::from(i);
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        let Some(view) = view else {
            // Not a view of ours (never written, or of a resource the
            // title released): nothing to read.
            if made[index as usize] != 0 {
                made[index as usize] = 0;
                self.make_null(index);
            }
            return None;
        };
        if made[index as usize] != key {
            let later = !for_draw && matches!(view, View::Placed(..));
            let kept = if later {
                None
            } else {
                make(&view, self.cpu(index))
            };
            if let Some(kept) = kept {
                made[index as usize] = key;
                match kept {
                    Kept::Nothing => {
                        self.keep(index, None, None);
                    }
                    Kept::Resource { resource, copy } => {
                        self.keep(index, Some(resource), copy);
                    }
                }
            } else {
                made[index as usize] = 0;
                if !self.sampler {
                    null_for(&view, self.cpu(index));
                }
                self.keep(index, None, None);
            }
            self.note_view(index, &view, None);
        }
        Some(view)
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let key = *self
            .made
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(index as usize)?;
        if key == 0 {
            return Some("null".into());
        }
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A view description from its words, with the format as the host has it.
fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// A buffer view of title memory at `address` on its host copy: the first
/// element moves by the copy's offset.
/// Bytes per element of a buffer view.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (address, _, unordered, words) = buffer_parts(view)?;
    let (from, bytes, _) = extent(address, unordered, words);
    Some((from, bytes))
}

fn uav_element_size(desc: &D3D12_UNORDERED_ACCESS_VIEW_DESC) -> Option<u64> {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        Some(u64::from(buffer.StructureByteStride))
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        Some(4)
    } else {
        crate::resource::element(desc.Format.0 as u32).map(|(bytes, _)| bytes)
    }
}

impl HostHeap {
    /// Slot `index` was just made from `view` (`made`: what [`make`] gave):
    /// writes its runtime stride table entry, and for a structured or typed
    /// buffer view makes the raw view of the same bytes the entry names.
    fn note_view(&self, index: u32, view: &View, made: Option<&Kept>) {
        if self.strides.is_none() {
            return;
        }
        let Some(kept) = made else {
            // A null view: nothing to read through any handle.
            return self.forget_view(index);
        };
        let resource = match kept {
            Kept::Resource { resource, .. } => Some(resource),
            Kept::Nothing => None,
        };
        let stride = view_stride(view);
        let entry = match viewed_bytes(view, resource) {
            // Not a buffer view, or a raw one: the slot's own view.
            Viewed::Own => {
                self.unalias(index);
                StrideEntry {
                    stride,
                    ..StrideEntry::default()
                }
            }
            Viewed::Bytes {
                buffer,
                start,
                len,
                unordered,
            } => {
                let (raw_slot, bias) = self.alias(index, &buffer, start, len, unordered);
                StrideEntry {
                    stride,
                    raw_slot: Some(raw_slot),
                    bias,
                }
            }
            Viewed::Unknown => {
                once!("a buffer view whose bytes in its host copy are not found has no raw view");
                self.unalias(index);
                StrideEntry {
                    stride,
                    raw_slot: Some(self.null_alias(view_is_unordered(view))),
                    bias: 0,
                }
            }
        };
        self.set_entry(index, entry);
    }

    /// Slot `index`'s runtime stride table entry, as shaders read it.
    #[cfg(test)]
    pub(crate) fn entry_words(&self, index: u32) -> [u32; 2] {
        let t = self.strides.as_ref().expect("a view heap's table");
        assert!(index < t.len);
        // SAFETY: as in `set_entry`.
        unsafe { (t.ptr as *const [u32; 2]).add(index as usize).read() }
    }

    /// The first slot of the raw view pool.
    #[cfg(test)]
    pub(crate) fn alias_base(&self) -> u32 {
        self.alias_base
    }

    /// Slot `index` holds no view shaders read through the runtime stride
    /// table: its entry is cleared and its raw view given back.
    pub(super) fn forget_view(&self, index: u32) {
        if self.strides.is_none() {
            return;
        }
        self.unalias(index);
        self.set_entry(index, StrideEntry::default());
    }

    /// The slot of the null raw view standing in for a raw view the host
    /// cannot make: reads give zeros, writes are dropped.
    fn null_alias(&self, unordered: bool) -> u32 {
        self.alias_base + u32::from(unordered)
    }

    /// Gives back the raw view slot of `index`, if it has one.
    fn unalias(&self, index: u32) {
        let mut a = self.aliases.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = a.of.remove(&index) {
            a.free.push(slot);
        }
    }

    /// Makes the raw view of bytes `[start, start + len)` of `buffer` for
    /// slot `index` (in the slot it already has for one, or a free one): the
    /// heap slot it is in, and the bytes from its start to `start`.
    ///
    /// The raw view ends with the view's last element, so reads past it
    /// give zeros and writes are dropped, as through the view itself.
    /// shortcut: it ends on the dword holding the last byte and starts up
    /// to 12 bytes before the first (raw views start on 16 bytes); only an
    /// offset wrapping below the view's start reaches those bytes.
    fn alias(
        &self,
        index: u32,
        buffer: &ID3D12Resource,
        start: u64,
        len: u64,
        unordered: bool,
    ) -> (u32, u32) {
        let Some(window) = RawWindow::covering(start, len) else {
            // shortcut: a view starting off a dword of its host copy (an
            // element size that is not a multiple of 4) has no raw view; a
            // raw load could not address its bytes either.
            once!(
                start,
                len,
                "a buffer view off the dwords of its host copy has no raw view"
            );
            self.unalias(index);
            return (self.null_alias(unordered), 0);
        };
        let slot = {
            let mut a = self.aliases.lock().unwrap_or_else(|e| e.into_inner());
            let slot = match a.of.get(&index) {
                Some(&slot) => Some(slot),
                None => a.free.pop().or_else(|| {
                    (a.next < ALIASES).then(|| {
                        a.next += 1;
                        a.next - 1
                    })
                }),
            };
            let Some(slot) = slot else {
                once!("no free raw view slots: a buffer read without a known stride reads zeros");
                return (self.null_alias(unordered), 0);
            };
            a.of.insert(index, slot);
            self.alias_base + slot
        };
        let Some(host) = host() else {
            return (self.null_alias(unordered), 0);
        };
        // SAFETY: a live buffer holding the range (the slot's own view was
        // just made on it) and a slot of our heap.
        unsafe {
            if unordered {
                let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                    Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_UAV {
                            FirstElement: window.first,
                            NumElements: window.dwords,
                            Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                            ..Default::default()
                        },
                    },
                };
                host.device.CreateUnorderedAccessView(
                    buffer,
                    None::<&ID3D12Resource>,
                    Some(&desc),
                    self.cpu(slot),
                );
            } else {
                let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_SRV {
                            FirstElement: window.first,
                            NumElements: window.dwords,
                            StructureByteStride: 0,
                            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        },
                    },
                };
                host.device
                    .CreateShaderResourceView(buffer, Some(&desc), self.cpu(slot));
            }
        }
        (slot, window.bias)
    }
}

/// What a view the host made reads, for its raw view.
enum Viewed {
    /// Not a buffer, or through a raw view already.
    Own,
    /// These bytes of this host buffer (the view's own), read or also
    /// written.
    Bytes {
        buffer: ID3D12Resource,
        start: u64,
        len: u64,
        unordered: bool,
    },
    /// A structured or typed buffer view whose host buffer is not the one
    /// the bytes are found in now.
    Unknown,
}

/// Whether `view` is for unordered access.
fn view_is_unordered(view: &View) -> bool {
    matches!(view, View::Unordered(..) | View::Placed(true, ..))
}

/// The bytes of its host copy the structured or typed buffer view `view`
/// reads: as [`make`] places it (`kept`: the resource it made the view on),
/// from the first element's byte in the copy for the view's length.
fn viewed_bytes(view: &View, kept: Option<&ID3D12Resource>) -> Viewed {
    use windows::core::Interface;
    const BUFFER: u32 = 1;
    // The title buffer's address and length, and the view's description.
    let (base, len, words, unordered): (u64, u64, &[u32; 10], bool) = match view {
        View::Shader(r, Some(desc)) | View::Unordered(r, _, Some(desc)) => {
            let r = *r as crate::object::This;
            if crate::resource::host_resource(r).is_some() {
                return Viewed::Own;
            }
            let Some((data, len)) = crate::resource::buffer_bytes(r) else {
                return Viewed::Own;
            };
            (data as u64, len, desc, view_is_unordered(view))
        }
        View::Placed(unordered, resource, desc) if resource[0] == BUFFER => {
            let Ok(words) = desc[..10].try_into() else {
                return Viewed::Own;
            };
            (
                u64::from(desc[10]) | u64::from(desc[11]) << 32,
                u64::from(resource[4]) | u64::from(resource[5]) << 32,
                words,
                *unordered,
            )
        }
        _ => return Viewed::Own,
    };
    // (first element, elements, bytes per element), none for a raw view.
    let elements = if unordered {
        let desc = uav_desc(words);
        // SAFETY: a buffer view's union member.
        let b = unsafe { desc.Anonymous.Buffer };
        (b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 == 0).then(|| {
            (
                b.FirstElement,
                b.NumElements,
                uav_element_size(&desc).unwrap_or(0),
            )
        })
    } else {
        let desc = srv_desc(words);
        // SAFETY: a buffer view's union member.
        let b = unsafe { desc.Anonymous.Buffer };
        (b.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 == 0)
            .then(|| (b.FirstElement, b.NumElements, element_size(&desc)))
    };
    let Some((first, count, element)) = elements else {
        return Viewed::Own;
    };
    let (Some(kept), Some(shadow)) = (kept, crate::shadow::find(base, len)) else {
        return Viewed::Unknown;
    };
    if element == 0 || shadow.buffer.as_raw() != kept.as_raw() {
        return Viewed::Unknown;
    }
    Viewed::Bytes {
        buffer: shadow.buffer.clone(),
        start: shadow.offset(base) + first * element,
        len: u64::from(count) * element,
        unordered,
    }
}

/// Element byte stride of `view` for the runtime stride table (`0` for
/// non-buffer views or descriptors without a known element size).
fn view_stride(view: &View) -> u32 {
    let bytes = match view {
        View::Shader(_, Some(desc)) => element_size(&srv_desc(desc)),
        View::Placed(false, resource, desc) if resource[0] == 1 => desc[..10]
            .try_into()
            .ok()
            .map_or(0, |w: &[u32; 10]| element_size(&srv_desc(w))),
        View::Unordered(_, _, Some(desc)) => uav_element_size(&uav_desc(desc)).unwrap_or(0),
        View::Placed(true, resource, desc) if resource[0] == 1 => desc[..10]
            .try_into()
            .ok()
            .and_then(|w: &[u32; 10]| uav_element_size(&uav_desc(w)))
            .unwrap_or(0),
        _ => 0,
    };
    u32::try_from(bytes).unwrap_or(0)
}

/// A buffer view as [`make`] takes it: where its resource's memory starts and
/// how long it is (its host copy is found by these), whether it is unordered
/// access, and its descriptor's words.
fn buffer_parts(view: &View) -> Option<(u64, u64, bool, &[u32; 10])> {
    let registered = |r: usize| {
        crate::resource::buffer_bytes(r as crate::object::This)
            .map(|(data, len)| (data as u64, len))
    };
    match view {
        View::Shader(r, Some(desc)) => registered(*r).map(|(at, len)| (at, len, false, desc)),
        View::Unordered(r, _, Some(desc)) => registered(*r).map(|(at, len)| (at, len, true, desc)),
        View::Placed(unordered, resource, desc) if resource[0] == 1 => Some((
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            // The resource description's width, after the alignment.
            u64::from(resource[4]) | u64::from(resource[5]) << 32,
            *unordered,
            desc[..10].try_into().ok()?,
        )),
        _ => None,
    }
}

/// A buffer view's own range in the resource at `address` and the bytes of
/// one of its elements (0 for a format of unknown size).
fn extent(address: u64, unordered: bool, words: &[u32; 10]) -> (u64, u64, u64) {
    if unordered {
        unordered_extent(address, &uav_desc(words))
    } else {
        shader_extent(address, &srv_desc(words))
    }
}

/// [`extent`] of a shader view.
fn shader_extent(address: u64, desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> (u64, u64, u64) {
    let (from, bytes) = view_range_of(address, desc);
    (from, bytes, element_size(desc))
}

/// [`extent`] of an unordered access view.
fn unordered_extent(address: u64, desc: &D3D12_UNORDERED_ACCESS_VIEW_DESC) -> (u64, u64, u64) {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    (
        address.saturating_add(buffer.FirstElement * element),
        u64::from(buffer.NumElements) * element,
        element,
    )
}

/// Whether [`make`] gave this buffer view a copy of just its bytes, because it
/// starts off an element of its buffer's copy: the copy `readable` brings up
/// to date for it, and the one its writes go back from.
pub(crate) fn off_element(view: &View) -> bool {
    let Some((address, len, unordered, words)) = buffer_parts(view) else {
        return false;
    };
    let (from, bytes, element) = extent(address, unordered, words);
    crate::shadow::find(address, len)
        .is_some_and(|buffer| matches!(plan(bytes, element, buffer.offset(from)), Copy::OwnBytes))
}

/// What a slot made by [`make`] views keeps alive.
pub(crate) enum Kept {
    /// Made, and keeps nothing alive (a sampler slot).
    Nothing,
    /// Made, viewing `resource`; `copy` is the copy of just the view's bytes
    /// it reads instead of its buffer's copy, if it made one.
    Resource {
        resource: ID3D12Resource,
        copy: Option<std::sync::Arc<crate::shadow::Shadow>>,
    },
}

/// The title memory a buffer shader view reads, from where its descriptor puts
/// it in the resource at `address`: its first element and the bytes its
/// elements take.
fn view_range_of(address: u64, desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> (u64, u64) {
    let element = element_size(desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    (
        address.saturating_add(buffer.FirstElement * element),
        u64::from(buffer.NumElements) * element,
    )
}

/// The host copy a buffer view of the resource at `address` (`len` bytes)
/// uses, the element its start is, and whether that copy holds only the
/// view's bytes. `extent` is the view's range and element size.
///
/// [`Copy::OwnBytes`] is for a view of memory the title placed inside another
/// object (`View::Placed`: methods 0x230 and 0x238), whose start need not be a
/// multiple of its stride. Its copy is a host copy like any other, so its own
/// refresh brings it up to date from title memory, what shaders write in it
/// goes back there after the submission, and a shader sees what the rest of
/// the pipeline sees.
fn view_copy(
    address: u64,
    len: u64,
    (from, bytes, element): (u64, u64, u64),
) -> Result<(std::sync::Arc<crate::shadow::Shadow>, u64, bool), &'static str> {
    // The view's own range is also the key `aligned` caches its copy by (the
    // one `buffer_range` gives `readable`).
    let buffer = crate::shadow::find(address, len).ok_or("no host copy of its buffer")?;
    match plan(bytes, element, buffer.offset(from)) {
        Copy::Element { first } => Ok((buffer, first, false)),
        Copy::OwnBytes => {
            let copy = crate::shadow::aligned(from, bytes).ok_or("no copy of the view")?;
            Ok((copy, 0, true))
        }
        Copy::None { why } => Err(why),
    }
}

/// What a buffer shader view reads.
enum Copy {
    /// The buffer's copy, at this element.
    Element { first: u64 },
    /// A copy of just the view's bytes, starting at the view.
    OwnBytes,
    /// Nothing, for the reason given.
    None { why: &'static str },
}

/// Where a buffer shader view's bytes come from: the buffer's copy when the
/// view starts on one of its elements (`element` bytes each, at `offset` in
/// the copy), else a copy of the view's own bytes.
fn plan(len: u64, element: u64, offset: u64) -> Copy {
    // Elements of a format whose size is not known: left null, as before.
    if element == 0 {
        return Copy::None {
            why: "elements of unknown size",
        };
    }
    if len == 0 {
        return Copy::None {
            why: "a view of no bytes",
        };
    }
    if offset.is_multiple_of(element) {
        return Copy::Element {
            first: offset / element,
        };
    }
    Copy::OwnBytes
}

/// Strides from `VICTORY_NULL_VIEW_STRIDES`: decimal or `0x` hex, comma
/// separated; anything else is named and skipped.
#[cfg(feature = "diagnostics")]
fn parsed_strides(value: &str) -> Vec<u32> {
    let mut out = Vec::new();
    for part in value.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        match part
            .strip_prefix("0x")
            .and_then(|hex| u32::from_str_radix(hex, 16).ok())
            .or_else(|| part.parse().ok())
        {
            Some(n) => out.push(n),
            None => tracing::warn!(
                part,
                "VICTORY_NULL_VIEW_STRIDES expects strides in decimal or 0x hex"
            ),
        }
    }
    out
}

/// With `VICTORY_NULL_VIEW_STRIDES` (diagnostics): buffer views of these
/// strides are left null, to confirm in the game which ones a pass needs.
/// Decimal or `0x` hex strides, comma separated.
fn nulled_stride(stride: u32) -> bool {
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = stride;
        false
    }
    #[cfg(feature = "diagnostics")]
    {
        if stride == 0 {
            return false;
        }
        static NULLED: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
        let nulled = NULLED.get_or_init(|| {
            parsed_strides(&std::env::var("VICTORY_NULL_VIEW_STRIDES").unwrap_or_default())
        });
        nulled.contains(&stride)
    }
}

fn kind(placed: bool) -> &'static str {
    if placed { "placed" } else { "registered" }
}

fn buffer_srv(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    placed: bool,
) -> Option<Kept> {
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.NumElements == 0 {
        return None;
    }
    let stride = buffer.StructureByteStride;
    if nulled_stride(stride) {
        once!(
            stride,
            "a buffer shader view left null (VICTORY_NULL_VIEW_STRIDES)"
        );
        return None;
    }
    let (shadow, first, own) = view_copy(address, len, shader_extent(address, &desc))
        .map_err(|why| {
            once!(
                view = kind(placed),
                address = format_args!("{address:#x}"),
                len,
                element,
                stride,
                why,
                "a buffer shader view left null"
            );
        })
        .ok()?;
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    // `first` is the element the view's own start is from the copy's start,
    // so it replaces the title's first element rather than adding to it: on a
    // copy of just the view's bytes that is 0, since the copy starts there.
    buffer.FirstElement = if own { 0 } else { first };
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateShaderResourceView(&shadow.buffer, Some(&desc), at)
    };
    Some(Kept::Resource {
        resource: shadow.buffer.clone(),
        copy: own.then(|| shadow.clone()),
    })
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; none if the copy cannot be
/// written by shaders (upload memory). What shaders write goes back to title
/// memory after the submission (`readable` declares the view's range).
fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    placed: bool,
) -> Option<Kept> {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.NumElements == 0 {
        return None;
    }
    let extent = unordered_extent(address, &desc);
    let (shadow, first, own) = view_copy(address, len, extent)
        .map_err(|why| {
            once!(
                view = kind(placed),
                address = format_args!("{address:#x}"),
                len,
                element = extent.2,
                why,
                "an unordered buffer view left null"
            );
        })
        .ok()?;
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    // As for a shader view: the element the view starts at in the copy.
    buffer.FirstElement = if own { 0 } else { first };
    buffer.CounterOffsetInBytes = 0;
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device.CreateUnorderedAccessView(
            &shadow.buffer,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
    Some(Kept::Resource {
        resource: shadow.buffer.clone(),
        copy: own.then(|| shadow.clone()),
    })
}

/// Adapts the requested view format to a format compatible with host D3D12.
///
/// Returns `true` if the view is permitted, mutating `format` to a valid typed format
/// when necessary (e.g. mapping a typeless view on a typeless texture to its primary
/// concrete format). Returns `false` if the view format belongs to an incompatible
/// family or cannot be created.
pub(crate) fn adapt_view_format(
    own: u32,
    mut format: Option<&mut DXGI_FORMAT>,
    unordered: bool,
) -> bool {
    use crate::resource::{family, first_concrete, is_typeless};
    let typeless = is_typeless(own);
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    if view == 0 && format.is_none() {
        return !typeless;
    }
    if view != 0 && family(view) != family(own) {
        once!(view, own, "view format of another family");
        return false;
    }
    if let Some(format) = format {
        if !typeless {
            let is_srgb_uav = unordered && crate::resource::SRGB.iter().any(|&(s, _)| s == own);
            if !is_srgb_uav {
                *format = DXGI_FORMAT(own as i32);
            }
        } else if view == 0 || is_typeless(view) {
            let concrete = first_concrete(own);
            tracing::debug!(
                own,
                view,
                concrete,
                "typeless view on typeless resource remapped to concrete format"
            );
            *format = DXGI_FORMAT(concrete as i32);
        }
    }
    true
}

fn supports_typed_uav(device: &ID3D12Device, format: DXGI_FORMAT) -> bool {
    let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        Format: format,
        ..Default::default()
    };
    // SAFETY: `support` is sized and initialized for D3D12_FEATURE_FORMAT_SUPPORT.
    unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_FORMAT_SUPPORT,
            (&raw mut support).cast(),
            std::mem::size_of_val(&support) as u32,
        )
    }
    .is_ok()
        && support.Support1.0 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW.0 != 0
}

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in another format than a
/// typed resource's. A view of a typed texture in another format of its
/// family takes the texture's format.
/// shortcut: such a view (sRGB of UNORM, say) reads without its
/// conversion; cast with castable formats if colors come out wrong.
fn fits(
    device: &ID3D12Device,
    texture: &ID3D12Resource,
    mut format: Option<&mut DXGI_FORMAT>,
    unordered: bool,
) -> bool {
    use crate::resource::{DEPTH_FORMATS, family, is_typeless};
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        return false;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    if DEPTH_FORMATS.iter().any(|&(typed, _)| typed == view) {
        once!(unordered, view, own, "view the host does not allow");
        return false;
    }
    let typeless = is_typeless(own);
    if family(view) != family(own) || (typeless && is_typeless(view)) {
        once!(view, own, "view format of another family");
        return false;
    }
    if view == 0 {
        return !typeless && (!unordered || supports_typed_uav(device, d.Format));
    }
    if !adapt_view_format(own, format.as_deref_mut(), unordered) {
        return false;
    }
    let effective = format.as_deref().copied().unwrap_or(d.Format);
    if unordered && !supports_typed_uav(device, effective) {
        once!(
            view = effective.0,
            own,
            "view format does not support typed unordered access"
        );
        return false;
    }
    true
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
    match view {
        // make_at and the mirror path write a typed null for this view.
        View::Constant(0, _) => None,
        &View::Constant(address, size) => {
            let shadow = crate::shadow::find(address, u64::from(size))?;
            let gpu = shadow.gpu(address);
            if gpu % 256 != 0 {
                once!(
                    address = format_args!("{address:#x}"),
                    "constant buffer not 256-byte aligned on the host"
                );
                return None;
            }
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                BufferLocation: gpu,
                SizeInBytes: size.next_multiple_of(256),
            };
            // SAFETY: a slot of our heap; the range lies in the host copy.
            unsafe { device.CreateConstantBufferView(Some(&desc), at) };
            Some(Kept::Resource {
                resource: shadow.buffer.clone(),
                copy: None,
            })
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::object::This;
            if let Some(texture) = crate::resource::host_resource(resource) {
                let mut desc = desc.as_ref().map(srv_desc);
                if let Some(desc) = desc.as_mut()
                    && desc.Format.0 == 0
                    && let Some(format) = crate::resource::view_format(resource)
                {
                    desc.Format = DXGI_FORMAT(format as i32);
                }
                if !fits(device, texture, desc.as_mut().map(|d| &mut d.Format), false) {
                    return None;
                }
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    )
                };
                Some(Kept::Resource {
                    resource: texture.clone(),
                    copy: None,
                })
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at, false)
            } else {
                None
            }
        }
        View::Unordered(resource, _counter, desc) => {
            let resource = *resource as crate::object::This;
            let Some(texture) = crate::resource::host_resource(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at, false)
                    }
                    _ => None,
                };
            };
            let mut desc = desc.as_ref().map(uav_desc);
            if let Some(desc) = desc.as_mut()
                && desc.Format.0 == 0
                && let Some(format) = crate::resource::view_format(resource)
            {
                desc.Format = DXGI_FORMAT(format as i32);
            }
            if !fits(device, texture, desc.as_mut().map(|d| &mut d.Format), true) {
                return None;
            }
            // The host refuses (and removes the device for) a typed unordered
            // view in a format it cannot store to: the packed 16-bit ones
            // (B5G5R5A1, B4G4R4A4), 2026-11-24. Such a view is null.
            // SAFETY: a live resource.
            let final_format = desc
                .as_ref()
                .map(|d| d.Format)
                .filter(|&f| f.0 != 0)
                .unwrap_or_else(|| unsafe { texture.GetDesc() }.Format);
            if !supports_typed_uav(device, final_format) {
                once!(format = final_format.0, "an unordered view in a format the host cannot store");
                return None;
            }
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            Some(Kept::Resource {
                resource: texture.clone(),
                copy: None,
            })
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(Kept::Nothing)
        }
        View::ConsoleSampler(words) => {
            let desc = crate::sampler::console(words);
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(Kept::Nothing)
        }
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return make(&view, at);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            // The view names its own start: it can land off an element.
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at, true)
            } else {
                buffer_srv(address, len, srv_desc(words), at, true)
            }
        }
    }
}

/// A shader view of a texture the title placed in memory itself, as a view
/// of the resource made for that memory (on first use).
pub(crate) fn placed_texture(view: &View) -> Option<View> {
    const BUFFER: u32 = 1;
    let View::Placed(unordered, resource, desc) = view else {
        return None;
    };
    if resource[0] == BUFFER {
        return None;
    }
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let texture = crate::resource::texture_at(address, resource)?;
    if *unordered {
        Some(View::Unordered(
            texture as usize,
            0,
            desc[..10].try_into().ok(),
        ))
    } else {
        Some(View::Shader(texture as usize, desc[..10].try_into().ok()))
    }
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or a 2D texture) or a shader
/// view. Writes to it are dropped, reads give zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    // Unordered views: whether a buffer, and the title's view dimension.
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => (resource[0] == 1, Some(desc[1])),
        // Shader views of buffers read nothing through a null buffer view:
        // a null 2D view where the shader expects a buffer is a mismatch
        // the hardware does not survive in general (GPU-based validation
        // reports it).
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            return null_buffer_view(at);
        }
        View::Placed(false, resource, _) if resource[0] == 1 => return null_buffer_view(at),
        // Textures: a null view of the title's dimension (1D, arrays, 3D,
        // cubes), not always 2D.
        View::Shader(_, Some(desc)) => return null_like(desc, at),
        View::Placed(false, _, desc) => {
            return null_like(desc[..10].try_into().unwrap_or(&[0; 10]), at);
        }
        View::Constant(..) => {
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
            // SAFETY: a slot of our heap; a zero location makes a null view.
            return unsafe { host.device.CreateConstantBufferView(Some(&desc), at) };
        }
        _ => return null_view(at),
    };
    let desc = if buffer {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            ..Default::default()
        }
    } else {
        null_uav_desc(dimension)
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A null texture UAV of the title's view dimension (1D, 1D or 2D arrays,
/// 3D), 2D if it names none: a 2D null view where the shader writes a 3D
/// texture is a mismatch GPU-based validation reports (the world's wind
/// field, CS_CopyWindFieldBeforeUpdateWithSmooth, 2026-11-24).
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A raw unordered access view of a buffer that drops writes and reads
/// zeros.
fn null_raw_uav(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_UAV {
                NumElements: 1,
                Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The title's slots past a capped host heap are counted, and the log
    /// names the first of them and every doubling after it: a warning per
    /// skip floods the log (310 720 lines in one world load).
    #[test]
    fn the_slots_past_a_capped_heap_are_counted_and_the_log_names_a_few() {
        let past = PastHeap::new();
        assert_eq!(past.get(), (0, 0, 0));
        assert_eq!(past.note(1_245_183, 934_464), (1, 1_245_183, 934_464));
        assert_eq!(past.note(934_464, 934_464), (2, 1_245_183, 934_464));
        assert_eq!(past.get(), (2, 1_245_183, 934_464));
        // The warnings: the first skip and then every doubling.
        assert!(worth_a_warning(1));
        assert!(worth_a_warning(2));
        assert!(!worth_a_warning(3));
        assert!(worth_a_warning(4));
        assert!(!worth_a_warning(5));
        assert!(worth_a_warning(1 << 20));
        assert!(!worth_a_warning((1 << 20) + 1));
        // A run with 2 097 152 skips names 22 of them.
        let named = (1..=2_097_152).filter(|n| worth_a_warning(*n)).count();
        assert_eq!(named, 22);
        // The title's slots are numbered from zero: the ones past the heap
        // are those from the heap's last slot to the highest index seen.
        assert_eq!(slots_past(934_464, 1_245_183), 310_720);
        assert_eq!(slots_past(934_464, 934_463), 0);
        assert_eq!(slots_past(934_464, 934_464), 1);
    }

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// A buffer view of no elements, which the host refuses (removing the
    /// device), makes a null slot; one of an element is made as usual.
    #[test]
    fn buffer_views_of_no_elements_are_null() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        // A title buffer the view's memory lies in (its copy is made on the
        // host; title memory is not read).
        let (address, len) = (0x7e00_0000_0000_u64, 0x1000_u64);
        crate::shadow::register(usize::MAX - 1, address, len);
        let mut resource = [0u32; 14];
        resource[0] = 1; // a buffer
        resource[4] = len as u32;
        let placed = |unordered: bool, elements: u32| {
            let mut desc = [0u32; 16];
            desc[1] = 1; // buffer dimension
            if unordered {
                // NumElements, StructureByteStride
                (desc[4], desc[5]) = (elements, 16);
            } else {
                desc[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                (desc[6], desc[7]) = (elements, 16);
            }
            (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
            View::Placed(unordered, resource, desc)
        };
        let kept = |index: u32| heap.kept.lock().unwrap()[index as usize].is_some();
        for unordered in [false, true] {
            heap.make_at(0, &placed(unordered, 1));
            assert!(kept(0), "a view of an element (unordered: {unordered})");
            heap.make_at(1, &placed(unordered, 0));
            assert!(!kept(1), "a view of no elements (unordered: {unordered})");
            // SAFETY: a plain device query.
            let removed = unsafe { host.device.GetDeviceRemovedReason() };
            assert!(
                removed.is_ok(),
                "device removed: {removed:?} (unordered: {unordered})"
            );
        }
        crate::shadow::unregister(usize::MAX - 1, address);
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let a = heap.dynamic(heap.dynamic - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (4, 4 + heap.dynamic - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        heap.release(a, 50);
        assert_eq!(heap.dynamic(50), Some(4));
        assert_eq!(heap.dynamic(40), Some(4 + heap.dynamic - 40));
    }

    /// A contiguous free range crossing the dynamic search cursor is found (#871).
    #[test]
    fn dynamic_allocator_finds_range_crossing_search_cursor() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        {
            let mut guard = heap.next.lock().unwrap();
            guard.0 = 3;
            guard.1.fill(true);
            guard.1[1..=4].fill(false);
        }
        // Slots 1..=4 are free, cursor is at 3. Requesting 4 slots must succeed at index count + 1.
        assert_eq!(heap.dynamic(4), Some(heap.count + 1));
    }

    /// A large requested descriptor count is capped at PORTABLE_VIEWS
    /// rather than creating a heap that violates D3D12 limits (#735).
    #[test]
    fn large_view_heap_is_capped_at_portable_limit() {
        if host().is_none() {
            return;
        }
        if let Some(heap) = HostHeap::new(false, 1_048_576) {
            assert!(heap.count + heap.dynamic + ALIASES < PORTABLE_VIEWS);
            assert_eq!(heap.count, PORTABLE_VIEWS - heap.dynamic - ALIASES - 1);
        }
    }

    /// An unordered view off an element of its buffer's copy writes a copy of
    /// just its bytes, which a shader view of the same bytes shares; one on
    /// an element writes the buffer's copy. The device takes all of them.
    #[test]
    fn an_unordered_view_off_an_element_writes_its_own_copy() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        // A title buffer the views' memory lies in (its copy is made on the
        // host; title memory is not read).
        let (address, len) = (0x7d00_0000_0000_u64, 0x1_0000_u64);
        crate::shadow::register(usize::MAX - 2, address, len);
        // Memory placed `start` bytes into it, of four 56-byte elements.
        let placed = |unordered: bool, start: u64| {
            let mut resource = [0u32; 14];
            resource[0] = 1; // a buffer
            resource[4] = 4 * 56;
            let mut desc = [0u32; 16];
            desc[1] = 1; // buffer dimension
            if unordered {
                // NumElements, StructureByteStride
                (desc[4], desc[5]) = (4, 56);
            } else {
                desc[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                (desc[6], desc[7]) = (4, 56);
            }
            let at = address + start;
            (desc[10], desc[11]) = (at as u32, (at >> 32) as u32);
            View::Placed(unordered, resource, desc)
        };
        let made = |index: usize| heap.kept.lock().unwrap()[index].is_some();
        let copy = |index: usize| heap.kept_copy.lock().unwrap()[index].clone();
        heap.make_at(0, &placed(true, 112));
        assert!(
            made(0) && copy(0).is_none(),
            "on an element: the buffer's copy"
        );
        assert!(!off_element(&placed(true, 112)));
        // The hit-detection offset: 48 bytes in.
        heap.make_at(1, &placed(true, 48));
        heap.make_at(2, &placed(false, 48));
        let uav = copy(1).expect("an unordered view off an element: its own copy");
        let srv = copy(2).expect("a shader view off an element: its own copy");
        assert!(
            std::sync::Arc::ptr_eq(&uav, &srv),
            "one copy of the same bytes"
        );
        assert!(off_element(&placed(true, 48)) && off_element(&placed(false, 48)));
        // SAFETY: a plain device query.
        let removed = unsafe { host.device.GetDeviceRemovedReason() };
        assert!(removed.is_ok(), "device removed: {removed:?}");
        crate::shadow::unregister(usize::MAX - 2, address);
    }

    /// A nulled-stride list parses decimal and hex, and ignores junk.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn nulled_strides_parse_decimal_and_hex() {
        assert_eq!(parsed_strides("56, 336,0x20"), vec![56, 336, 32]);
        assert_eq!(parsed_strides(" nope , 56"), vec![56]);
        assert!(parsed_strides("").is_empty());
    }

    /// A view's range is its own elements where its descriptor puts them, and
    /// `buffer_range` (what `readable` refreshes) gives the same one, so the
    /// copy `buffer_srv` or `buffer_uav` makes is found again for the refresh.
    #[test]
    fn a_views_range_is_the_one_its_copy_is_keyed_by() {
        // Whatever the words mean, both paths read them the same way: the
        // copy's key and the range `readable` refreshes cannot diverge.
        for words in [
            [0u32, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0u32, 4, 2, 0, 1, 56, 0, 0, 0, 0],
            [0u32, 4, 0, 0, 64, 336, 0, 0, 0, 0],
        ] {
            let mut resource = [0u32; 14];
            resource[0] = 1; // a buffer
            let mut placed = [0u32; 16];
            placed[..10].copy_from_slice(&words);
            placed[10] = 0x1000;
            let range = |(from, bytes, _): (u64, u64, u64)| Some((from, bytes));
            assert_eq!(
                buffer_range(&View::Placed(false, resource, placed)),
                range(shader_extent(0x1000, &srv_desc(&words)))
            );
            assert_eq!(
                buffer_range(&View::Placed(true, resource, placed)),
                range(unordered_extent(0x1000, &uav_desc(&words)))
            );
        }
    }

    /// A sampler heap larger than 2048 descriptors is capped at PORTABLE_SAMPLERS (#744).
    #[test]
    fn sampler_heap_larger_than_2048_is_capped() {
        if host().is_none() {
            return;
        }
        if let Some(heap) = HostHeap::new(true, 4096) {
            assert_eq!(heap.count, PORTABLE_SAMPLERS);
        }
    }

    /// Buffer views mirror on creation when their host copy exists, so
    /// shaders indexing the heap directly at run time read the real view
    /// rather than a null descriptor.
    /// A buffer view without a host shadow falls back to null, keeping its
    /// key at 0 so it can be retried once the memory is registered.
    #[test]
    fn buffer_views_mirror_on_creation() {
        if host().is_none() {
            return;
        }
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let mut title_slot = [0u64; 4];
        let s0 = title_slot.as_mut_ptr() as u64;
        super::super::create_srv(std::ptr::null(), std::ptr::null(), std::ptr::null(), s0);
        // SAFETY: s0 is our stack-allocated array.
        let [tag, _i, _generation] = unsafe { *(s0 as *const [u32; 3]) };
        assert_eq!(tag, VIEW_TAG);
        heap.mirror(0, s0, false);
        // Resource is null, so make() returns None and made[0] stays 0 for later retry.
        assert_eq!(heap.made.lock().unwrap()[0], 0);
    }

    /// Typeless SRV and UAV views on typeless resources are remapped to the first
    /// concrete format of their family so D3D12 can create the view without E_INVALIDARG.
    #[test]
    fn typeless_view_on_typeless_resource_remaps_to_concrete_format() {
        // R8G8B8A8_TYPELESS (27) with R8G8B8A8_TYPELESS view (27) -> R8G8B8A8_UNORM (28)
        let mut fmt = DXGI_FORMAT(27);
        assert!(adapt_view_format(27, Some(&mut fmt), false));
        assert_eq!(fmt.0, 28);

        // B8G8R8A8_TYPELESS (90) with B8G8R8A8_TYPELESS view (90) -> B8G8R8A8_UNORM (87)
        let mut fmt = DXGI_FORMAT(90);
        assert!(adapt_view_format(90, Some(&mut fmt), false));
        assert_eq!(fmt.0, 87);

        // R32_TYPELESS (39) with R32_TYPELESS view (39) -> R32_FLOAT (41)
        let mut fmt = DXGI_FORMAT(39);
        assert!(adapt_view_format(39, Some(&mut fmt), false));
        assert_eq!(fmt.0, 41);

        // UNKNOWN view (0) on typeless texture -> first concrete format
        let mut fmt = DXGI_FORMAT(0);
        assert!(adapt_view_format(27, Some(&mut fmt), false));
        assert_eq!(fmt.0, 28);

        // Typed view (29 = R8G8B8A8_UNORM_SRGB) on typeless texture (27) stays 29
        let mut fmt = DXGI_FORMAT(29);
        assert!(adapt_view_format(27, Some(&mut fmt), false));
        assert_eq!(fmt.0, 29);

        // Typeless view (27) on typed texture (28) adopts texture's format (28)
        let mut fmt = DXGI_FORMAT(27);
        assert!(adapt_view_format(28, Some(&mut fmt), false));
        assert_eq!(fmt.0, 28);

        // Typeless texture with None format is refused
        assert!(!adapt_view_format(27, None, false));

        // Incompatible family is refused
        let mut fmt = DXGI_FORMAT(10);
        assert!(!adapt_view_format(27, Some(&mut fmt), false));

        // UAV with sRGB (29) converts to linear (28)
        let mut fmt = DXGI_FORMAT(29);
        assert!(adapt_view_format(27, Some(&mut fmt), true));
        assert_eq!(fmt.0, 28);
    }

    /// When dynamic slots are exhausted, recycling provides a ring-buffer fallback
    /// instead of dropping the draw.
    #[test]
    fn dynamic_descriptor_recycling_survives_pool_exhaustion() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let full = heap.dynamic(heap.dynamic).unwrap();
        assert_eq!(full, 4);
        // Standard allocation fails because the pool is full.
        assert_eq!(heap.dynamic(10), None);
        // Recycling succeeds and wraps cleanly within dynamic bounds.
        let recycled = heap.recycle_dynamic(10).unwrap();
        assert_eq!(recycled, 4);
        let recycled2 = heap.recycle_dynamic(10).unwrap();
        assert_eq!(recycled2, 14);
    }

    /// Contiguous spans that cross the search cursor are found.
    #[test]
    fn dynamic_cursor_wraparound_finds_contiguous_span() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        // Fill the heap in two parts: first dynamic - 10 slots, then 10 slots.
        let a = heap.dynamic(heap.dynamic - 10).unwrap();
        let b = heap.dynamic(10).unwrap();
        assert_eq!(a, 4);
        assert_eq!(b, 4 + heap.dynamic - 10);
        // Release the first part.
        heap.release(a, heap.dynamic - 10);
        // Request 10 slots: finds them in the freed range at the start.
        let c = heap.dynamic(10).unwrap();
        assert_eq!(c, 4);
    }

    /// Requesting more than the host limit (> 1,000,000) caps gracefully without panicking.
    #[test]
    fn heap_request_over_one_million_caps_gracefully() {
        let Some(heap) = HostHeap::new(false, 1_245_184) else {
            return; // no host GPU or device failed
        };
        assert!(heap.count + heap.dynamic <= PORTABLE_VIEWS);
        assert!(heap.count > 0);
        assert!(heap.dynamic > 0);
        // Mirroring a slot past the capped host count handles it gracefully without panic.
        let mut dummy_handle: [u32; 3] = [0, 0, 0];
        let handle_ptr = (&raw mut dummy_handle) as u64;
        assert!(heap.mirror(1_245_000, handle_ptr, false).is_none());
    }

    /// Placed texture views produce real UAVs and SRVs instead of dropping to null.
    #[test]
    fn placed_texture_views_produce_uav_and_srv() {
        let dummy_tex = 0x1234_5678_usize;
        let address = 0x8000_1000_u64;
        let mut resource = [0u32; 14];
        resource[0] = 2; // Texture2D, not BUFFER (1)
        let mut desc = [0u32; 16];
        desc[0] = 28; // DXGI_FORMAT_R8G8B8A8_UNORM
        desc[10] = address as u32;
        desc[11] = (address >> 32) as u32;

        // Populate PLACED_TEXTURES map with our dummy resource
        crate::resource::PLACED_TEXTURES
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert((address, resource), dummy_tex);

        // SRV: View::Placed(false, ...) -> View::Shader
        let srv_view = View::Placed(false, resource, desc);
        let srv_res = placed_texture(&srv_view);
        assert_eq!(
            srv_res,
            Some(View::Shader(dummy_tex, desc[..10].try_into().ok()))
        );

        // UAV: View::Placed(true, ...) -> View::Unordered
        let uav_view = View::Placed(true, resource, desc);
        let uav_res = placed_texture(&uav_view);
        assert_eq!(
            uav_res,
            Some(View::Unordered(dummy_tex, 0, desc[..10].try_into().ok()))
        );

        // Buffer: resource[0] == 1 -> None
        let mut buf_resource = resource;
        buf_resource[0] = 1;
        let buf_view = View::Placed(true, buf_resource, desc);
        assert_eq!(placed_texture(&buf_view), None);
    }

    fn placed_tex_desc(width: u64, height: u32, format: u32, flags: u32) -> [u32; 14] {
        [
            3,
            0,
            0,
            0,
            width as u32,
            (width >> 32) as u32,
            height,
            1 | (1 << 16),
            format,
            1,
            0,
            0,
            flags,
            0,
        ]
    }

    fn placed_tex_uav_desc(va: u64, view_format: u32) -> [u32; 16] {
        let mut desc = [0u32; 16];
        desc[0] = view_format;
        desc[1] = 4; // D3D12_UAV_DIMENSION_TEXTURE2D
        desc[10] = va as u32;
        desc[11] = (va >> 32) as u32;
        desc
    }

    fn placed_tex_srv_desc(va: u64, view_format: u32) -> [u32; 16] {
        let mut desc = [0u32; 16];
        desc[0] = view_format;
        desc[1] = 4; // D3D12_SRV_DIMENSION_TEXTURE2D
        desc[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        desc[5] = 1; // MipLevels
        desc[10] = va as u32;
        desc[11] = (va >> 32) as u32;
        desc
    }

    fn compile_compute(
        source: &str,
        device: &ID3D12Device,
    ) -> (ID3D12RootSignature, ID3D12PipelineState) {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::core::s;

        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        // SAFETY: source and entrypoint names are valid for the duration of the call.
        let compiled = unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("placed_uav_test"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                Some(&mut errors),
            )
        };
        if let Err(e) = compiled {
            let msg = errors
                .map(|b| {
                    // SAFETY: blob owns its error message bytes.
                    unsafe {
                        String::from_utf8_lossy(std::slice::from_raw_parts(
                            b.GetBufferPointer().cast::<u8>(),
                            b.GetBufferSize(),
                        ))
                        .into_owned()
                    }
                })
                .unwrap_or_default();
            panic!("D3DCompile failed ({e}): {msg}");
        }
        let code = code.unwrap();
        // SAFETY: blob owns compiled DXBC with embedded root signature.
        unsafe {
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes).unwrap();
            let pso = device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        }
    }

    #[test]
    fn placed_uav_textures_create_real_host_uavs() {
        let Some(host) = host() else {
            return;
        };
        let Some(heap) = HostHeap::new(false, 32) else {
            return;
        };
        let cases = [
            (28u32, 28u32, 4u32, true),  // R8G8B8A8_UNORM
            (27u32, 28u32, 4u32, true),  // R8G8B8A8_TYPELESS -> R8G8B8A8_UNORM
            (29u32, 29u32, 4u32, true),  // R8G8B8A8_UNORM_SRGB -> linear R8G8B8A8_UNORM
            (29u32, 0u32, 4u32, true),   // R8G8B8A8_UNORM_SRGB + UNKNOWN -> linear 28
            (91u32, 91u32, 4u32, true),  // B8G8R8A8_UNORM_SRGB -> linear B8G8R8A8_UNORM
            (91u32, 0u32, 4u32, true),   // B8G8R8A8_UNORM_SRGB + UNKNOWN -> linear 87
            (28u32, 28u32, 0u32, false), // missing ALLOW_UNORDERED_ACCESS -> null fallback
            (93u32, 93u32, 4u32, false), // B8G8R8X8_UNORM_SRGB -> unsupported typed UAV
            (93u32, 0u32, 4u32, false),  // B8G8R8X8_UNORM_SRGB + UNKNOWN -> unsupported typed UAV
            (92u32, 88u32, 4u32, false), // B8G8R8X8_TYPELESS -> B8G8R8X8_UNORM unsupported UAV
            (92u32, 93u32, 4u32, false), // B8G8R8X8_TYPELESS -> B8G8R8X8_UNORM_SRGB unsupported UAV
            (40u32, 40u32, 4u32, false), // D32_FLOAT + UAV flag (no depth flag) -> null fallback
            (55u32, 55u32, 4u32, false), // D16_UNORM + UAV flag (no depth flag) -> null fallback
            (39u32, 40u32, 4u32, false), // R32_TYPELESS + D32_FLOAT view -> null fallback
            (53u32, 55u32, 4u32, false), // R16_TYPELESS + D16_UNORM view -> null fallback
            (27u32, 27u32, 4u32, false), // R8G8B8A8_TYPELESS + typeless view -> null fallback
        ];
        let mut slots = [[0u32; 8]; 24];
        for (idx, (res_fmt, view_fmt, flags, expect_real)) in cases.into_iter().enumerate() {
            let backing = Box::leak(Box::new([0u8; 256]));
            let va = backing.as_ptr() as u64;
            let resource = placed_tex_desc(4, 4, res_fmt, flags);
            let desc = placed_tex_uav_desc(va, view_fmt);
            let slot_addr = slots[idx].as_mut_ptr() as u64;
            crate::descriptor::create_placed_uav(std::ptr::null(), &resource, &desc, slot_addr);
            let view = crate::descriptor::view(slot_addr).unwrap();
            assert!(matches!(
                placed_texture(&view),
                Some(View::Unordered(_, 0, Some(_)))
            ));
            assert!(heap.mirror(idx as u32, slot_addr, false).is_some());
            assert!(
                heap.kept.lock().unwrap()[idx].is_none(),
                "case {idx} mirror(false) should defer host creation"
            );
            let mirrored = heap.mirror(idx as u32, slot_addr, true).unwrap();
            let dyn_slot = heap.count() + idx as u32;
            heap.make_at(dyn_slot, &mirrored);
            let kept = heap.kept.lock().unwrap();
            assert_eq!(kept[idx].is_some(), expect_real, "case {idx} mirror");
            assert_eq!(
                kept[dyn_slot as usize].is_some(),
                expect_real,
                "case {idx} make_at"
            );
        }
        for (offset, (res_fmt, view_fmt, expect_real)) in [
            (39u32, 40u32, false), // R32_TYPELESS + D32_FLOAT SRV -> null fallback
            (53u32, 55u32, false), // R16_TYPELESS + D16_UNORM SRV -> null fallback
            (29u32, 0u32, true),   // R8G8B8A8_UNORM_SRGB + UAV + UNKNOWN SRV -> 29
            (91u32, 0u32, true),   // B8G8R8A8_UNORM_SRGB + UAV + UNKNOWN SRV -> 91
        ]
        .into_iter()
        .enumerate()
        {
            let idx = cases.len() + offset;
            let backing = Box::leak(Box::new([0u8; 256]));
            let va = backing.as_ptr() as u64;
            let resource = placed_tex_desc(4, 4, res_fmt, 4);
            let desc = placed_tex_srv_desc(va, view_fmt);
            let slot_addr = slots[idx].as_mut_ptr() as u64;
            crate::descriptor::create_placed_srv(std::ptr::null(), &resource, &desc, slot_addr);
            heap.mirror(idx as u32, slot_addr, true).unwrap();
            assert_eq!(
                heap.kept.lock().unwrap()[idx].is_some(),
                expect_real,
                "srv case {offset}"
            );
        }
        // SAFETY: plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    #[test]
    fn compute_dispatch_writes_placed_texture_and_buffer_uavs() {
        use crate::object::{S_OK, This, guid, slot};
        use crate::resource::{HeapProperties, ResourceDesc, create_committed};
        use std::ffi::c_void;
        use victory_shader::{ResourceType, RootType, UsedResource};

        let Some(host) = host() else {
            return;
        };
        let Some(heap) = HostHeap::new(false, 8) else {
            return;
        };

        let tex1_mem = Box::leak(Box::new([0x11u8; 256]));
        let tex1_va = tex1_mem.as_ptr() as u64;
        let tex1_res = placed_tex_desc(4, 4, 28, 4);
        let tex1_uav_desc = placed_tex_uav_desc(tex1_va, 28);
        let tex1_srv_desc = placed_tex_srv_desc(tex1_va, 28);

        let tex2_mem = Box::leak(Box::new([0x22u8; 256]));
        let tex2_va = tex2_mem.as_ptr() as u64;
        // sRGB 29 + ALLOW_UNORDERED_ACCESS + DXGI_FORMAT_UNKNOWN (0) -> linear R8G8B8A8_UNORM (28)
        let tex2_res = placed_tex_desc(4, 4, 29, 4);
        let tex2_uav_desc = placed_tex_uav_desc(tex2_va, 0);

        let buf_res = [1u32, 0, 0, 0, 256, 0, 1, 1 | (1 << 16), 0, 1, 0, 1, 0, 0];
        let props = [1u32, 0, 0, 0, 0];
        let resource_iid = guid(
            0x6964_42be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let mut buf_obj: This = std::ptr::null();
        assert_eq!(
            create_committed(
                std::ptr::null(),
                props.as_ptr().cast::<HeapProperties>(),
                0,
                buf_res.as_ptr().cast::<ResourceDesc>(),
                0,
                std::ptr::null(),
                &resource_iid,
                &mut buf_obj,
            ),
            S_OK
        );
        let mut buf_ptr: *mut u8 = std::ptr::null_mut();
        type Map = extern "system" fn(This, u32, *const c_void, *mut *mut u8) -> i32;
        assert_eq!(
            slot::<Map>(buf_obj, 0x40)(buf_obj, 0, std::ptr::null(), &mut buf_ptr),
            S_OK
        );
        let buf_va = buf_ptr as u64;
        let mut buf_uav_desc = [0u32; 16];
        buf_uav_desc[0] = 42; // DXGI_FORMAT_R32_UINT
        buf_uav_desc[1] = 1; // D3D12_UAV_DIMENSION_BUFFER
        buf_uav_desc[4] = 64; // NumElements
        buf_uav_desc[10] = buf_va as u32;
        buf_uav_desc[11] = (buf_va >> 32) as u32;

        let mut title_slots = [[0u32; 8]; 3];
        let slot0 = title_slots[0].as_mut_ptr() as u64;
        let slot1 = title_slots[1].as_mut_ptr() as u64;
        let slot2 = title_slots[2].as_mut_ptr() as u64;
        crate::descriptor::create_placed_uav(std::ptr::null(), &tex1_res, &tex1_uav_desc, slot0);
        crate::descriptor::create_placed_uav(std::ptr::null(), &buf_res, &buf_uav_desc, slot1);
        crate::descriptor::create_placed_srv(std::ptr::null(), &tex1_res, &tex1_srv_desc, slot2);

        let view0 = heap.mirror(0, slot0, true).unwrap();
        let view1 = heap.mirror(1, slot1, true).unwrap();
        let view2 = heap.mirror(2, slot2, true).unwrap();
        let view3 = View::Placed(true, tex2_res, tex2_uav_desc);
        heap.make_at(3, &view3);

        let (root1, pso1) = compile_compute(
            r#"
            #define RS "DescriptorTable(UAV(u0, numDescriptors=2))"
            RWTexture2D<float4> outTex : register(u0);
            RWBuffer<uint> outBuf : register(u1);
            [RootSignature(RS)]
            [numthreads(4, 4, 1)]
            void main(uint3 id : SV_DispatchThreadID) {
                outTex[id.xy] = float4((id.x + 1.0) / 4.0, (id.y + 1.0) / 4.0, 0.25, 1.0);
                if (id.x == 0 && id.y == 0) outBuf[0] = 0x1234ABCDu;
            }
            "#,
            &host.device,
        );
        let (root2, pso2) = compile_compute(
            r#"
            #define RS "DescriptorTable(SRV(t0, numDescriptors=1)), DescriptorTable(UAV(u0, numDescriptors=1))"
            Texture2D<float4> inTex : register(t0);
            RWTexture2D<float4> outTex : register(u0);
            [RootSignature(RS)]
            [numthreads(4, 4, 1)]
            void main(uint3 id : SV_DispatchThreadID) {
                outTex[id.xy] = inTex.Load(int3(id.xy, 0));
            }
            "#,
            &host.device,
        );

        let readback_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let readback_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 2048,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain readback buffer creation.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &readback_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &readback_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
        }
        let readback = readback.unwrap();

        let list = host.new_list().unwrap();
        let uav_used = UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        };
        let srv_used = UsedResource {
            kind: ResourceType::Srv,
            ..uav_used
        };

        // SAFETY: live heap, root signatures, PSOs, and resources on an open list.
        unsafe {
            let gpu_start = heap.heap.GetGPUDescriptorHandleForHeapStart();
            let gpu_at = |idx: u32| D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: gpu_start.ptr + u64::from(idx) * u64::from(heap.step),
            };
            list.list.SetDescriptorHeaps(&[Some(heap.heap.clone())]);

            crate::draw::readable(&list, &uav_used, &view0);
            crate::draw::readable(&list, &uav_used, &view1);
            list.list.SetComputeRootSignature(&root1);
            list.list.SetPipelineState(&pso1);
            list.list.SetComputeRootDescriptorTable(0, gpu_at(0));
            list.list.Dispatch(1, 1, 1);

            crate::draw::readable(&list, &srv_used, &view2);
            crate::draw::readable(&list, &uav_used, &view3);
            list.list.SetComputeRootSignature(&root2);
            list.list.SetPipelineState(&pso2);
            list.list.SetComputeRootDescriptorTable(0, gpu_at(2));
            list.list.SetComputeRootDescriptorTable(1, gpu_at(3));
            list.list.Dispatch(1, 1, 1);

            let tex1_obj = crate::resource::texture_at(tex1_va, &tex1_res).unwrap();
            let tex2_obj = crate::resource::texture_at(tex2_va, &tex2_res).unwrap();
            crate::list::ensure_state(&list, tex1_obj, 0x800);
            crate::list::ensure_state(&list, tex2_obj, 0x800);

            for (offset, tex_obj) in [(0u64, tex1_obj), (1024u64, tex2_obj)] {
                let host_tex = crate::resource::host_resource(tex_obj).unwrap();
                let to = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&readback),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: offset,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                                Width: 4,
                                Height: 4,
                                Depth: 1,
                                RowPitch: 256,
                            },
                        },
                    },
                };
                let from = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(host_tex),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        SubresourceIndex: 0,
                    },
                };
                list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            }
        }
        list.close();
        host.execute(&[&list]);
        // SAFETY: plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());

        let mut mapped = std::ptr::null_mut();
        // SAFETY: execution completed via execute(); readback buffer is 2048 bytes.
        let bytes = unsafe {
            readback
                .Map(
                    0,
                    Some(&D3D12_RANGE {
                        Begin: 0,
                        End: 2048,
                    }),
                    Some(&mut mapped),
                )
                .unwrap();
            std::slice::from_raw_parts(mapped.cast::<u8>(), 2048)
        };
        let expected = [64u8, 128, 191, 255];
        for (name, base) in [
            ("placed UAV texture", 0usize),
            ("placed SRV->UAV texture", 1024),
        ] {
            for row in 0..4 {
                for col in 0..4 {
                    let at = base + row * 256 + col * 4;
                    let px = &bytes[at..at + 4];
                    assert!(
                        px[0].abs_diff(expected[col]) <= 1
                            && px[1].abs_diff(expected[row]) <= 1
                            && px[2].abs_diff(64) <= 1
                            && px[3] == 255,
                        "{name} pixel ({col},{row}) mismatch: {px:?}"
                    );
                }
            }
        }
        // SAFETY: `readback` was mapped at subresource 0 above.
        unsafe { readback.Unmap(0, None) };
        // SAFETY: buf_ptr points to at least 256 bytes of committed title memory.
        assert_eq!(unsafe { *buf_ptr.cast::<u32>() }, 0x1234_ABCD);
        type Release = extern "system" fn(This) -> u32;
        assert_eq!(slot::<Release>(buf_obj, 0x10)(buf_obj), 0);
    }

    #[test]
    fn gpu_ranges_unregister_removes_only_matching_heap() {
        let mut ranges = Vec::new();
        register_gpu_range(&mut ranges, 1, 0x1000, 0x400, 32, false);
        register_gpu_range(&mut ranges, 2, 0x1000, 0x400, 32, false);
        register_gpu_range(&mut ranges, 3, 0x2000, 0x200, 16, true);

        unregister_gpu_range(&mut ranges, 1);
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0].id, 2);
        assert_eq!(ranges[1].id, 3);

        unregister_gpu_range(&mut ranges, 2);
        assert_eq!(
            near_heaps_in(&ranges, 0x2040),
            "; sampler heap 0x2000+0x200: inside, slot 4"
        );
    }

    #[test]
    fn near_heaps_formats_below_inside_and_past_end() {
        let mut ranges = Vec::new();
        register_gpu_range(&mut ranges, 1, 0x1000, 0x100, 32, false);

        assert_eq!(
            near_heaps_in(&ranges, 0x0f80),
            "; view heap 0x1000+0x100: 0x80 below"
        );
        assert_eq!(
            near_heaps_in(&ranges, 0x1040),
            "; view heap 0x1000+0x100: inside, slot 2"
        );
        assert_eq!(
            near_heaps_in(&ranges, 0x1140),
            "; view heap 0x1000+0x100: 0x40 past its end (slot 10)"
        );
    }
}
