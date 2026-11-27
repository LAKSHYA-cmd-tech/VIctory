//! SSE4a's `extrq` and `insertq` with their field in the instruction, as SSE2.
//!
//! Intel CPUs have no SSE4a, so each one faults and `crash` emulates it, about a microsecond
//! an exception, and titles run them in hot loops: hundreds of exceptions a second at a single
//! place. The first fault at a place replaces the instruction's first five bytes by a jump to
//! the same work in SSE2, in memory within reach of the title's code; the bytes left after the
//! jump are never run. The forms with their field in a register, and places where the jump
//! would cross 16 aligned bytes, stay emulated.

use std::sync::Mutex;

/// `extrq xmm, len, idx` (`66 [rex] 0f 78 /0 ib ib`, `dst` the r/m register) or
/// `insertq dst, src, len, idx` (`f2 [rex] 0f 78 /r ib ib`).
#[derive(Clone, Copy, Debug)]
struct Field {
    insert: bool,
    dst: u8,
    src: u8,
    /// Bits in the field, 1 to 64, and where it starts.
    len: u32,
    idx: u32,
    /// Bytes in the instruction.
    size: usize,
}

fn decode(code: &[u8]) -> Option<Field> {
    let prefix = *code.first()?;
    let (rex, at) = match *code.get(1)? {
        r @ 0x40..=0x4f => (r, 2),
        _ => (0, 1),
    };
    if !matches!(prefix, 0x66 | 0xf2) || code.get(at..at + 2)? != [0x0f, 0x78] {
        return None;
    }
    let &[modrm, len, idx] = code.get(at + 2..at + 5)? else {
        return None;
    };
    if modrm >> 6 != 3 {
        return None;
    }
    let reg = (modrm >> 3) & 7 | (rex & 4) << 1;
    let rm = modrm & 7 | (rex & 1) << 3;
    let insert = prefix == 0xf2;
    if !insert && reg != 0 {
        return None;
    }
    let len = match u32::from(len & 63) {
        0 => 64,
        n => n,
    };
    let (dst, src) = if insert { (reg, rm) } else { (rm, rm) };
    Some(Field { insert, dst, src, len, idx: u32::from(idx & 63), size: at + 5 })
}

/// `jmp rel32` at `at` to `to`, if it reaches.
fn jump(at: usize, to: usize) -> Option<[u8; 5]> {
    let rel = i32::try_from(to as i64 - (at as i64 + 5)).ok()?;
    let [a, b, c, d] = rel.to_le_bytes();
    Some([0xe9, a, b, c, d])
}

/// The instruction `f` at `at` as SSE2 code placed at `stub`, entered 16 bytes in (the first
/// 16 are the bits `insertq` keeps, for its `pand`), then a jump back past it. Only the
/// destination changes, its high quadword zeroed as the emulation does; `insertq` borrows a
/// third register, saved on the stack and put back.
fn stub(f: Field, at: usize, place: usize) -> Option<Vec<u8>> {
    // `prefix [rex] 0f code` with registers `reg` and `rm`.
    fn op(prefix: u8, code: u8, reg: u8, rm: u8) -> Vec<u8> {
        let rex = (reg & 8) >> 1 | (rm & 8) >> 3;
        let mut v = vec![prefix];
        if rex != 0 {
            v.push(0x40 | rex);
        }
        v.extend([0x0f, code, 0xc0 | (reg & 7) << 3 | rm & 7]);
        v
    }
    // psrlq (`down`) or psllq of `x` by `by`, nothing for 0.
    fn shift(down: bool, x: u8, by: u32) -> Vec<u8> {
        if by == 0 {
            return Vec::new();
        }
        let mut v = op(0x66, 0x73, if down { 2 } else { 6 }, x);
        v.push(by as u8);
        v
    }
    let mut code = vec![0; 16];
    let x = f.dst;
    if !f.insert {
        code.extend(shift(true, x, f.idx));
        code.extend(shift(false, x, 64 - f.len));
        code.extend(shift(true, x, 64 - f.len));
        code.extend(op(0xf3, 0x7e, x, x)); // movq x, x
    } else {
        let ones = if f.len == 64 { u64::MAX } else { (1 << f.len) - 1 };
        code[..8].copy_from_slice(&(!(ones << f.idx)).to_le_bytes());
        let t = (0..3).find(|&t| t != f.dst && t != f.src)?;
        code.extend([0x48, 0x8d, 0x64, 0x24, 0xf0]); // lea rsp, [rsp - 16]
        code.extend([0xf3, 0x0f, 0x7f, t << 3 | 4, 0x24]); // movdqu [rsp], t
        code.extend(op(0xf3, 0x7e, t, f.src)); // movq t, src
        code.extend(shift(false, t, 64 - f.len));
        code.extend(shift(true, t, 64 - f.len));
        code.extend(shift(false, t, f.idx));
        // pand dst, [the kept bits]
        code.extend(if x >= 8 { &[0x66, 0x44][..] } else { &[0x66] });
        let disp = -i32::try_from(code.len() + 7).ok()?;
        code.extend([0x0f, 0xdb, (x & 7) << 3 | 5]);
        code.extend(disp.to_le_bytes());
        code.extend(op(0x66, 0xeb, x, t)); // por dst, t
        code.extend([0xf3, 0x0f, 0x6f, t << 3 | 4, 0x24]); // movdqu t, [rsp]
        code.extend([0x48, 0x8d, 0x64, 0x24, 0x10]); // lea rsp, [rsp + 16]
    }
    let end = place + code.len();
    code.extend(jump(end, at + f.size)?);
    Some(code)
}

const STUBS_LEN: usize = 64 << 10;

/// The memory new stubs go in, near the title's code, and the bytes used. Its lock is held
/// while code is changed: one thread at a time.
static STUBS: Mutex<Option<(usize, usize)>> = Mutex::new(None);

/// Memory for code within reach of a `jmp rel32` at `at`.
fn near(at: usize) -> Option<usize> {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc,
    };
    let at = at & !0xffff;
    let mut places = (0x10_0000..0x7000_0000)
        .step_by(0x10_0000)
        .flat_map(|d| [at.checked_sub(d), at.checked_add(d)])
        .flatten();
    places.find_map(|place| {
        let rights = PAGE_EXECUTE_READWRITE;
        // SAFETY: reserves free memory at that address or fails.
        let got = unsafe {
            VirtualAlloc(place as *const _, STUBS_LEN, MEM_COMMIT | MEM_RESERVE, rights)
        };
        (!got.is_null()).then_some(got as usize)
    })
}

/// After the instruction at `at` faulted and was emulated: replaces it by a jump to its
/// [`stub`], if it is an `extrq` or `insertq` with its field in the instruction and the jump
/// fits in its aligned 16 bytes.
///
/// # Safety
/// `at` is in code that stays mapped, at an instruction that just faulted.
pub(crate) unsafe fn replace(at: usize) {
    if (at & 15) + 5 > 16 || !std::is_x86_feature_detected!("cmpxchg16b") {
        return;
    }
    let n = (0x1000 - (at & 0xfff)).min(7);
    // SAFETY: the instruction's bytes, which the CPU just fetched, none past its page.
    let code = unsafe { std::slice::from_raw_parts(at as *const u8, n) };
    if decode(code).is_none() {
        return;
    }
    let mut stubs = STUBS.lock().unwrap_or_else(|e| e.into_inner());
    // Again under the lock: a second thread's fault at the same place finds the jump.
    // SAFETY: as above.
    let code = unsafe { std::slice::from_raw_parts(at as *const u8, n) };
    let Some(f) = decode(code) else {
        return;
    };
    let fits = |&(place, used): &(usize, usize)| {
        let body = stub(f, at, place + used)?;
        let jump = jump(at, place + used + 16)?;
        (used + body.len() <= STUBS_LEN).then_some((body, jump))
    };
    // A full block, or one out of reach of this code, is kept for the stubs in it and a new
    // one made.
    if stubs.as_ref().and_then(fits).is_none() {
        *stubs = near(at).map(|place| (place, 0));
    }
    let Some(((body, jump), (place, used))) = stubs.as_mut().and_then(|s| Some((fits(&*s)?, s)))
    else {
        return;
    };
    let at_stub = *place + *used;
    // SAFETY: memory made for the stubs, past those in use, which nothing runs yet.
    unsafe { std::ptr::copy_nonoverlapping(body.as_ptr(), at_stub as *mut u8, body.len()) };
    *used += body.len().next_multiple_of(16);
    let mut was = [0; 5];
    was.copy_from_slice(&code[..5]);
    // SAFETY: as for this function; the stub does what the instruction does.
    unsafe { patch(at, was, jump) };
}

/// Replaces the 5 bytes at `at`, which are `was`, by `jump` in one 16-byte store, so that a
/// thread about to run them sees either all old or all new.
///
/// # Safety
/// `at` is in code that stays mapped, the 5 bytes inside its aligned 16, and the caller holds
/// [`STUBS`]'s lock.
unsafe fn patch(at: usize, was: [u8; 5], jump: [u8; 5]) {
    use std::arch::x86_64::cmpxchg16b;
    use std::sync::atomic::Ordering::SeqCst;
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let word = (at & !15) as *mut u128;
    let base = word as *const core::ffi::c_void;
    let here = (at & 15)..(at & 15) + 5;
    let mut old = 0;
    // SAFETY: the aligned 16 bytes lie in one page of mapped code, made writable for the store
    // and put back; the CPU has cmpxchg16b (the caller asked).
    unsafe {
        if VirtualProtect(base, 16, PAGE_EXECUTE_READWRITE, &mut old) == 0 {
            return;
        }
        // What is there (zeros are replaced by zeros).
        let mut now = cmpxchg16b(word, 0, 0, SeqCst, SeqCst);
        loop {
            let mut bytes = now.to_le_bytes();
            if bytes[here.clone()] != was {
                break;
            }
            bytes[here.clone()].copy_from_slice(&jump);
            let there = cmpxchg16b(word, now, u128::from_le_bytes(bytes), SeqCst, SeqCst);
            if there == now {
                break;
            }
            now = there;
        }
        VirtualProtect(base, 16, old, &mut old);
        FlushInstructionCache(GetCurrentProcess(), base, 16);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc, VirtualFree,
    };

    fn mask(len: u32) -> u64 {
        if len == 64 { u64::MAX } else { (1 << len) - 1 }
    }

    /// Each case run twice: the first pass faults on a CPU without SSE4a, is emulated and
    /// replaced, the second runs the SSE2. Both leave the destination as the instruction
    /// does, its high quadword zeroed, and every other register as it was.
    #[test]
    fn an_extrq_or_insertq_becomes_sse2() {
        crate::crash::report_crashes();
        let (d_in, s_in) = (0x0123_4567_89ab_cdefu64, 0xfedc_ba98_7654_3210u64);
        // The bytes, where in its aligned 16 the instruction lies, and whether a jump fits.
        let cases: [(&[u8], usize, bool); 6] = [
            (&[0xf2, 0x0f, 0x78, 0xc1, 0x10, 0x10], 1, true),
            (&[0x66, 0x0f, 0x78, 0xc2, 0x08, 0x14], 7, true),
            (&[0xf2, 0x45, 0x0f, 0x78, 0xc8, 0x0c, 0x28], 11, true),
            (&[0x66, 0x41, 0x0f, 0x78, 0xc1, 0x00, 0x04], 3, true),
            (&[0xf2, 0x0f, 0x78, 0xc9, 0x05, 0x03], 0, true),
            (&[0xf2, 0x0f, 0x78, 0xc1, 0x10, 0x10], 12, false),
        ];
        let replaced = !std::is_x86_feature_detected!("sse4a");
        for (insn, offset, fits) in cases {
            let f = decode(insn).unwrap();
            let (dst, src) = (f.dst, f.src);
            let d_in = if src == dst { s_in } else { d_in };
            let m = mask(f.len);
            let expected = if f.insert {
                d_in & !(m << f.idx) | (s_in & m) << f.idx
            } else {
                (d_in >> f.idx) & m
            };
            // fn(destination, source, out): xmm8 and xmm9 kept for the caller, xmm0-2 all
            // ones, both values in both halves, the instruction, then the destination, the
            // source and xmm0-2 stored.
            let (r, low) = (|x: u8| (x & 8) >> 1, |x: u8| (x & 7) << 3);
            let movq = |x: u8, gpr: u8| [0x66, 0x48 | r(x), 0x0f, 0x6e, 0xc0 | low(x) | gpr];
            let both = |x: u8| [0x66, 0x40 | r(x) | r(x) >> 2, 0x0f, 0x6c, 0xc0 | ((x & 7) * 9)];
            let store = |x: u8, at: u8| [0xf3, 0x41 | r(x), 0x0f, 0x7f, 0x40 | low(x), at];
            let mut code = vec![0x48, 0x8d, 0x64, 0x24, 0xd8];
            code.extend([0xf3, 0x44, 0x0f, 0x7f, 0x04, 0x24]);
            code.extend([0xf3, 0x44, 0x0f, 0x7f, 0x4c, 0x24, 0x10]);
            code.extend([0x66, 0x0f, 0x76, 0xc0, 0x66, 0x0f, 0x76, 0xc9, 0x66, 0x0f, 0x76, 0xd2]);
            code.extend(movq(dst, 1));
            code.extend(both(dst));
            if src != dst {
                code.extend(movq(src, 2));
                code.extend(both(src));
            }
            while code.len() % 16 != offset {
                code.push(0x90);
            }
            let here = code.len();
            code.extend(insn);
            code.extend(store(dst, 0));
            code.extend(store(src, 16));
            for x in 0..3 {
                code.extend(store(x, 32 + 16 * x));
            }
            code.extend([0xf3, 0x44, 0x0f, 0x6f, 0x04, 0x24]);
            code.extend([0xf3, 0x44, 0x0f, 0x6f, 0x4c, 0x24, 0x10]);
            code.extend([0x48, 0x8d, 0x64, 0x24, 0x28, 0xc3]);
            // SAFETY: memory of our own, written, then run as the function it holds.
            unsafe {
                let rights = PAGE_EXECUTE_READWRITE;
                let page = VirtualAlloc(std::ptr::null(), 4096, MEM_COMMIT | MEM_RESERVE, rights);
                assert!(!page.is_null());
                let page = page as usize;
                std::ptr::copy_nonoverlapping(code.as_ptr(), page as *mut u8, code.len());
                let run: extern "system" fn(u64, u64, *mut [u64; 10]) = std::mem::transmute(page);
                for pass in 0..2 {
                    let mut out = [0u64; 10];
                    run(d_in, s_in, &mut out);
                    assert_eq!(out[..2], [expected, 0], "{f:?}, pass {pass}");
                    if src != dst {
                        assert_eq!(out[2..4], [s_in, s_in], "source kept: {f:?}, pass {pass}");
                    }
                    for x in 0..3u8 {
                        if x != dst && x != src {
                            let at = 4 + 2 * x as usize;
                            assert_eq!(out[at..at + 2], [u64::MAX; 2], "xmm{x} kept: {f:?}");
                        }
                    }
                }
                let now = std::slice::from_raw_parts((page + here) as *const u8, insn.len());
                if replaced && fits {
                    assert_eq!(now[0], 0xe9, "a jump: {f:?}");
                } else {
                    assert_eq!(now, insn, "left alone: {f:?}");
                }
                VirtualFree(page as *mut _, 0, MEM_RELEASE);
            }
        }
        assert!(decode(&[0xf2, 0x0f, 0x79, 0xc1]).is_none(), "the field in a register");
        assert!(decode(&[0x66, 0x0f, 0x78, 0xca, 0x08, 0x14]).is_none(), "extrq with reg not 0");
    }
}
