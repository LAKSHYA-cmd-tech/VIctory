//! End-to-end tests of synthetic GFX10 VALU shaders lowered to DXIL, validated
//! by `dxil.dll` when present, and executed on the host GPU and WARP.
//!
//! Covers RDNA2 hardware semantics that differ from naive DXIL lowering:
//! `v_med3_f32`/`v_med3_f16` with NaN operands, `v_fract_f32`/`v_fract_f16`
//! near `1.0`, turn-range reduction for `v_sin_*`/`v_cos_*`, and the
//! layouts `tbuffer_store_format_*` writes.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SOP2 `op sdst, src0, src1`.
const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000
        | (op as u32 & 0x7F) << 23
        | (sdst as u32 & 0x7F) << 16
        | (src1 as u32) << 8
        | src0 as u32
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], 0` (soffset = null).
const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// VOP1 `op vdst, src0` (`src0`: 0..105 = SGPR, 256..511 = VGPR).
const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3 `op vdst, src0, src1, src2` (no modifiers).
const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// MUBUF `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

const fn sgpr(n: u8) -> u16 {
    n as u16
}

const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

/// Builds and lowers a 1-thread synthetic RDNA2 compute shader that receives
/// 4 dwords of root constants in `s0..s3`, runs `body` (leaving its result in
/// `v0`), and stores `v0` to dword 0 of the raw UAV at heap index 0.
fn build_valu_cs(body: &[u32]) -> HostShader {
    try_build_valu_cs(body).expect("lower")
}

/// [`build_valu_cs`], with the lowering's refusal as an error.
fn try_build_valu_cs(body: &[u32]) -> Result<HostShader, String> {
    let mut words = Vec::new();
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push(s_mov_b32(5, 0x80)); // inline constant 0
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    // Copy root constants s0..s3 into v1..v4 so VALU instructions operate on
    // VGPRs just as they do in real shaders.
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, sgpr(i))); // v_mov_b32
    }
    words.extend_from_slice(body);
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.push(S_ENDPGM);
    try_lower_cs(&words)
}

/// Lowers `words` as a 1-thread compute program with [`build_valu_cs`]'s
/// layout: 4 root constants in `s0..s3`, the address of a table holding a
/// UAV in `s4`.
fn lower_cs(words: &[u32]) -> HostShader {
    try_lower_cs(words).expect("lower")
}

fn try_lower_cs(words: &[u32]) -> Result<HostShader, String> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [1, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let res = resource::track(&p, &native.layout, native.stage).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let st = structure::structure(&p).expect("structure");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        options: super::TranslateOptions::default(),
    })
    .map_err(|e| format!("{e:?}"))?;

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        address_srvs: lowered.address_srvs,
        stride_table: lowered.stride_table,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    Ok(shader)
}

// One test at a time: with these tests on the hardware adapter and WARP
// from several test threads at once, now and then none of their
// dispatches ever completes and the test binary hangs.
pub(crate) static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Runs `shader` on every available SM 6.6 host (hardware GPU and WARP) for
/// each 4-dword input set and invokes `check(backend, case_index, inputs, result_dword)`.
fn run_cases(
    shader: &HostShader,
    cases: &[[u32; 4]],
    mut check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for (i, &inputs) in cases.iter().enumerate() {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    (root, _) => super::host::guard(root, 1, 0)
                        .unwrap_or_else(|| panic!("unexpected user data {:?}", (root, u.slot))),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(1)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            check(label, i, inputs, out[0][0]);
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

fn is_f16_nan(bits: u16) -> bool {
    (bits & 0x7FFF) > 0x7C00
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;
    let f32_bits = match exp {
        0 => {
            if mant == 0 {
                sign
            } else {
                let mut m = u32::from(mant);
                let mut e = 127 - 14;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                sign | (e << 23) | ((m & 0x3FF) << 13)
            }
        }
        0x1F => sign | (0xFF << 23) | (u32::from(mant) << 13),
        _ => sign | ((u32::from(exp) + 112) << 23) | (u32::from(mant) << 13),
    };
    f32::from_bits(f32_bits)
}

#[test]
fn packed_add_selects_halves_of_inline_constants() {
    // Add the inline constant 1/(2*pi) to v1 (zero) to read back its two halves.
    // The same GFX10 source field (248) has a different bit pattern for a
    // packed floating-point and a packed integer instruction.
    let cases = [
        // v_pk_add_f16 v0, 0x3118, v1
        ([0xcc0f4000, 0x1a0202f8], 0x0000_3118),
        // v_pk_add_f16 v0, 0x3118, v1 op_sel_hi:[0,1] (use low twice)
        ([0xcc0f4000, 0x120202f8], 0x3118_3118),
        // v_pk_add_u16 v0, 0x3e22f983, v1
        ([0xcc0a4000, 0x1a0202f8], 0x3e22_f983),
    ];
    for (words, expected) in cases {
        let shader = build_valu_cs(&words);
        run_cases(&shader, &[[0; 4]], |backend, _, _, got| {
            assert_eq!(got, expected, "{backend}: {words:08x?}");
        });
    }
}

#[test]
fn v_med3_f32_matches_rdna2_with_nan() {
    // v_med3_f32 v0, v1, v2, v3 (VOP3 opcode 0x157)
    let shader = build_valu_cs(&vop3(0x157, 0, vgpr(1), vgpr(2), vgpr(3)));
    let nan = f32::NAN.to_bits();
    let f = f32::to_bits;

    let cases: &[([u32; 4], Option<f32>)] = &[
        // Normal medians (all orderings).
        ([f(1.0), f(2.0), f(3.0), 0], Some(2.0)),
        ([f(3.0), f(1.0), f(2.0), 0], Some(2.0)),
        ([f(-5.0), f(0.0), f(5.0), 0], Some(0.0)),
        ([f(1.0), f(1.0), f(2.0), 0], Some(1.0)),
        // One NaN: RDNA2 returns min3 of the non-NaN operands.
        ([f(1.0), f(2.0), nan, 0], Some(1.0)),
        ([nan, f(1.0), f(0.0), 0], Some(0.0)),
        ([nan, f(0.0), f(1.0), 0], Some(0.0)),
        ([f(2.0), nan, f(1.0), 0], Some(1.0)),
        ([f(-3.0), nan, f(4.0), 0], Some(-3.0)),
        // Two NaNs: returns the sole non-NaN operand.
        ([nan, f(5.0), nan, 0], Some(5.0)),
        ([nan, nan, f(-2.0), 0], Some(-2.0)),
        ([f(7.0), nan, nan, 0], Some(7.0)),
        // Three NaNs: returns NaN.
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert_eq!(
                got_f,
                want,
                "{backend}: v_med3_f32({:?}, {:?}, {:?}) = {got_f} ({got:#010x}), want {want}",
                f32::from_bits(inp[0]),
                f32::from_bits(inp[1]),
                f32::from_bits(inp[2]),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_med3_f32(NaN, NaN, NaN) = {got_f} ({got:#010x}), want NaN",
            ),
        }
    });
}

/// RDNA2 orders -0 below +0 in its maximums and minimums; DXIL's FMax and
/// FMin compare and select, which can give either zero.
#[test]
fn max_and_min_order_negative_zero_below_positive_zero() {
    let (pz, nz) = (0_u32, 0x8000_0000_u32);
    // Each case has a -0 and a +0 among its first two operands.
    let cases = [
        [nz, pz, nz, 0],
        [pz, nz, nz, 0],
        [nz, pz, pz, 0],
        [pz, nz, pz, 0],
    ];
    let ops = [
        ("v_max_f32", 0x110, pz),
        ("v_min_f32", 0x10F, nz),
        ("v_max3_f32", 0x154, pz),
        ("v_min3_f32", 0x151, nz),
    ];
    for (name, op, want) in ops {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &cases, |backend, _, inp, got| {
            assert_eq!(got, want, "{backend}: {name} {inp:x?}");
        });
    }
}

#[test]
fn v_med3_f16_matches_rdna2_with_nan() {
    // v_med3_f16 v0, v1, v2, v3 (VOP3 opcode 0x357)
    let shader = build_valu_cs(&vop3(0x357, 0, vgpr(1), vgpr(2), vgpr(3)));
    let h0 = 0x0000u32; // 0.0
    let h1 = 0x3C00u32; // 1.0
    let h2 = 0x4000u32; // 2.0
    let h3 = 0x4200u32; // 3.0
    let h5 = 0x4500u32; // 5.0
    let hn2 = 0xC000u32; // -2.0
    let nan = 0x7E00u32;

    let cases: &[([u32; 4], Option<u16>)] = &[
        ([h1, h2, h3, 0], Some(h2 as u16)),
        ([hn2, h0, h3, 0], Some(h0 as u16)),
        ([h1, h2, nan, 0], Some(h1 as u16)),
        ([nan, h1, h0, 0], Some(h0 as u16)),
        ([nan, h0, h1, 0], Some(h0 as u16)),
        ([h2, nan, h1, 0], Some(h1 as u16)),
        ([nan, h5, nan, 0], Some(h5 as u16)),
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_med3_f16({:#06x}, {:#06x}, {:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0], inp[1], inp[2],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_med3_f16(NaN, NaN, NaN) = {got_h:#06x}, want NaN",
            ),
        }
    });
}

/// `v_cmp_lg_*` is "less or greater": ordered and not equal, so false when
/// either operand is a NaN (RDNA2 ISA; the same on gfx1036 hardware).
#[test]
fn v_cmp_lg_is_false_for_nan() {
    // v_cmp_lg_{f32,f16}_e64 s20, v1, v2 (VOPC 0x005 / 0x0cd in VOP3);
    // v_cndmask_b32_e64 v0, 0, 1, s20 (VOP3 0x101)
    let nan = f32::NAN.to_bits();
    let (inf, ninf) = (f32::INFINITY.to_bits(), f32::NEG_INFINITY.to_bits());
    let f = f32::to_bits;
    let f32_cases: &[([u32; 4], u32)] = &[
        ([f(1.0), f(2.0), 0, 0], 1),
        ([f(2.0), f(1.0), 0, 0], 1),
        ([f(1.0), f(1.0), 0, 0], 0),
        ([f(0.0), f(-0.0), 0, 0], 0),
        ([inf, ninf, 0, 0], 1),
        ([inf, inf, 0, 0], 0),
        ([nan, f(1.0), 0, 0], 0),
        ([f(1.0), nan, 0, 0], 0),
        ([nan, nan, 0, 0], 0),
        ([0x7F80_0001, f(0.0), 0, 0], 0), // a signalling NaN
    ];
    let f16_cases: &[([u32; 4], u32)] = &[
        ([0x3C00, 0x4000, 0, 0], 1),
        ([0x4000, 0x3C00, 0, 0], 1),
        ([0x3C00, 0x3C00, 0, 0], 0),
        ([0x0000, 0x8000, 0, 0], 0),
        ([0x7C00, 0xFC00, 0, 0], 1),
        ([0x7E00, 0x3C00, 0, 0], 0),
        ([0x3C00, 0x7E00, 0, 0], 0),
        ([0x7E00, 0x7E00, 0, 0], 0),
        ([0x7C01, 0x0000, 0, 0], 0),           // a signalling NaN
        ([0xABCD_3C00, 0x1234_7E00, 0, 0], 0), // only the low halves are read
    ];
    for (op, what, cases) in [
        (0x005u16, "v_cmp_lg_f32", f32_cases),
        (0x0cd, "v_cmp_lg_f16", f16_cases),
    ] {
        let mut body = vop3(op, 20, vgpr(1), vgpr(2), 0x80).to_vec();
        body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(20)));
        let shader = build_valu_cs(&body);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(&shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(
                got, want,
                "{backend}: {what}({:#010x}, {:#010x}) = {got}, want {want}",
                inp[0], inp[1],
            );
        });
    }
}

#[test]
fn v_fract_f32_never_returns_one() {
    // v_fract_f32 v0, v1 (VOP1 opcode 0x020)
    let shader = build_valu_cs(&[vop1(0x020, 0, vgpr(1))]);
    let f = f32::to_bits;
    let below_one = 0x3F7F_FFFFu32;

    let cases: &[(u32, Option<u32>)] = &[
        // Tiny negatives: must clamp strictly below 1.0 (not 0x3f800000).
        (f(-1e-10), Some(below_one)),
        (f(-1e-8), Some(below_one)),
        (f(-2.0f32.powi(-25)), Some(below_one)),
        (f(-2.0f32.powi(-24)), Some(below_one)),
        // Normal values.
        (f(0.0), Some(f(0.0))),
        (f(-0.0), Some(f(0.0))),
        (f(0.25), Some(f(0.25))),
        (f(3.75), Some(f(0.75))),
        (f(-0.25), Some(f(0.75))),
        (f(-3.25), Some(f(0.75))),
        (f(1.0), Some(f(0.0))),
        (f(-1.0), Some(f(0.0))),
        // Non-finite inputs yield NaN.
        (f(f32::NAN), None),
        (f(f32::INFINITY), None),
        (f(f32::NEG_INFINITY), None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        match expect {
            Some(want) => assert_eq!(
                got,
                want,
                "{backend}: v_fract_f32({:?} / {:#010x}) = {got:#010x} ({:?}), want {want:#010x}",
                f32::from_bits(inp[0]),
                inp[0],
                f32::from_bits(got),
            ),
            None => assert!(
                f32::from_bits(got).is_nan(),
                "{backend}: v_fract_f32({:?}) = {got:#010x}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });
}

#[test]
fn v_fract_f16_never_returns_one() {
    // v_fract_f16 v0, v1 (VOP1 opcode 0x05f)
    let shader = build_valu_cs(&[vop1(0x05f, 0, vgpr(1))]);
    let below_one = 0x3BFFu16; // 1.0 - 2^-11

    assert_eq!(f16_to_f32(0x9400), -2.0f32.powi(-10));
    assert_eq!(f16_to_f32(0x9000), -2.0f32.powi(-11));
    assert_eq!(f16_to_f32(0x8C00), -2.0f32.powi(-12));

    let cases: &[(u16, Option<u16>)] = &[
        // Negatives in (-2^-11, 0): must clamp to 0x3BFF, never 0x3C00 (1.0).
        (0x8C00, Some(below_one)), // -2^-12
        (0x8400, Some(below_one)), // -2^-14
        (0x8001, Some(below_one)), // -2^-24 (smallest negative f16 subnormal)
        (0x9000, Some(below_one)), // -2^-11 (exact 1 - 2^-11 = 0x3BFF)
        (0x9400, Some(0x3BFE)),    // -2^-10 (exact 1 - 2^-10 = 0x3BFE)
        // Normal values.
        (0x0000, Some(0x0000)), // 0.0 -> 0.0
        (0x3400, Some(0x3400)), // 0.25 -> 0.25
        (0xB400, Some(0x3A00)), // -0.25 -> 0.75
        (0x4300, Some(0x3800)), // 3.5 -> 0.5
        (0x3C00, Some(0x0000)), // 1.0 -> 0.0
        // Non-finite inputs yield NaN.
        (0x7E00, None), // NaN
        (0x7C00, None), // +inf
        (0xFC00, None), // -inf
    ];

    let inputs: Vec<[u32; 4]> = cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

/// RDNA2's flt16_to_snorm16 / flt16_to_unorm16: clamp, scale by 32767 or
/// 65535, round the exact product to nearest even; NaN gives 0.
fn cvt_norm16_reference(bits: u16, signed: bool) -> u16 {
    let x = f16_to_f32(bits);
    if x.is_nan() {
        return 0;
    }
    let (lo, scale) = if signed {
        (-1.0, 32767.0)
    } else {
        (0.0, 65535.0)
    };
    // f64 holds the product of an f16 and a 16-bit scale exactly.
    let p = f64::from(x.clamp(lo, 1.0)) * scale;
    p.round_ties_even() as i32 as u16
}

#[test]
fn v_cvt_norm_i16_and_u16_f16_round_the_exact_product() {
    // Every 251st f16 pattern, plus the values where an f32 product would
    // round before the rounding to an integer, ties, and the edges.
    let mut bits: Vec<u16> = (0..=u16::MAX).step_by(251).collect();
    bits.extend([
        0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3800, 0xB800, 0x3801, 0xB801, 0x37FF,
        0x3BFF, 0xBBFF, 0x3C00, 0xBC00, 0x3C01, 0x7BFF, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01,
        0x3A9B, 0xBA9B, 0x2E66, 0x1400,
    ]);
    for (op, signed, what) in [
        (0x063u16, true, "v_cvt_norm_i16_f16"),
        (0x064, false, "v_cvt_norm_u16_f16"),
    ] {
        // v_cvt_norm_*16_f16 v0, v1 (VOP1); v0's high half starts at 0.
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = bits.iter().map(|&b| [u32::from(b), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = cvt_norm16_reference(bits[i], signed);
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                bits[i],
            );
        });
    }
}

#[test]
fn v_sin_and_v_cos_f32_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x035, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x036, 0, vgpr(1))]);
    let f = f32::to_bits;
    let half_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;

    let sin_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(0.0)),
        (0.25, Some(1.0)),
        (255.0, Some(0.0)),
        (255.25, Some(1.0)),
        (255.5, Some(0.0)),
        (-255.25, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_sin_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_sin_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    let cos_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(1.0)),
        (0.25, Some(0.0)),
        (255.0, Some(1.0)),
        (255.25, Some(0.0)),
        (255.5, Some(-1.0)),
        (-255.5, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_cos_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_cos_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    // Main's `trig_turns` preserves sin(-0.0) = -0.0 and produces exact +0.0 at
    // half-turns (sin) and quarter-turns (cos).
    let exact_sin: &[(u32, u32)] = &[
        (0x8000_0000, 0x8000_0000), // sin(-0.0) = -0.0
        (f(255.0), 0),              // sin(255.0) = +0.0
        (f(255.5), 0),              // sin(255.5) = +0.0
    ];
    let exact_sin_in: Vec<[u32; 4]> = exact_sin.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&sin_shader, &exact_sin_in, |backend, i, inp, got| {
        let want = exact_sin[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_sin_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
    let exact_cos: &[(u32, u32)] = &[
        (f(0.25), 0),   // cos(0.25) = +0.0
        (f(255.25), 0), // cos(255.25) = +0.0
    ];
    let exact_cos_in: Vec<[u32; 4]> = exact_cos.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&cos_shader, &exact_cos_in, |backend, i, inp, got| {
        let want = exact_cos[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cos_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
}

#[test]
fn v_cvt_u16_and_i16_f16_saturate() {
    // v_mov_b32 v0, 0; v_cvt_u16_f16 / v_cvt_i16_f16 v0, v1 (VOP1 0x052 / 0x053).
    let cases: &[(u16, u16, u16)] = &[
        // (f16 input, u16 result, i16 result)
        (0x0000, 0, 0),          // 0
        (0x8000, 0, 0),          // -0
        (0x3E00, 1, 1),          // 1.5 -> 1 (toward zero)
        (0xBE00, 0, 0xFFFF),     // -1.5 -> 0 / -1
        (0x57D0, 125, 125),      // 125.0
        (0x7800, 32768, 32767),  // 32768 saturates i16
        (0x77FF, 32752, 32752),  // 32752, the largest f16 below 32768
        (0xF800, 0, 0x8000),     // -32768 fits i16
        (0xF801, 0, 0x8000),     // -32800 saturates i16
        (0x7BFF, 65504, 32767),  // max finite f16
        (0xFBFF, 0, 0x8000),     // -65504
        (0x7C00, 0xFFFF, 32767), // +inf
        (0xFC00, 0, 0x8000),     // -inf
        (0x7E00, 0, 0),          // NaN
        (0xFE00, 0, 0),          // -NaN
        (0x0001, 0, 0),          // smallest subnormal
    ];
    for (op, what, pick) in [
        (0x052u16, "v_cvt_u16_f16", 0usize),
        (0x053, "v_cvt_i16_f16", 1),
    ] {
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| [u32::from(c.0), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = if pick == 0 { cases[i].1 } else { cases[i].2 };
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                cases[i].0,
            );
        });
    }
}

#[test]
fn v_sin_and_v_cos_f16_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x060, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x061, 0, vgpr(1))]);

    // Sanity-check the f16 bit encodings of large turns.
    assert_eq!(f16_to_f32(0x5BF8), 255.0);
    assert_eq!(f16_to_f32(0x5BFA), 255.25);
    assert_eq!(f16_to_f32(0x5BFC), 255.5);
    assert_eq!(f16_to_f32(0x7BFF), 65504.0);

    let sin_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(0.0)),  // 255.0
        (0x5BFA, Some(1.0)),  // 255.25
        (0x5BFC, Some(0.0)),  // 255.5
        (0xDBFA, Some(-1.0)), // -255.25
        (0x7BFF, Some(0.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_sin_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_sin_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });

    let cos_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(1.0)),  // 255.0
        (0x5BFA, Some(0.0)),  // 255.25
        (0x5BFC, Some(-1.0)), // 255.5
        (0x7BFF, Some(1.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_cos_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_cos_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

#[test]
fn bfe_signed_and_unsigned_match_rdna2() {
    // v_bfe_i32 v0, v1, v2, v3 (VOP3 opcode 0x149)
    let v_signed = build_valu_cs(&vop3(0x149, 0, vgpr(1), vgpr(2), vgpr(3)));
    // v_bfe_u32 v0, v1, v2, v3 (VOP3 opcode 0x148)
    let v_unsigned = build_valu_cs(&vop3(0x148, 0, vgpr(1), vgpr(2), vgpr(3)));
    // s_bfe_i32 s6, s0, s1 (SOP2 opcode 0x28: s1 = off[4:0] | width[22:16]); v_mov_b32 v0, s6
    let s_signed = build_valu_cs(&[sop2(0x28, 6, 0, 1), vop1(0x001, 0, sgpr(6))]);

    let signed_cases: &[([u32; 4], u32)] = &[
        // Field crosses bit 31: extends from bit 31.
        ([0xd2f9_0159, 28, 16, 0], 0xffff_fffd),
        ([0x8000_0000, 31, 4, 0], 0xffff_ffff),
        ([0x7000_0000, 28, 8, 0], 0x0000_0007),
        // Normal fields inside 32 bits.
        ([0x0000_0080, 4, 4, 0], 0xffff_fff8),
        ([0x0000_0070, 4, 4, 0], 0x0000_0007),
        // Edge widths: 0 yields 0; v_bfe masks width to 5 bits (32 -> 0),
        // while 31 extracts all remaining bits from offset 4.
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0xf800_0000),
    ];
    let v_in: Vec<[u32; 4]> = signed_cases.iter().map(|c| c.0).collect();
    run_cases(&v_signed, &v_in, |backend, i, inp, got| {
        let want = signed_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_i32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
    // v_bfe_i32 width is 5-bit (src2[4:0]), so width=32 wraps to 0.
    run_cases(
        &v_signed,
        &[[0x8000_0000, 4, 32, 0]],
        |backend, _, _, got| {
            assert_eq!(got, 0, "{backend}: v_bfe_i32 width 32 should mask to 0");
        },
    );

    // s_bfe_i32 width is 7-bit (src1[22:16]), so width=32 keeps all 32 bits.
    let mut s_cases = signed_cases.to_vec();
    s_cases.push(([0x8000_0000, 4, 32, 0], 0xf800_0000));
    let s_in: Vec<[u32; 4]> = s_cases
        .iter()
        .map(|&(inp, _)| [inp[0], (inp[1] & 31) | ((inp[2] & 0x7F) << 16), 0, 0])
        .collect();
    run_cases(&s_signed, &s_in, |backend, i, inp, got| {
        let want = s_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: s_bfe_i32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });

    let unsigned_cases: &[([u32; 4], u32)] = &[
        ([0xd2f9_0159, 28, 16, 0], 0x0000_000d),
        ([0x8000_0000, 31, 4, 0], 0x0000_0001),
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0x0800_0000),
    ];
    let u_in: Vec<[u32; 4]> = unsigned_cases.iter().map(|c| c.0).collect();
    run_cases(&v_unsigned, &u_in, |backend, i, inp, got| {
        let want = unsigned_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_u32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
}


#[test]
fn v_cmp_class_f16_matches_rdna2() {
    // v_cmp_class_f16_e64 s6, v1, v2 (VOP3 opcode 0x08f); v_cndmask_b32_e64 v0, 0, 1, s6
    let cmp_lo = build_valu_cs(&[
        vop3(0x08f, 6, vgpr(1), vgpr(2), 0)[0],
        vop3(0x08f, 6, vgpr(1), vgpr(2), 0)[1],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);
    // Same with op_sel:[1,0] (high half of v1) + neg on src0 (bit 29 of w1).
    let [mut w0, mut w1] = vop3(0x08f, 6, vgpr(1), vgpr(2), 0);
    w0 |= 1 << 11; // opsel[0] = 1 (high half)
    w1 |= 1 << 29; // neg[0] = 1
    let cmp_hi_neg = build_valu_cs(&[
        w0,
        w1,
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);

    // All 10 IEEE-754 binary16 classes: bit k corresponds to class k.
    let classes: &[(u32, u32)] = &[
        (0x7C01, 0), // sNaN (+sNaN, quiet bit 0x0200 clear)
        (0xFDFF, 0), // -sNaN (sign ignored for NaNs)
        (0x7E00, 1), // +qNaN
        (0xFE01, 1), // -qNaN
        (0xFC00, 2), // -inf
        (0xBC00, 3), // -normal (-1.0)
        (0x8001, 4), // -denormal
        (0x8000, 5), // -0.0
        (0x0000, 6), // +0.0
        (0x0001, 7), // +denormal
        (0x3C00, 8), // +normal (+1.0)
        (0x7C00, 9), // +inf
    ];
    let mut cases = Vec::new();
    for &(bits, cls) in classes {
        // Exact class bit -> true; complement of class bit across all 10 bits -> false.
        cases.push(([bits, 1u32 << cls, 0, 0], 1u32));
        cases.push(([bits, 0x3FF & !(1u32 << cls), 0, 0], 0u32));
    }
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&cmp_lo, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f16({:#06x}, {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });

    // High half with neg: high half 0x3C00 (+1.0) negated -> -normal (class 3);
    // high half 0x7C01 (+sNaN) negated -> still sNaN (class 0, not quieted).
    let hi_cases: &[([u32; 4], u32)] = &[
        ([0x3C00_0000, 1 << 3, 0, 0], 1),
        ([0x3C00_0000, 1 << 8, 0, 0], 0),
        ([0x7C01_FFFF, 1 << 0, 0, 0], 1),
        ([0x7C01_FFFF, 1 << 1, 0, 0], 0),
    ];
    let hi_in: Vec<[u32; 4]> = hi_cases.iter().map(|c| c.0).collect();
    run_cases(&cmp_hi_neg, &hi_in, |backend, i, inp, got| {
        let want = hi_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f16(-hi({:#010x}), {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });
}

/// MTBUF `op vdata, v5, s[8:11], 0 format:fmt offset:16` (no offen/idxen;
/// `op` 4..7 stores x..xyzw, 12..15 their d16 forms).
const fn mtbuf_store(op: u8, fmt: u8, vdata: u8) -> [u32; 2] {
    [
        0x3A << 26 | (fmt as u32) << 19 | (op as u32 & 7) << 16 | 16,
        0x80 << 24 | (op as u32 >> 3 & 1) << 21 | (8 >> 2) << 16 | (vdata as u32) << 8 | 5,
    ]
}

/// Runs `shader` once with root constants `inputs` on every available
/// SM 6.6 host and returns the first `dwords` of its raw UAV, per device.
fn run_dwords(shader: &HostShader, inputs: [u32; 4], dwords: u32) -> Vec<(&'static str, Vec<u32>)> {
    let mut out = Vec::new();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match (u.root, u.slot) {
                (RootType::Constants, 0) => inputs[u.dword as usize],
                (RootType::Table, 1) => 0,
                (root, _) => super::host::guard(root, 1, 0)
                    .unwrap_or_else(|| panic!("unexpected user data {:?}", (root, u.slot))),
            };
        }
        let r = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(dwords)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push((label, r[0].clone()));
    }
    assert!(
        !out.is_empty(),
        "neither hardware nor WARP SM 6.6 device available"
    );
    out
}

#[test]
fn tbuffer_store_format_writes_the_formats_layout() {
    let f = |x: f32| x.to_bits();
    // (what, op, format, data in v1..v4, dwords 4..8 afterwards)
    type Case = (&'static str, u8, u8, [u32; 4], [u32; 4]);
    let cases: &[Case] = &[
        (
            "32_32_32_32 float xyzw",
            7,
            77,
            [f(1.5), f(-0.0), 0x7FC0_1234, 0x0000_0001],
            [f(1.5), f(-0.0), 0x7FC0_1234, 0x0000_0001],
        ),
        (
            "32_32_32_32 sint xyzw",
            7,
            76,
            [u32::MAX, 0x8000_0000, 7, 0x1234_5678],
            [u32::MAX, 0x8000_0000, 7, 0x1234_5678],
        ),
        (
            "32_32_32 float xyz",
            6,
            74,
            [f(2.0), f(3.0), f(4.0), 0xDEAD_BEEF],
            [f(2.0), f(3.0), f(4.0), 0],
        ),
        (
            "32_32 uint xy",
            5,
            63,
            [0xAABB_CCDD, 9, 0xDEAD_BEEF, 0xDEAD_BEEF],
            [0xAABB_CCDD, 9, 0, 0],
        ),
        (
            "32 float x",
            4,
            22,
            [f(-7.25), 1, 2, 3],
            [f(-7.25), 0, 0, 0],
        ),
        (
            // d16: the halves are stored as they are.
            "16_16 float d16_xy",
            13,
            29,
            [0xBC00_3555, 0xDEAD_BEEF, 0, 0],
            [0xBC00_3555, 0, 0, 0],
        ),
        (
            "16_16_16_16 float d16_xyzw",
            15,
            71,
            [0x1234_5678, 0x7C00_FC00, 0, 0],
            [0x1234_5678, 0x7C00_FC00, 0, 0],
        ),
    ];
    for &(what, op, fmt, inputs, want) in cases {
        let mut body = vec![vop1(0x001, 5, 0x80)]; // v_mov_b32 v5, 0
        body.extend(mtbuf_store(op, fmt, 1));
        let shader = build_valu_cs(&body);
        for (backend, out) in run_dwords(&shader, inputs, 8) {
            assert_eq!(
                out[4..8],
                want,
                "{backend}: tbuffer_store {what}: wrote {:#010x?}, want {want:#010x?}",
                &out[4..8],
            );
        }
    }
}

#[test]
fn tbuffer_store_format_refuses_what_it_does_not_convert() {
    for (what, op, fmt, reason) in [
        ("8_8_8_8 unorm", 7, 56, "a converted or narrow format"),
        ("16_16 uint", 5, 27, "a converted or narrow format"),
        ("16 float", 4, 13, "an element under a dword"),
        (
            "32_32_32_32 float, xy only",
            5,
            77,
            "some of its format's channels",
        ),
        ("32 float d16_x", 12, 22, "a converted or narrow format"),
        // f32 to f16 waits for nearest-even narrowing (#422).
        ("16_16 float xy", 5, 29, "a converted or narrow format"),
        (
            "16_16_16_16 float xyzw",
            7,
            71,
            "a converted or narrow format",
        ),
    ] {
        let mut body = vec![vop1(0x001, 5, 0x80)];
        body.extend(mtbuf_store(op, fmt, 1));
        match try_build_valu_cs(&body) {
            Ok(_) => panic!("tbuffer_store {what} lowered; it should be refused"),
            Err(e) => assert!(e.contains(reason), "tbuffer_store {what}: refused with {e}"),
        }
    }
}

/// Runs `body` with an address in root constants (`s[2:3]`) and a root SRV
/// bound for it (`t0` in `ROOT_ADDRESS_SPACE`, the root parameter after the
/// user data) holding `0x100 + k` in dword `k`; checks that `v0` ends up
/// holding dword 3.
fn reads_dword_3_through_a_root_constant_pair(body: &[u32]) {
    let shader = build_valu_cs(body);
    assert!(shader.root_srvs.is_empty());
    let [view] = shader.address_srvs[..] else {
        panic!("{:?}", shader.address_srvs)
    };
    assert_eq!((view.lo.dword, view.hi.dword), (2, 3));
    let data: Vec<u32> = (0..8).map(|k| 0x100 + k).collect();
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        // The constants hold an address the host does not use: it binds the
        // buffer itself.
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            if u.root == RootType::Constants {
                ud[u.index as usize] = 0x1_0000 * (u.dword + 1);
            }
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![
                        (0, Root::Constants(ud)),
                        (
                            1,
                            Root::View {
                                srv: true,
                                data: data.clone(),
                            },
                        ),
                    ],
                    views: &[View::RawUav(1)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(out[0][0], data[3], "{label}: byte 12 of the buffer");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// A global load through an address the title passes as two root constants
/// reads the root SRV the host binds at that address.
#[test]
fn global_loads_through_a_root_constant_pair_read_its_root_srv() {
    reads_dword_3_through_a_root_constant_pair(&[
        vop1(0x001, 5, 128 + 8), // v_mov_b32 v5, 8
        0xDC30_8004,
        0x0002_0005, // global_load_dword v0, v5, s[2:3] offset:4
    ]);
}

/// A scalar load through the same address reads it too.
#[test]
fn scalar_loads_through_a_root_constant_pair_read_its_root_srv() {
    reads_dword_3_through_a_root_constant_pair(&[
        0xF400_0181,
        0xFA00_000C,             // s_load_dword s6, s[2:3], 0xc
        vop1(0x001, 0, sgpr(6)), // v_mov_b32 v0, s6
    ]);
}

#[test]
fn v_mov_b32_modifiers_are_bitwise() {
    let vop3_mods = |op: u16, dst: u8, s0: u16, s1: u16, s2: u16, abs: u32, neg: u32| {
        let [mut w0, mut w1] = vop3(op, dst, s0, s1, s2);
        w0 |= (abs & 7) << 8;
        w1 |= (neg & 7) << 29;
        [w0, w1]
    };

    // v_mov_b32_e64 v0, -v1 / |v1| / -|v1| (VOP3 opcode 0x181)
    let mov_neg = build_valu_cs(&vop3_mods(0x181, 0, vgpr(1), 0, 0, 0, 1));
    let mov_abs = build_valu_cs(&vop3_mods(0x181, 0, vgpr(1), 0, 0, 1, 0));
    let mov_neg_abs = build_valu_cs(&vop3_mods(0x181, 0, vgpr(1), 0, 0, 1, 1));
    // v_mov_b32_sdwa v0, -v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
    let mov_sdwa_neg = build_valu_cs(&[vop1(0x001, 0, 0xF9), 1 | (6 << 8) | (6 << 16) | (1 << 20)]);
    // v_mov_b32_e64 v0, -v1 clamp mul:2 (output modifiers still run on floats)
    let [mut mov_co_w0, mut mov_co_w1] = vop3_mods(0x181, 0, vgpr(1), 0, 0, 0, 1);
    mov_co_w0 |= 1 << 15; // clamp
    mov_co_w1 |= 1 << 27; // omod = mul:2
    let mov_neg_clamp_omod = build_valu_cs(&[mov_co_w0, mov_co_w1]);

    // Normals, signed zeros, infinities, subnormals (must not flush to zero),
    // and signalling/quiet NaNs (must not quiet payload or lose sign bit).
    let values: &[u32] = &[
        0x3F80_0000, // +1.0
        0xBF80_0000, // -1.0
        0x0000_0000, // +0.0
        0x8000_0000, // -0.0
        0x7F80_0000, // +inf
        0xFF80_0000, // -inf
        0x0000_0001, // +min denormal
        0x8000_0001, // -min denormal
        0x007F_FFFF, // +max denormal
        0x807F_FFFF, // -max denormal
        0x7F80_0001, // +sNaN
        0xFF80_0001, // -sNaN
        0x7FC0_1234, // +qNaN with payload
        0xFFC0_1234, // -qNaN with payload
    ];
    let mov_in: Vec<[u32; 4]> = values.iter().map(|&x| [x, 0, 0, 0]).collect();

    run_cases(&mov_neg, &mov_in, |backend, _, inp, got| {
        let want = inp[0] ^ 0x8000_0000;
        assert_eq!(
            got, want,
            "{backend}: v_mov_b32_e64(-{:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
    run_cases(&mov_abs, &mov_in, |backend, _, inp, got| {
        let want = inp[0] & 0x7FFF_FFFF;
        assert_eq!(
            got, want,
            "{backend}: v_mov_b32_e64(|{:#010x}|) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
    run_cases(&mov_neg_abs, &mov_in, |backend, _, inp, got| {
        let want = inp[0] | 0x8000_0000;
        assert_eq!(
            got, want,
            "{backend}: v_mov_b32_e64(-|{:#010x}|) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
    run_cases(&mov_sdwa_neg, &mov_in, |backend, _, inp, got| {
        let want = inp[0] ^ 0x8000_0000;
        assert_eq!(
            got, want,
            "{backend}: v_mov_b32_sdwa(-{:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
    let clamp_omod_cases: &[([u32; 4], u32)] = &[
        ([(-0.25f32).to_bits(), 0, 0, 0], 0.5f32.to_bits()),
        ([(-1.0f32).to_bits(), 0, 0, 0], 1.0f32.to_bits()),
        ([0.25f32.to_bits(), 0, 0, 0], 0.0f32.to_bits()),
    ];
    let co_in: Vec<[u32; 4]> = clamp_omod_cases.iter().map(|c| c.0).collect();
    run_cases(&mov_neg_clamp_omod, &co_in, |backend, i, inp, got| {
        let want = clamp_omod_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_mov_b32_e64(-{:#010x}, clamp, mul:2) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
}

#[test]
fn v_cndmask_b32_modifiers_are_bitwise() {
    let vop3_mods = |op: u16, dst: u8, s0: u16, s1: u16, s2: u16, abs: u32, neg: u32| {
        let [mut w0, mut w1] = vop3(op, dst, s0, s1, s2);
        w0 |= (abs & 7) << 8;
        w1 |= (neg & 7) << 29;
        [w0, w1]
    };

    let values: &[u32] = &[
        0x3F80_0000, // +1.0
        0xBF80_0000, // -1.0
        0x0000_0000, // +0.0
        0x8000_0000, // -0.0
        0x7F80_0000, // +inf
        0xFF80_0000, // -inf
        0x0000_0001, // +min denormal
        0x8000_0001, // -min denormal
        0x007F_FFFF, // +max denormal
        0x807F_FFFF, // -max denormal
        0x7F80_0001, // +sNaN
        0xFF80_0001, // -sNaN
        0x7FC0_1234, // +qNaN with payload
        0xFFC0_1234, // -qNaN with payload
    ];

    // v_cmp_ne_u32_e64 s6, v3, 0 (VOP3 0x0c5);
    // v_cndmask_b32_e64 v0, -v1, |v2|, s6 (VOP3 0x101)
    let cmp_cond = vop3(0x0c5, 6, vgpr(3), 0x80, 0);
    let cnd_neg_abs = vop3_mods(0x101, 0, vgpr(1), vgpr(2), sgpr(6), 0b10, 0b01);
    let cndmask = build_valu_cs(&[cmp_cond[0], cmp_cond[1], cnd_neg_abs[0], cnd_neg_abs[1]]);
    let cnd_both = vop3_mods(0x101, 0, vgpr(1), vgpr(2), sgpr(6), 0b01, 0b11);
    let cndmask_both = build_valu_cs(&[cmp_cond[0], cmp_cond[1], cnd_both[0], cnd_both[1]]);
    let cnd_one_side = vop3_mods(0x101, 0, vgpr(1), vgpr(2), sgpr(6), 0, 0b01);
    let cndmask_one_side =
        build_valu_cs(&[cmp_cond[0], cmp_cond[1], cnd_one_side[0], cnd_one_side[1]]);

    let mut cnd_in = Vec::new();
    let mut cnd_want = Vec::new();
    for &x in values {
        // cond = 0 selects src0 (-v1); cond = 1 selects src1 (|v2|).
        cnd_in.push([x, 0x3F80_0000, 0, 0]);
        cnd_want.push(x ^ 0x8000_0000);
        cnd_in.push([0x3F80_0000, x, 1, 0]);
        cnd_want.push(x & 0x7FFF_FFFF);
    }
    run_cases(&cndmask, &cnd_in, |backend, i, inp, got| {
        let want = cnd_want[i];
        assert_eq!(
            got, want,
            "{backend}: v_cndmask_b32_e64(-{:#010x}, |{:#010x}|, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });

    let mut both_in = Vec::new();
    let mut both_want = Vec::new();
    for &x in values {
        // cond = 0 selects -|v1|; cond = 1 selects -v2.
        both_in.push([x, 0, 0, 0]);
        both_want.push(x | 0x8000_0000);
        both_in.push([0, x, 1, 0]);
        both_want.push(x ^ 0x8000_0000);
    }
    run_cases(&cndmask_both, &both_in, |backend, i, inp, got| {
        let want = both_want[i];
        assert_eq!(
            got, want,
            "{backend}: v_cndmask_b32_e64(-|{:#010x}|, -{:#010x}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });

    let mut one_in = Vec::new();
    let mut one_want = Vec::new();
    for &x in values {
        // cond = 0 selects -v1; cond = 1 passes unmodified v2 through bitwise.
        one_in.push([x, 0, 0, 0]);
        one_want.push(x ^ 0x8000_0000);
        one_in.push([0, x, 1, 0]);
        one_want.push(x);
    }
    run_cases(&cndmask_one_side, &one_in, |backend, i, inp, got| {
        let want = one_want[i];
        assert_eq!(
            got, want,
            "{backend}: v_cndmask_b32_e64(-{:#010x}, {:#010x}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
}

/// EXP `exp target, v0, v1, v2, v3 [done] [vm]` (`en`: 4-bit channel mask).
const fn exp_inst(
    target: u8,
    en: u8,
    v: [u8; 4],
    done: bool,
    vm: bool,
) -> [u32; 2] {
    let w0 = 0x3E << 26
        | (vm as u32) << 12
        | (done as u32) << 11
        | (target as u32 & 0x3F) << 4
        | (en as u32 & 0xF);
    let w1 = (v[3] as u32) << 24 | (v[2] as u32) << 16 | (v[1] as u32) << 8 | (v[0] as u32);
    [w0, w1]
}

#[test]
fn vs_rt_and_viewport_index_exports_validate_and_create_pso() {
    use super::container::SigElement;
    use crate::dxbc::{Container, FourCC};

    let build_stage = |stage: HwStage,
                       words: &[u32],
                       layout: Vec<RootParam>,
                       stage_io: StageIo,
                       outputs: Vec<SigElement>|
     -> HostShader {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage,
            wave64: false,
            vgprs: 8,
            layout,
            used: Vec::new(),
            io: stage_io,
            code,
        };
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        let st = structure::structure(&p).expect("structure");
        let res = resource::track(&p, &native.layout, native.stage).expect("resource::track");
        let io = io::raise(&p, &native).expect("io::raise");
        let lowered = lower::lower(&lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs,
            prims: Vec::new(),
            mesh_stride: 0,
            options: super::TranslateOptions::default(),
        })
        .expect("lower");
        let shader = HostShader {
            stage,
            dxil: lowered.dxil,
            user_data: lowered.user_data,
            samplers: lowered.samplers,
            static_samplers: Vec::new(),
            root_cbvs: lowered.root_cbvs,
            root_srvs: lowered.root_srvs,
            address_srvs: lowered.address_srvs,
            stride_table: lowered.stride_table,
        };
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            let mut copy = shader.dxil.clone();
            let ours = copy[4..20].to_vec();
            v.validate(&mut copy).expect("DXIL validation failed");
            assert_eq!(copy[4..20], ours[..], "container hash mismatch");
        }
        shader
    };

    // Minimal pixel shader writing SV_Target0 = (1, 0, 0, 1).
    let ps_words = [
        vop1(0x001, 0, 0xF2), // v0 = 1.0
        vop1(0x001, 1, 0x80), // v1 = 0.0
        vop1(0x001, 2, 0x80), // v2 = 0.0
        vop1(0x001, 3, 0xF2), // v3 = 1.0
        exp_inst(0, 0xF, [0, 1, 2, 3], true, true)[0],
        exp_inst(0, 0xF, [0, 1, 2, 3], true, true)[1],
        S_ENDPGM,
    ];
    let ps = build_stage(
        HwStage::Ps,
        &ps_words,
        Vec::new(),
        StageIo::Ps {
            inputs: Vec::new(),
            input_addr: 0,
            db_shader_control: 0,
            patches: Vec::new(),
        },
        vec![SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0,
            half: false,
        }],
    );

    // Three VS variants: RT index only, viewport index only, and both + clip distance in pos2.
    let configs = [
        (true, false, false),
        (false, true, false),
        (true, true, true),
    ];
    let mut vs_shaders = Vec::new();
    for (has_rt, has_vp, has_clip) in configs {
        let mut words = vec![
            vop1(0x001, 0, 0x80),    // pos.x = 0.0
            vop1(0x001, 1, 0x80),    // pos.y = 0.0
            vop1(0x001, 2, 0x80),    // pos.z = 0.0
            vop1(0x001, 3, 0xF2),    // pos.w = 1.0
            vop1(0x001, 4, sgpr(0)), // pos1.z = packed (viewport << 16) | rt
            vop1(0x001, 5, sgpr(1)), // pos2.x = clip distance
        ];
        words.extend_from_slice(&exp_inst(12, 0xF, [0, 1, 2, 3], false, false)); // pos0
        words.extend_from_slice(&exp_inst(13, 0x4, [0, 0, 4, 0], !has_clip, false)); // pos1.z
        if has_clip {
            words.extend_from_slice(&exp_inst(14, 0x1, [5, 0, 0, 0], true, false)); // pos2.x
        }
        words.push(S_ENDPGM);

        let mut outputs = vec![SigElement {
            name: "SV_Position".into(),
            index: 0,
            kind: 3,
            comp: 9,
            interp: 4,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0,
            half: false,
        }];
        let mut next_row = 1u32;
        if has_rt {
            outputs.push(SigElement {
                name: "SV_RenderTargetArrayIndex".into(),
                index: 0,
                kind: 4,
                comp: 5,
                interp: 1,
                row: Some(next_row),
                col: 0,
                cols: 1,
                usage: 0,
                half: false,
            });
            next_row += 1;
        }
        if has_vp {
            outputs.push(SigElement {
                name: "SV_ViewportArrayIndex".into(),
                index: 0,
                kind: 5,
                comp: 5,
                interp: 1,
                row: Some(next_row),
                col: 0,
                cols: 1,
                usage: 0,
                half: false,
            });
            next_row += 1;
        }
        if has_clip {
            outputs.push(SigElement {
                name: "SV_ClipDistance".into(),
                index: 0,
                kind: 6,
                comp: 9,
                interp: 2,
                row: Some(next_row),
                col: 0,
                cols: 1,
                usage: 0,
                half: false,
            });
        }

        let vs = build_stage(
            HwStage::Vs,
            &words,
            vec![RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            StageIo::Vs {
                inputs: Vec::new(),
                outputs: Vec::new(),
            },
            outputs,
        );
        let c = Container::parse(&vs.dxil).expect("parse emitted VS DXIL");
        let sfi0 = u64::from_le_bytes(
            c.part(FourCC::SFI0).expect("SFI0").data[..8]
                .try_into()
                .unwrap(),
        );
        assert_ne!(
            sfi0 & 0x2000,
            0,
            "SFI0 should set ViewPortAndRTArrayIndexFromAnyShaderFeedingRasterizer"
        );
        vs_shaders.push(vs);
    }

    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        for (i, vs) in vs_shaders.iter().enumerate() {
            host.pipeline(vs, &ps)
                .unwrap_or_else(|e| panic!("{label}: graphics pipeline case {i} failed: {e}"));
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP supports SM 6.6");
}

/// SMEM `s_load_dwordx8 s[dst:dst+7], s[base:base+1], 0` (soffset = null).
const fn s_load_dwordx8(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x03 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// MIMG `image_store vdata, vaddr, s[srsrc:srsrc+7] dmask:dmask dim:2D unorm [d16]`.
const fn image_store_2d(vdata: u8, vaddr: u8, srsrc: u8, dmask: u8, d16: bool) -> [u32; 2] {
    let w0 = 0x3C << 26 | 0x08 << 18 | 1 << 12 | (dmask as u32 & 0xF) << 8 | 1 << 3;
    let w1 = (d16 as u32) << 31
        | ((srsrc as u32 >> 2) & 0x1F) << 16
        | (vdata as u32) << 8
        | (vaddr as u32);
    [w0, w1]
}

fn try_build_image_store_cs(dmask: u8, d16: bool) -> Result<HostShader, crate::ShaderError> {
    let mut words = Vec::new();
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push(s_mov_b32(5, 0x80)); // inline constant 0
    words.extend_from_slice(&s_load_dwordx8(8, 4));
    // Data in v1..v4 from root constants s0..s3; 2D texel coordinate (0, 0) in v5..v6.
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, sgpr(i))); // v_mov_b32
    }
    words.push(vop1(0x001, 5, 0x80)); // v5 = 0
    words.push(vop1(0x001, 6, 0x80)); // v6 = 0
    words.extend_from_slice(&image_store_2d(1, 5, 8, dmask, d16));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [1, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage)?;
    let io = io::raise(&p, &native)?;
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        options: super::TranslateOptions::default(),
    })?;

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        address_srvs: lowered.address_srvs,
        stride_table: lowered.stride_table,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    Ok(shader)
}

#[test]
fn image_store_partial_dmask_writes_expected_channels() {
    // Non-contiguous dmask (e.g. 0x5 = x|z) is rejected because DXIL
    // `textureStore` requires a contiguous mask from channel 0.
    assert!(matches!(
        try_build_image_store_cs(0x5, false),
        Err(crate::ShaderError::Unsupported {
            what: "an image store of some channels",
            ..
        })
    ));

    // (dmask, d16, input root constants, expected channels written)
    let cases: &[(u8, bool, [u32; 4], &[f32])] = &[
        (0x0, false, [1.5f32.to_bits(), 0, 0, 0], &[0.0, 0.0, 0.0, 0.0]),
        (0x1, false, [1.5f32.to_bits(), 0, 0, 0], &[1.5]),
        (
            0x3,
            false,
            [1.5f32.to_bits(), (-2.25f32).to_bits(), 0, 0],
            &[1.5, -2.25],
        ),
        (
            0x7,
            false,
            [
                1.5f32.to_bits(),
                (-2.25f32).to_bits(),
                4.0f32.to_bits(),
                0,
            ],
            &[1.5, -2.25, 4.0],
        ),
        // d16: 1.5 (0x3E00) in low half, -2.0 (0xC000) in high half, 4.0 (0x4400) in next VGPR.
        (0x1, true, [0x0000_3E00, 0, 0, 0], &[1.5]),
        (0x3, true, [0xC000_3E00, 0, 0, 0], &[1.5, -2.0]),
        (0x7, true, [0xC000_3E00, 0x0000_4400, 0, 0], &[1.5, -2.0, 4.0]),
    ];

    let shaders: Vec<HostShader> = cases
        .iter()
        .map(|&(dmask, d16, ..)| {
            try_build_image_store_cs(dmask, d16)
                .unwrap_or_else(|e| panic!("dmask={dmask:#x} d16={d16}: {e}"))
        })
        .collect();

    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        for (&(dmask, d16, inputs, want), shader) in cases.iter().zip(&shaders) {
            let (root, pso) = host
                .compute_pipeline(shader)
                .unwrap_or_else(|e| panic!("{label}: dmask={dmask:#x} d16={d16}: {e}"));
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    (root, _) => super::host::guard(root, 1, 0)
                        .unwrap_or_else(|| panic!("unexpected user data {:?}", (root, u.slot))),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::TextureUav(1, 1)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            let got: Vec<f32> = out[0][..want.len()]
                .iter()
                .map(|&b| f32::from_bits(b))
                .collect();
            assert_eq!(
                got.as_slice(),
                want,
                "{label}: image_store dmask={dmask:#x} d16={d16} wrote {got:?}, want {want:?}"
            );
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP supports SM 6.6");
}

/// A descriptor at a heap address the program's data computes (the table
/// address plus an index, as bindless code adds one it loads). Stale data
/// gives any index; one past the host heap is undefined: NVIDIA wraps it
/// onto another slot (without the guard, index 2^25 writes through slot 0
/// here), and a descriptor of another kind there hangs the GPU. Such an
/// index reads the heap's guard slot (null) instead: the store is dropped.
/// In range, the store lands.
#[test]
fn a_heap_index_past_the_heap_reads_the_guard_slot() {
    let words = [
        s_mov_b32(5, 0x80),     // s5 = 0
        sop2(0x1E, 6, 0, 0x85), // s_lshl_b32 s6, s0, 5: the index from data
        // s_load_dwordx4 s[8:11], s[4:5], s6
        s_load_dwordx4(8, 4)[0],
        6 << 25,
        vop1(0x001, 0, 0x87), // v_mov_b32 v0, 7
        buffer_store_dword(0, 8, 0)[0],
        buffer_store_dword(0, 8, 0)[1],
        S_ENDPGM,
    ];
    let shader = lower_cs(&words);
    // The guard is a user-data dword of its own, past the console's.
    let guard: Vec<_> = shader
        .user_data
        .iter()
        .filter(|u| u.root == RootType::ResourceHeap)
        .collect();
    assert_eq!(guard.len(), 1, "{:?}", shader.user_data);
    assert_eq!(guard[0].index, 5, "after s0..s4");
    let cases = [
        [0, 0, 0, 0],           // heap index 0: the UAV
        [1, 0, 0, 0],           // the guard slot itself
        [1 << 25, 0, 0, 0],     // far past the heap
        [0x07FF_FFFF, 0, 0, 0], // the last index an address gives
    ];
    run_cases(&shader, &cases, |label, i, k, got| {
        let want = if i == 0 { 7 } else { 0 };
        assert_eq!(got, want, "{label}: index {:#x}", k[0]);
    });
}

#[test]
fn v_mad_16_and_int16_clamp_match_rdna2() {
    let vop3_clamp = |op: u16, s0: u16, s1: u16, s2: u16, clamp: bool| {
        let [mut w0, w1] = vop3(op, 0, s0, s1, s2);
        if clamp {
            w0 |= 1 << 15;
        }
        [w0, w1]
    };
    let with_sentinel = |words: [u32; 2]| {
        build_valu_cs(&[
            vop1(0x001, 0, vgpr(4)), // v_mov_b32 v0, v4 (sentinel upper half)
            words[0],
            words[1],
        ])
    };

    // VOP3 opcodes: 0x303=v_add_nc_u16, 0x304=v_sub_nc_u16, 0x340=v_mad_u16, 0x35e=v_mad_i16.
    let mad_u16_wrap = with_sentinel(vop3_clamp(0x340, vgpr(1), vgpr(2), vgpr(3), false));
    let mad_u16_sat = with_sentinel(vop3_clamp(0x340, vgpr(1), vgpr(2), vgpr(3), true));
    let mad_i16_wrap = with_sentinel(vop3_clamp(0x35e, vgpr(1), vgpr(2), vgpr(3), false));
    let mad_i16_sat = with_sentinel(vop3_clamp(0x35e, vgpr(1), vgpr(2), vgpr(3), true));
    let add_u16_sat = with_sentinel(vop3_clamp(0x303, vgpr(1), vgpr(2), 0, true));
    let sub_u16_sat = with_sentinel(vop3_clamp(0x304, vgpr(1), vgpr(2), 0, true));

    let sentinel = 0xBEEF_0000u32;

    // v_mad_u16 wrapping & saturating (including u16::MAX * u16::MAX + u16::MAX without u32 wrap)
    let u_cases: &[(u16, u16, u16, u16, u16)] = &[
        (20, 30, 40, 640, 640),
        (300, 300, 100, (90_100u32 & 0xFFFF) as u16, u16::MAX),
        (u16::MAX, u16::MAX, u16::MAX, 0, u16::MAX),
    ];
    let u_in: Vec<[u32; 4]> = u_cases
        .iter()
        .map(|&(a, b, c, _, _)| [u32::from(a), u32::from(b), u32::from(c), sentinel])
        .collect();
    run_cases(&mad_u16_wrap, &u_in, |backend, i, _, got| {
        let want = sentinel | u32::from(u_cases[i].3);
        assert_eq!(got, want, "{backend}: v_mad_u16 wrap case {i}");
    });
    run_cases(&mad_u16_sat, &u_in, |backend, i, _, got| {
        let want = sentinel | u32::from(u_cases[i].4);
        assert_eq!(got, want, "{backend}: v_mad_u16 clamp case {i}");
    });

    // v_mad_i16 wrapping & saturating (positive & negative overflow extremes)
    let i_cases: &[(i16, i16, i16, i16, i16)] = &[
        (-10, 20, 50, -150, -150),
        (i16::MIN, i16::MIN, i16::MAX, i16::MAX, i16::MAX),
        (i16::MIN, i16::MAX, i16::MIN, 0, i16::MIN),
        (200, 200, 100, (40_100i32 as i16), i16::MAX),
        (-200, 200, -100, (-40_100i32 as i16), i16::MIN),
    ];
    let i_in: Vec<[u32; 4]> = i_cases
        .iter()
        .map(|&(a, b, c, _, _)| [u32::from(a as u16), u32::from(b as u16), u32::from(c as u16), sentinel])
        .collect();
    run_cases(&mad_i16_wrap, &i_in, |backend, i, _, got| {
        let want = sentinel | u32::from(i_cases[i].3 as u16);
        assert_eq!(got, want, "{backend}: v_mad_i16 wrap case {i}");
    });
    run_cases(&mad_i16_sat, &i_in, |backend, i, _, got| {
        let want = sentinel | u32::from(i_cases[i].4 as u16);
        assert_eq!(got, want, "{backend}: v_mad_i16 clamp case {i}");
    });

    // v_add_nc_u16 clamp & v_sub_nc_u16 clamp
    run_cases(
        &add_u16_sat,
        &[
            [60000, 10000, 0, sentinel],
            [123, 456, 0, sentinel],
        ],
        |backend, i, _, got| {
            let want = sentinel | [u32::from(u16::MAX), 579][i];
            assert_eq!(got, want, "{backend}: v_add_nc_u16 clamp case {i}");
        },
    );
    run_cases(
        &sub_u16_sat,
        &[
            [5, 10, 0, sentinel],
            [500, 200, 0, sentinel],
        ],
        |backend, i, _, got| {
            let want = sentinel | [0, 300][i];
            assert_eq!(got, want, "{backend}: v_sub_nc_u16 clamp case {i}");
        },
    );
}
