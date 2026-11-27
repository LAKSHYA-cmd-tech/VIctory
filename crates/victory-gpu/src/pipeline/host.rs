//! Host pipelines for title pipelines (decision 0007, step 3): the
//! title's graphics state with the translated shaders, under a root
//! signature of our own that the translated shaders' bindings define
//! (`victory_shader::emit`): per stage one block of root constants for the
//! user data, root views for the title's root views and for addresses the
//! title passes in root constants, static samplers for the immediate
//! samplers, and directly indexed heaps.
//!
//! A pipeline with a geometry stage runs the ES half of its vertex
//! container as the vertex shader and the GS half of its geometry
//! container, each with its own bindings as the other stages have.

use std::ffi::{CStr, CString};
use std::sync::Arc;

use victory_shader::emit::{
    AddressView, ROOT_ADDRESS_SPACE, ROOT_CBV_SPACE, SAMPLER_SPACE, STRIDE_SPACE, USER_DATA_SPACE,
};
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::PCSTR;

use super::{Program, Shader, Translated};
use crate::host::host;

/// A stream's graphics state, owned (the title's stream lives only for the
/// creation call). Subobjects the stream leaves out keep D3D12's defaults.
pub(crate) struct GraphicsState {
    blend: D3D12_BLEND_DESC,
    sample_mask: u32,
    rasterizer: D3D12_RASTERIZER_DESC,
    depth: D3D12_DEPTH_STENCIL_DESC,
    /// Input elements, their names pointing into `names`.
    inputs: Vec<D3D12_INPUT_ELEMENT_DESC>,
    names: Vec<CString>,
    strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE,
    pub(crate) topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE,
    pub(crate) targets: D3D12_RT_FORMAT_ARRAY,
    depth_format: DXGI_FORMAT,
    samples: DXGI_SAMPLE_DESC,
}

impl Default for GraphicsState {
    fn default() -> Self {
        let target = D3D12_RENDER_TARGET_BLEND_DESC {
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_ZERO,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_ZERO,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        };
        let keep = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        };
        GraphicsState {
            blend: D3D12_BLEND_DESC {
                RenderTarget: [target; 8],
                ..Default::default()
            },
            sample_mask: u32::MAX,
            rasterizer: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_BACK,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            depth: D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: true.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                DepthFunc: D3D12_COMPARISON_FUNC_LESS,
                StencilReadMask: 0xff,
                StencilWriteMask: 0xff,
                FrontFace: keep,
                BackFace: keep,
                ..Default::default()
            },
            inputs: Vec::new(),
            names: Vec::new(),
            strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
            topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED,
            targets: D3D12_RT_FORMAT_ARRAY::default(),
            depth_format: DXGI_FORMAT_UNKNOWN,
            samples: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
        }
    }
}

/// Integer render target formats: D3D12 cannot blend into them.
fn integer_format(f: DXGI_FORMAT) -> bool {
    use windows::Win32::Graphics::Dxgi::Common::*;
    matches!(
        f,
        DXGI_FORMAT_R32G32B32A32_UINT
            | DXGI_FORMAT_R32G32B32A32_SINT
            | DXGI_FORMAT_R32G32B32_UINT
            | DXGI_FORMAT_R32G32B32_SINT
            | DXGI_FORMAT_R16G16B16A16_UINT
            | DXGI_FORMAT_R16G16B16A16_SINT
            | DXGI_FORMAT_R32G32_UINT
            | DXGI_FORMAT_R32G32_SINT
            | DXGI_FORMAT_R10G10B10A2_UINT
            | DXGI_FORMAT_R8G8B8A8_UINT
            | DXGI_FORMAT_R8G8B8A8_SINT
            | DXGI_FORMAT_R16G16_UINT
            | DXGI_FORMAT_R16G16_SINT
            | DXGI_FORMAT_R32_UINT
            | DXGI_FORMAT_R32_SINT
            | DXGI_FORMAT_R8G8_UINT
            | DXGI_FORMAT_R8G8_SINT
            | DXGI_FORMAT_R16_UINT
            | DXGI_FORMAT_R16_SINT
            | DXGI_FORMAT_R8_UINT
            | DXGI_FORMAT_R8_SINT
    )
}

impl GraphicsState {
    /// The guest color buffer bypasses its blender for integer surfaces,
    /// even when the draw asks for blending. D3D12 validates blend enable
    /// against every active target format when the pipeline is created.
    pub(crate) fn apply_integer_blend_bypass(&mut self) {
        let integer = |f: &DXGI_FORMAT| integer_format(*f);
        let formats = &self.targets.RTFormats[..(self.targets.NumRenderTargets as usize).min(8)];
        if !self.blend.IndependentBlendEnable.as_bool() {
            let shared = self.blend.RenderTarget[0];
            if !shared.BlendEnable.as_bool() || !formats.iter().any(integer) {
                return;
            }
            // With shared blending D3D12 reads slot zero for every target.
            // Expand it before overriding individual targets, including
            // their shared write mask.
            self.blend.RenderTarget[..formats.len()].fill(shared);
            self.blend.IndependentBlendEnable = (formats.len() > 1).into();
        }
        for (format, target) in formats.iter().zip(&mut self.blend.RenderTarget) {
            if integer(format) {
                target.BlendEnable = false.into();
            }
        }
    }
    /// Takes in one subobject of a stream: its type, then its data at +4
    /// (+8 for those holding pointers).
    ///
    /// # Safety
    /// `at` points at a whole subobject of type `kind`.
    pub(crate) unsafe fn read(&mut self, kind: u32, at: *const u8) {
        // SAFETY: the caller passes a whole subobject.
        unsafe {
            let data = at.add(4);
            match kind {
                8 => self.blend = data.cast::<D3D12_BLEND_DESC>().read_unaligned(),
                9 => self.sample_mask = data.cast::<u32>().read_unaligned(),
                10 => self.rasterizer = data.cast::<D3D12_RASTERIZER_DESC>().read_unaligned(),
                11 => self.depth = data.cast::<D3D12_DEPTH_STENCIL_DESC>().read_unaligned(),
                21 => {
                    // shortcut: the depth bounds test is dropped (the host
                    // pipeline uses the plain description); bring it back
                    // with a stream if a title enables it.
                    let d = data.cast::<D3D12_DEPTH_STENCIL_DESC1>().read_unaligned();
                    self.depth = D3D12_DEPTH_STENCIL_DESC {
                        DepthEnable: d.DepthEnable,
                        DepthWriteMask: d.DepthWriteMask,
                        DepthFunc: d.DepthFunc,
                        StencilEnable: d.StencilEnable,
                        StencilReadMask: d.StencilReadMask,
                        StencilWriteMask: d.StencilWriteMask,
                        FrontFace: d.FrontFace,
                        BackFace: d.BackFace,
                    };
                }
                12 => {
                    let layout = at.add(8).cast::<D3D12_INPUT_LAYOUT_DESC>().read_unaligned();
                    if layout.pInputElementDescs.is_null() {
                        return;
                    }
                    let elements = std::slice::from_raw_parts(
                        layout.pInputElementDescs,
                        layout.NumElements as usize,
                    );
                    self.names = elements
                        .iter()
                        .map(|e| CStr::from_ptr(e.SemanticName.0.cast()).to_owned())
                        .collect();
                    self.inputs = elements
                        .iter()
                        .zip(&self.names)
                        .map(|(e, name)| D3D12_INPUT_ELEMENT_DESC {
                            SemanticName: PCSTR(name.as_ptr().cast()),
                            Format: DXGI_FORMAT(
                                crate::resource::pc_format(e.Format.0 as u32) as i32
                            ),
                            ..*e
                        })
                        .collect();
                }
                13 => {
                    self.strip_cut =
                        D3D12_INDEX_BUFFER_STRIP_CUT_VALUE(data.cast::<i32>().read_unaligned())
                }
                14 => {
                    self.topology =
                        D3D12_PRIMITIVE_TOPOLOGY_TYPE(data.cast::<i32>().read_unaligned())
                }
                15 => {
                    self.targets = data.cast::<D3D12_RT_FORMAT_ARRAY>().read_unaligned();
                    let count = (self.targets.NumRenderTargets as usize).min(8);
                    for f in &mut self.targets.RTFormats[..count] {
                        f.0 = crate::resource::pc_format(f.0 as u32) as i32;
                    }
                    for f in &mut self.targets.RTFormats[count..] {
                        *f = DXGI_FORMAT_UNKNOWN;
                    }
                }
                16 => {
                    let f = data.cast::<u32>().read_unaligned();
                    self.depth_format = DXGI_FORMAT(crate::resource::pc_format(f) as i32);
                }
                17 => self.samples = data.cast::<DXGI_SAMPLE_DESC>().read_unaligned(),
                _ => {}
            }
        }
    }

    /// Everything the state holds, as text: part of a pipeline's name in
    /// the library and of the key a title pipeline is started early by.
    pub(crate) fn describe(&self) -> String {
        let inputs: Vec<String> = self
            .inputs
            .iter()
            .zip(&self.names)
            .map(|(e, name)| {
                format!(
                    "{name:?} {} {} {} {} {} {}",
                    e.SemanticIndex,
                    e.Format.0,
                    e.InputSlot,
                    e.AlignedByteOffset,
                    e.InputSlotClass.0,
                    e.InstanceDataStepRate
                )
            })
            .collect();
        format!(
            "{:?} {} {:?} {:?} {inputs:?} {} {} {:?} {} {:?}",
            self.blend,
            self.sample_mask,
            self.rasterizer,
            self.depth,
            self.strip_cut.0,
            self.topology.0,
            self.targets,
            self.depth_format.0,
            self.samples
        )
    }

    /// The vertex buffer slots the input layout reads.
    fn slots(&self) -> u32 {
        self.inputs
            .iter()
            .fold(0, |m, e| m | 1 << e.InputSlot.min(31))
    }

    /// The input elements with their byte offsets worked out.
    fn elements(&self) -> Vec<Input> {
        let mut next = [0u32; 32];
        self.inputs
            .iter()
            .map(|e| {
                let slot = e.InputSlot.min(31) as usize;
                let format = e.Format.0 as u32;
                let offset = if e.AlignedByteOffset == D3D12_APPEND_ALIGNED_ELEMENT {
                    next[slot]
                } else {
                    e.AlignedByteOffset
                };
                let bytes = crate::resource::element(format).map_or(0, |(b, _)| b as u32);
                next[slot] = offset + bytes;
                Input {
                    slot: slot as u32,
                    offset,
                    format,
                    per_instance: e.InputSlotClass == D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA,
                }
            })
            .collect()
    }
}

/// One input element: vertex buffer slot, byte offset, format.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Input {
    pub(crate) slot: u32,
    pub(crate) offset: u32,
    pub(crate) format: u32,
    pub(crate) per_instance: bool,
}

/// The console's own primitive topology types.
const RECTANGLES: i32 = 19;
const QUADS: i32 = 32;

/// Whether root SRVs are bound as one-view tables of the buffer's host
/// copy (reads past its end return zeros) rather than root descriptors
/// (unbounded: a mesh shader's vertex fetch, which has no record limit,
/// read past the copy into memory the host had freed). Set
/// `VICTORY_ROOT_SRV_DIRECT=1` to go back.
pub(crate) fn bounded_root_srvs() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ROOT_SRV_DIRECT").is_none())
}

/// One stage of a host pipeline and where its bindings go.
pub(crate) struct Stage {
    pub(crate) shader: Arc<Shader>,
    /// Which of its container's programs runs.
    pub(crate) program: Program,
    /// The root parameter of its user data, and its size in dwords.
    pub(crate) user_data: Option<(u32, u32)>,
    /// Root parameters of its root views: (title root parameter, ours).
    pub(crate) root_cbvs: Vec<(u8, u32)>,
    pub(crate) root_srvs: Vec<(u8, u32)>,
    /// Root parameters of its root SRVs at addresses in root constants:
    /// (the two constants, ours), `t<n>` in [`ROOT_ADDRESS_SPACE`] for the
    /// one at position n.
    pub(crate) address_srvs: Vec<(AddressView, u32)>,
    /// Root parameter of the runtime stride table (`t0`, [`STRIDE_SPACE`]).
    pub(crate) stride_table: Option<u32>,
}

impl Stage {
    pub(crate) fn translated(&self) -> &Translated {
        self.shader
            .translated_as(self.program)
            .expect("host pipelines are made from translated shaders")
    }
}

pub(crate) struct HostPipeline {
    /// A compute pipeline (else graphics).
    pub(crate) compute: bool,
    /// A mesh pipeline (graphics without vertex input).
    pub(crate) mesh: bool,
    pub(crate) pso: ID3D12PipelineState,
    pub(crate) root: ID3D12RootSignature,
    /// Vertex (or mesh), geometry if any, then pixel stage; or the compute
    /// stage.
    pub(crate) stages: Vec<Stage>,
    /// The vertex buffer slots it reads (bit per slot).
    pub(crate) slots: u32,
    pub(crate) inputs: Vec<Input>,
    /// Its shaders, for reports of slow submissions.
    pub(crate) name: String,
}

/// Reads the required wave-lane range `(min, max)` from an emitted DXIL
/// container's `PSV0` runtime info (`MinimumExpectedWaveLaneCount` and
/// `MaximumExpectedWaveLaneCount`). Returns `None` when unconstrained
/// (`min == 0` and `max == u32::MAX` or `0`) or absent.
fn dxil_wave_lanes(dxil: &[u8]) -> Option<(u32, u32)> {
    let container = victory_shader::Container::parse(dxil).ok()?;
    let psv = container.part(victory_shader::FourCC::PSV0)?.data;
    if psv.len() < 28 {
        return None;
    }
    let info_size = u32::from_le_bytes(psv[0..4].try_into().ok()?);
    if info_size < 24 {
        return None;
    }
    let min = u32::from_le_bytes(psv[20..24].try_into().ok()?);
    let max = u32::from_le_bytes(psv[24..28].try_into().ok()?);
    if min == 0 && (max == 0 || max == u32::MAX) {
        return None;
    }
    Some((min.max(1), if max == 0 { u32::MAX } else { max }))
}

/// Whether a translated `stage` program cannot run on a host whose waves have
/// `host_lanes = (min, max)` lanes.
///
/// - If the emitted DXIL module declares a `PSV0` wave-lane range (e.g.
///   `WaveSize(64)` or `WaveSize(32)`), it is refused whenever the host's
///   `[min, max]` range does not overlap it. This lets per-lane or split
///   wave64 compute translations (declaring `WaveSize(32)` or unconstrained)
///   run on 32-lane hosts while still refusing modules that require 64 lanes
///   (and refusing 32/64-lane modules on 4-lane WARP).
/// - Console `wave64` mesh programs do not carry `PSV0` `WaveSize(64)`
///   metadata, so they are refused whenever `host_lanes.1 < 64`.
/// - Vertex and pixel programs run per lane and are kept unless their emitted
///   DXIL explicitly requires an unsupported wave size.
fn wave_lanes_unsupported(
    stage: &str,
    console_wave64: bool,
    emitted_wave: Option<(u32, u32)>,
    (host_min, host_max): (u32, u32),
) -> bool {
    if let Some((req_min, req_max)) = emitted_wave {
        return host_max < req_min || host_min > req_max;
    }
    stage == "mesh" && console_wave64 && host_max < 64
}

/// Logs a pipeline that gets no host pipeline, once per reason.
fn refuse(reason: &'static str, shaders: &[Arc<Shader>]) -> Option<HostPipeline> {
    static SEEN: std::sync::Mutex<Vec<&str>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&reason) {
        seen.push(reason);
        tracing::warn!(?shaders, "no host pipeline: {reason} (logged once)");
    }
    None
}

impl HostPipeline {
    /// A host pipeline for vertex (or mesh), optionally geometry, and pixel
    /// shaders with the graphics state `g`, or for one compute shader.
    pub(crate) fn new(shaders: &[Arc<Shader>], g: Option<&GraphicsState>) -> Option<HostPipeline> {
        let find = |stage| shaders.iter().find(|s| s.stage == stage);
        let compute = match shaders {
            [cs] if cs.stage == "compute" => Some(cs),
            _ => None,
        };
        if compute.is_none()
            && shaders
                .iter()
                .any(|s| !matches!(s.stage, "vertex" | "pixel" | "mesh" | "geometry"))
        {
            return refuse(
                "only vertex, mesh, geometry, pixel and compute stages are translated",
                shaders,
            );
        }
        let mesh = find("mesh");
        let geometry = find("geometry");
        let stages = if let Some(cs) = compute {
            vec![(cs, D3D12_SHADER_VISIBILITY_ALL, Program::Main)]
        } else {
            let first = match (mesh, find("vertex"), geometry) {
                (Some(_), _, Some(_)) => {
                    return refuse("a mesh pipeline with a geometry stage", shaders);
                }
                (Some(ms), _, None) => (ms, D3D12_SHADER_VISIBILITY_MESH, Program::Main),
                (None, Some(vs), gs) => (
                    vs,
                    D3D12_SHADER_VISIBILITY_VERTEX,
                    Program::of(vs.stage, gs.is_some()),
                ),
                (None, None, _) => return refuse("no vertex or mesh shader", shaders),
            };
            let mut stages = vec![first];
            stages.extend(geometry.map(|gs| (gs, D3D12_SHADER_VISIBILITY_GEOMETRY, Program::Gs)));
            stages
                .extend(find("pixel").map(|ps| (ps, D3D12_SHADER_VISIBILITY_PIXEL, Program::Main)));
            stages
        };
        if mesh.is_some() && !host()?.mesh_shaders() {
            return refuse(
                "the host GPU has no mesh shaders (or no shader model 6.6)",
                shaders,
            );
        }
        let host_lanes = host()?.wave_lanes();
        if let Some(ms) = mesh
            && wave_lanes_unsupported(ms.stage, ms.wave64(), None, host_lanes)
        {
            return refuse(
                if host_lanes.1 < 32 {
                    "a mesh program written for 64-lane waves, on a host without them (e.g. 4-lane WARP)"
                } else {
                    "a mesh program written for 64-lane waves, on a host without them"
                },
                shaders,
            );
        }
        for &(s, _, program) in &stages {
            let t = s.translated_as(program)?;
            let emitted = dxil_wave_lanes(&t.host.dxil);
            if wave_lanes_unsupported(s.stage, s.wave64(), emitted, host_lanes) {
                return refuse(
                    if host_lanes.1 < 32 {
                        "a shader requiring wider waves than this host supports (e.g. 4-lane WARP)"
                    } else {
                        "a shader requiring 64-lane waves, on a host without them"
                    },
                    shaders,
                );
            }
        }
        let host = host()?;

        // The root signature the translated shaders' bindings define.
        let mut params: Vec<D3D12_ROOT_PARAMETER1> = Vec::new();
        // Ranges the table parameters point at, alive until the root
        // signature is made.
        let mut ranges: Vec<Box<D3D12_DESCRIPTOR_RANGE1>> = Vec::new();
        let mut samplers = Vec::new();
        let mut bound = Vec::new();
        for &(shader, visibility, program) in &stages {
            let t = &shader.translated_as(program)?.host;
            let mut stage = Stage {
                shader: shader.clone(),
                program,
                user_data: None,
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                address_srvs: Vec::new(),
                stride_table: None,
            };
            let dwords = t.user_data_dwords();
            if dwords > 0 {
                stage.user_data = Some((params.len() as u32, dwords));
                params.push(D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Constants: D3D12_ROOT_CONSTANTS {
                            ShaderRegister: 0,
                            RegisterSpace: USER_DATA_SPACE,
                            Num32BitValues: dwords,
                        },
                    },
                    ShaderVisibility: visibility,
                });
            }
            for (slots, kind, out) in [
                (
                    &t.root_cbvs,
                    D3D12_ROOT_PARAMETER_TYPE_CBV,
                    &mut stage.root_cbvs,
                ),
                (
                    &t.root_srvs,
                    D3D12_ROOT_PARAMETER_TYPE_SRV,
                    &mut stage.root_srvs,
                ),
            ] {
                for &slot in slots {
                    if kind == D3D12_ROOT_PARAMETER_TYPE_SRV && bounded_root_srvs() {
                        // A one-view table: the view's size bounds the reads,
                        // which a root descriptor leaves unbounded.
                        ranges.push(Box::new(D3D12_DESCRIPTOR_RANGE1 {
                            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                            NumDescriptors: 1,
                            BaseShaderRegister: u32::from(slot),
                            RegisterSpace: ROOT_CBV_SPACE,
                            Flags: D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE
                                | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE,
                            OffsetInDescriptorsFromTableStart: 0,
                        }));
                        let range: *const D3D12_DESCRIPTOR_RANGE1 = &**ranges.last()?;
                        out.push((slot, params.len() as u32));
                        params.push(D3D12_ROOT_PARAMETER1 {
                            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 {
                                    NumDescriptorRanges: 1,
                                    pDescriptorRanges: range,
                                },
                            },
                            ShaderVisibility: visibility,
                        });
                        continue;
                    }
                    out.push((slot, params.len() as u32));
                    params.push(D3D12_ROOT_PARAMETER1 {
                        ParameterType: kind,
                        Anonymous: D3D12_ROOT_PARAMETER1_0 {
                            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                                ShaderRegister: u32::from(slot),
                                RegisterSpace: ROOT_CBV_SPACE,
                                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                            },
                        },
                        ShaderVisibility: visibility,
                    });
                }
            }
            for (n, &view) in t.address_srvs.iter().enumerate() {
                stage.address_srvs.push((view, params.len() as u32));
                params.push(D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                            ShaderRegister: n as u32,
                            RegisterSpace: ROOT_ADDRESS_SPACE,
                            Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                        },
                    },
                    ShaderVisibility: visibility,
                });
            }
            if t.stride_table {
                stage.stride_table = Some(params.len() as u32);
                params.push(D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                            ShaderRegister: 0,
                            RegisterSpace: STRIDE_SPACE,
                            Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                        },
                    },
                    ShaderVisibility: visibility,
                });
            }
            for (n, s) in t.static_samplers.iter().enumerate() {
                samplers.push(D3D12_STATIC_SAMPLER_DESC {
                    Filter: D3D12_FILTER(s.filter as i32),
                    AddressU: D3D12_TEXTURE_ADDRESS_MODE(s.address[0] as i32),
                    AddressV: D3D12_TEXTURE_ADDRESS_MODE(s.address[1] as i32),
                    AddressW: D3D12_TEXTURE_ADDRESS_MODE(s.address[2] as i32),
                    MipLODBias: s.mip_lod_bias_for_host(),
                    MaxAnisotropy: s.max_anisotropy,
                    ComparisonFunc: D3D12_COMPARISON_FUNC(s.comparison as i32),
                    BorderColor: D3D12_STATIC_BORDER_COLOR(s.border as i32),
                    MinLOD: s.min_lod,
                    MaxLOD: s.max_lod,
                    ShaderRegister: n as u32,
                    RegisterSpace: SAMPLER_SPACE,
                    ShaderVisibility: visibility,
                });
            }
            bound.push(stage);
        }
        let (root, root_blob) =
            match root_signature(&host.device, &params, &samplers, compute.is_some()) {
                Ok(root) => root,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host root signature refused");
                    return None;
                }
            };
        // The bytecode of the stage with this visibility (none if absent).
        let code = |visibility: D3D12_SHADER_VISIBILITY| {
            let stage = stages.iter().find(|s| s.1 == visibility);
            stage
                .and_then(|&(s, _, program)| s.translated_as(program))
                .map_or(D3D12_SHADER_BYTECODE::default(), |t| {
                    D3D12_SHADER_BYTECODE {
                        pShaderBytecode: t.host.dxil.as_ptr().cast(),
                        BytecodeLength: t.host.dxil.len(),
                    }
                })
        };
        if compute.is_some() {
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                // SAFETY: a borrowed pointer the description never releases.
                pRootSignature: unsafe { std::mem::transmute_copy(&root) },
                CS: code(D3D12_SHADER_VISIBILITY_ALL),
                ..Default::default()
            };
            let name = super::library::name(
                "compute",
                &root_blob,
                &[&code_bytes(&code(D3D12_SHADER_VISIBILITY_ALL))],
                "",
            );
            let made = super::library::pipeline(
                &name,
                // SAFETY: the description points at live data.
                |library, name| unsafe { library.LoadComputePipeline(name, &desc) },
                // SAFETY: as above.
                || unsafe { host.device.CreateComputePipelineState(&desc) },
            );
            let pso = match made {
                Ok(pso) => pso,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host pipeline refused");
                    host.log_messages();
                    return None;
                }
            };
            tracing::info!(?shaders, "host pipeline");
            crate::host::name(&pso, || format!("{shaders:?}"));
            return Some(HostPipeline {
                compute: true,
                mesh: false,
                pso,
                root,
                stages: bound,
                name: format!("{shaders:?}"),
                slots: 0,
                inputs: Vec::new(),
            });
        }
        let Some(g) = g else {
            return refuse("graphics state not kept (plain description)", shaders);
        };
        let state = g.describe();
        let (vs, ps) = (
            code(if mesh.is_some() {
                D3D12_SHADER_VISIBILITY_MESH
            } else {
                D3D12_SHADER_VISIBILITY_VERTEX
            }),
            code(D3D12_SHADER_VISIBILITY_PIXEL),
        );
        if mesh.is_some() {
            // "mesh triangles": the stream names its topology since it does not
            // load from the library without one (see `mesh_pipeline`); a
            // new name keeps older entries without topology from being
            // looked up again.
            let name = super::library::name(
                "mesh triangles",
                &root_blob,
                &[&code_bytes(&vs), &code_bytes(&ps)],
                &state,
            );
            let pso = match mesh_pipeline(&host.device, &root, vs, ps, g, &name) {
                Ok(pso) => pso,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host mesh pipeline refused");
                    host.log_messages();
                    return None;
                }
            };
            tracing::info!(?shaders, "host mesh pipeline");
            crate::host::name(&pso, || format!("{shaders:?}"));
            return Some(HostPipeline {
                compute: false,
                mesh: true,
                pso,
                root,
                stages: bound,
                name: format!("{shaders:?}"),
                slots: 0,
                inputs: Vec::new(),
            });
        }

        let gs = code(D3D12_SHADER_VISIBILITY_GEOMETRY);
        let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // SAFETY: a borrowed pointer the description never releases.
            pRootSignature: unsafe { std::mem::transmute_copy(&root) },
            VS: vs,
            PS: ps,
            GS: gs,
            BlendState: g.blend,
            SampleMask: g.sample_mask,
            RasterizerState: g.rasterizer,
            DepthStencilState: g.depth,
            InputLayout: D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: if g.inputs.is_empty() {
                    std::ptr::null()
                } else {
                    g.inputs.as_ptr()
                },
                NumElements: g.inputs.len() as u32,
            },
            IBStripCutValue: g.strip_cut,
            // The console's rectangle and quad types draw as triangles on
            // the host (see `draw`).
            // shortcut: a geometry stage is always given triangles (the
            // translated GS declares triangle input); a title drawing
            // another primitive through one would need the GS translated
            // for that primitive.
            PrimitiveTopologyType: match g.topology.0 {
                _ if geometry.is_some() => D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                RECTANGLES | QUADS => D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                _ => g.topology,
            },
            NumRenderTargets: g.targets.NumRenderTargets,
            RTVFormats: g.targets.RTFormats,
            DSVFormat: g.depth_format,
            SampleDesc: g.samples,
            ..Default::default()
        };
        // Pipelines without a geometry stage keep the names they had.
        let name = if geometry.is_some() {
            super::library::name(
                "geometry",
                &root_blob,
                &[&code_bytes(&vs), &code_bytes(&gs), &code_bytes(&ps)],
                &state,
            )
        } else {
            super::library::name(
                "graphics",
                &root_blob,
                &[&code_bytes(&vs), &code_bytes(&ps)],
                &state,
            )
        };
        let made = super::library::pipeline(
            &name,
            // SAFETY: the description points at live data.
            |library, name| unsafe { library.LoadGraphicsPipeline(name, &desc) },
            // SAFETY: as above.
            || unsafe { host.device.CreateGraphicsPipelineState(&desc) },
        );
        let pso = match made {
            Ok(pso) => pso,
            Err(e) => {
                tracing::warn!(error = %e, ?shaders, "host pipeline refused");
                host.log_messages();
                return None;
            }
        };
        tracing::info!(?shaders, "host pipeline");
        crate::host::name(&pso, || format!("{shaders:?}"));
        tracing::debug!(
            ?shaders,
            depth = ?g.depth,
            dsv = g.depth_format.0,
            targets = ?&g.targets.RTFormats[..g.targets.NumRenderTargets.min(8) as usize],
            blend = ?g.blend.RenderTarget[0],
            cull = g.rasterizer.CullMode.0,
            "host graphics state"
        );
        Some(HostPipeline {
            compute: false,
            mesh: false,
            pso,
            root,
            stages: bound,
            name: format!("{shaders:?}"),
            slots: g.slots(),
            inputs: g.elements(),
        })
    }
}

/// A mesh pipeline from the title's graphics state: the pipeline-state
/// stream form (no input layout or topology).
fn mesh_pipeline(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    ms: D3D12_SHADER_BYTECODE,
    ps: D3D12_SHADER_BYTECODE,
    g: &GraphicsState,
    name: &str,
) -> windows::core::Result<ID3D12PipelineState> {
    use windows::core::Interface;
    /// A stream subobject: its type, then its description, pointer-aligned.
    #[repr(C, align(8))]
    struct Sub<T>(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, T);
    #[repr(C)]
    struct Stream {
        root: Sub<*mut std::ffi::c_void>,
        ms: Sub<D3D12_SHADER_BYTECODE>,
        ps: Sub<D3D12_SHADER_BYTECODE>,
        blend: Sub<D3D12_BLEND_DESC>,
        mask: Sub<u32>,
        raster: Sub<D3D12_RASTERIZER_DESC>,
        depth: Sub<D3D12_DEPTH_STENCIL_DESC>,
        targets: Sub<D3D12_RT_FORMAT_ARRAY>,
        dsv: Sub<DXGI_FORMAT>,
        samples: Sub<DXGI_SAMPLE_DESC>,
        topology: Sub<D3D12_PRIMITIVE_TOPOLOGY_TYPE>,
    }
    let mut stream = Stream {
        root: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE,
            root.as_raw(),
        ),
        ms: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, ms),
        ps: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps),
        blend: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, g.blend),
        mask: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK,
            g.sample_mask,
        ),
        raster: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, g.rasterizer),
        depth: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, g.depth),
        targets: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS,
            g.targets,
        ),
        dsv: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT,
            g.depth_format,
        ),
        samples: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, g.samples),
        topology: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY,
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        ),
    };
    let desc = D3D12_PIPELINE_STATE_STREAM_DESC {
        SizeInBytes: size_of::<Stream>(),
        pPipelineStateSubobjectStream: (&raw mut stream).cast(),
    };
    let device = device.cast::<ID3D12Device2>()?;
    super::library::pipeline(
        name,
        // SAFETY: the stream holds whole subobjects pointing at live data.
        |library, name| unsafe { library.LoadPipeline(name, &desc) },
        // SAFETY: as above.
        || unsafe { device.CreatePipelineState(&desc) },
    )
}

/// The bytes of shader code (none for an absent stage).
fn code_bytes(code: &D3D12_SHADER_BYTECODE) -> Vec<u8> {
    if code.pShaderBytecode.is_null() {
        return Vec::new();
    }
    // SAFETY: a translation's own bytes, alive while its shader is.
    unsafe { std::slice::from_raw_parts(code.pShaderBytecode.cast::<u8>(), code.BytecodeLength) }
        .to_vec()
}

fn root_signature(
    device: &ID3D12Device,
    params: &[D3D12_ROOT_PARAMETER1],
    samplers: &[D3D12_STATIC_SAMPLER_DESC],
    compute: bool,
) -> windows::core::Result<(ID3D12RootSignature, Vec<u8>)> {
    let input = if compute {
        D3D12_ROOT_SIGNATURE_FLAG_NONE
    } else {
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
    };
    let heap_indexing = D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
        | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED;
    let desc_with = |flags| D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: windows::Win32::Graphics::Direct3D12::D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: samplers.len() as u32,
                pStaticSamplers: samplers.as_ptr(),
                Flags: flags,
            },
        },
    };
    let desc = desc_with(input | heap_indexing);
    // SAFETY: the description points at live arrays; the blob is read in
    // full before it is released.
    unsafe {
        let mut blob: Option<ID3DBlob> = None;
        let mut error: Option<ID3DBlob> = None;
        if let Err(e) = D3D12SerializeVersionedRootSignature(&desc, &mut blob, Some(&mut error)) {
            if let Some(ref error) = error {
                let text = std::slice::from_raw_parts(
                    error.GetBufferPointer().cast::<u8>(),
                    error.GetBufferSize(),
                );
                tracing::warn!(
                    "root signature: runtime rejected directly-indexed flags ({}), retrying without them",
                    String::from_utf8_lossy(text)
                );
            }
            drop(error.take());
            blob = None;
            let retry = desc_with(input);
            let mut retry_error: Option<ID3DBlob> = None;
            if let Err(e2) =
                D3D12SerializeVersionedRootSignature(&retry, &mut blob, Some(&mut retry_error))
            {
                if let Some(error) = retry_error {
                    let text = std::slice::from_raw_parts(
                        error.GetBufferPointer().cast::<u8>(),
                        error.GetBufferSize(),
                    );
                    tracing::warn!("root signature retry: {}", String::from_utf8_lossy(text));
                }
                return Err(e2);
            }
            let _ = e;
        }
        let blob = blob.ok_or_else(windows::core::Error::empty)?;
        let bytes =
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        let root = match device.CreateRootSignature(0, bytes) {
            Ok(root) => (root, bytes.to_vec()),
            Err(e) => {
                let retry = desc_with(input);
                let mut fallback_blob: Option<ID3DBlob> = None;
                if D3D12SerializeVersionedRootSignature(&retry, &mut fallback_blob, None).is_ok()
                    && let Some(fallback_blob) = fallback_blob
                {
                    let fallback_bytes = std::slice::from_raw_parts(
                        fallback_blob.GetBufferPointer().cast::<u8>(),
                        fallback_blob.GetBufferSize(),
                    );
                    let root = device.CreateRootSignature(0, fallback_bytes)?;
                    (root, fallback_bytes.to_vec())
                } else {
                    return Err(e);
                }
            }
        };
        Ok(root)
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::Ordering;

    use super::*;

    #[test]
    fn integer_target_bypasses_blending_and_preserves_mixed_target_masks() {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;

        let mut g = GraphicsState::default();
        g.targets.NumRenderTargets = 3;
        g.targets.RTFormats[..3].copy_from_slice(&[
            DXGI_FORMAT_R32_FLOAT,
            DXGI_FORMAT_B4G4R4A4_UNORM,
            DXGI_FORMAT_R8_UINT,
        ]);
        g.blend.IndependentBlendEnable = true.into();
        for (i, target) in g.blend.RenderTarget[..3].iter_mut().enumerate() {
            target.BlendEnable = true.into();
            target.BlendOp = D3D12_BLEND_OP_MAX;
            target.RenderTargetWriteMask = [15, 15, 1][i];
        }
        let before = g.blend;
        g.apply_integer_blend_bypass();
        assert_eq!(g.blend.RenderTarget[0..2], before.RenderTarget[0..2]);
        let mut expected = before.RenderTarget[2];
        expected.BlendEnable = false.into();
        assert_eq!(g.blend.RenderTarget[2], expected);

        g.blend = before;
        g.blend.IndependentBlendEnable = false.into();
        g.blend.RenderTarget[0].RenderTargetWriteMask = 3;
        g.targets.RTFormats.swap(0, 2);
        g.apply_integer_blend_bypass();
        assert!(g.blend.IndependentBlendEnable.as_bool());
        assert!(!g.blend.RenderTarget[0].BlendEnable.as_bool());
        for target in &g.blend.RenderTarget[1..3] {
            assert!(target.BlendEnable.as_bool());
            assert_eq!(target.BlendOp, D3D12_BLEND_OP_MAX);
            assert_eq!(target.RenderTargetWriteMask, 3);
        }
    }

    #[test]
    fn host_root_signature_serializes_and_creates() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let (root, bytes) = root_signature(&host.device, &[], &[], false).unwrap();
        assert!(!bytes.is_empty());
        let _ = root;
        let (comp_root, comp_bytes) = root_signature(&host.device, &[], &[], true).unwrap();
        assert!(!comp_bytes.is_empty());
        let _ = comp_root;
    }

    /// A triangle mesh shader of our own and a pixel shader for it.
    const MS: &str = "struct V { float4 p : SV_Position; };
        [outputtopology(\"triangle\")] [numthreads(1, 1, 1)]
        void main(out vertices V v[3], out indices uint3 t[1]) {
            SetMeshOutputCounts(3, 1);
            v[0].p = float4(0, 0, 0, 1);
            v[1].p = float4(1, 0, 0, 1);
            v[2].p = float4(0, 1, 0, 1);
            t[0] = uint3(0, 1, 2);
        }";
    const PS: &str = "float4 main() : SV_Target { return 1; }";

    /// The PC `dxc.exe` (`VICTORY_PC_DXC`, else the newest Windows SDK's).
    fn dxc() -> Option<PathBuf> {
        let dxc = match std::env::var_os("VICTORY_PC_DXC") {
            Some(p) => PathBuf::from(p),
            None => victory_shader::emit::validator::Validator::default_path()?
                .with_file_name("dxc.exe"),
        };
        dxc.exists().then_some(dxc)
    }

    fn compile(dxc: &Path, dir: &Path, src: &str, profile: &str) -> Vec<u8> {
        let (hlsl, out) = (
            dir.join(format!("{profile}.hlsl")),
            dir.join(format!("{profile}.dxil")),
        );
        std::fs::write(&hlsl, src).unwrap();
        let status = std::process::Command::new(dxc)
            .args(["-nologo", "-T", profile, "-E", "main", "-Fo"])
            .arg(&out)
            .arg(&hlsl)
            .status()
            .unwrap();
        assert!(status.success(), "dxc failed for {profile}");
        std::fs::read(out).unwrap()
    }

    fn code(bytes: &[u8]) -> D3D12_SHADER_BYTECODE {
        D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        }
    }

    /// A mesh pipeline made once is loaded from the library the next time,
    /// not compiled again (a stream without a topology never loads).
    #[test]
    fn a_mesh_pipeline_loads_from_the_library() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let Some(dxc) = dxc() else {
            eprintln!("no PC dxc.exe; skipped");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let ms = compile(&dxc, dir.path(), MS, "ms_6_5");
        let ps = compile(&dxc, dir.path(), PS, "ps_6_5");
        let (root, root_blob) = root_signature(&host.device, &[], &[], false).unwrap();
        let mut g = GraphicsState::default();
        g.targets.NumRenderTargets = 1;
        g.targets.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        let name = super::super::library::name("mesh test", &root_blob, &[&ms, &ps], &g.describe());
        mesh_pipeline(&host.device, &root, code(&ms), code(&ps), &g, &name).expect("made");
        let loaded = super::super::library::LOADED.load(Ordering::Relaxed);
        mesh_pipeline(&host.device, &root, code(&ms), code(&ps), &g, &name).expect("made again");
        assert!(
            super::super::library::LOADED.load(Ordering::Relaxed) > loaded,
            "the mesh pipeline was compiled again instead of loaded from the library"
        );
    }

    use super::{dxil_wave_lanes, wave_lanes_unsupported};

    fn synthetic_dxbc_with_psv0(min_lanes: u32, max_lanes: u32) -> Vec<u8> {
        let mut psv = Vec::new();
        psv.extend_from_slice(&48u32.to_le_bytes()); // PSVRuntimeInfo2 size
        let mut info = [0u8; 48];
        info[16..20].copy_from_slice(&min_lanes.to_le_bytes());
        info[20..24].copy_from_slice(&max_lanes.to_le_bytes());
        psv.extend_from_slice(&info);

        // DXBC header (32 bytes) + 1 part offset (4 bytes) + part header (8 bytes) + psv.
        let part_offset = 36u32;
        let total_size = part_offset + 8 + psv.len() as u32;
        let mut out = Vec::with_capacity(total_size as usize);
        out.extend_from_slice(b"DXBC");
        out.extend_from_slice(&[0u8; 16]); // digest
        out.extend_from_slice(&1u32.to_le_bytes()); // version
        out.extend_from_slice(&total_size.to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes()); // part count
        out.extend_from_slice(&part_offset.to_le_bytes());
        out.extend_from_slice(b"PSV0");
        out.extend_from_slice(&(psv.len() as u32).to_le_bytes());
        out.extend_from_slice(&psv);
        out
    }

    #[test]
    fn emitted_wave_size_checked_against_host_range() {
        // Compute declaring WaveSize(64) is refused on 32-lane NVIDIA and 4-lane WARP,
        // and allowed on 64-lane hosts.
        assert!(wave_lanes_unsupported(
            "compute",
            true,
            Some((64, 64)),
            (32, 32)
        ));
        assert!(wave_lanes_unsupported(
            "compute",
            true,
            Some((64, 64)),
            (4, 4)
        ));
        assert!(!wave_lanes_unsupported(
            "compute",
            true,
            Some((64, 64)),
            (32, 64)
        ));

        // Wave64 compute lowered to WaveSize(32) (e.g. split emulation) or per-lane
        // (no WaveSize) runs on 32-lane hosts; WaveSize(32) is still refused on 4-lane WARP.
        assert!(!wave_lanes_unsupported(
            "compute",
            true,
            Some((32, 32)),
            (32, 32)
        ));
        assert!(wave_lanes_unsupported(
            "compute",
            true,
            Some((32, 32)),
            (4, 4)
        ));
        assert!(!wave_lanes_unsupported("compute", true, None, (32, 32)));
        assert!(!wave_lanes_unsupported("compute", true, None, (4, 4)));

        // Console wave64 mesh has no PSV0 WaveSize metadata and still needs 64-lane hosts.
        assert!(wave_lanes_unsupported("mesh", true, None, (32, 32)));
        assert!(wave_lanes_unsupported("mesh", true, None, (4, 4)));
        assert!(!wave_lanes_unsupported("mesh", true, None, (32, 64)));
        assert!(!wave_lanes_unsupported("mesh", false, None, (32, 32)));

        // Vertex and pixel programs run per lane.
        assert!(!wave_lanes_unsupported("vertex", true, None, (32, 32)));
        assert!(!wave_lanes_unsupported("pixel", true, None, (4, 4)));
    }

    #[test]
    fn dxil_wave_lanes_reads_psv0_bounds() {
        assert_eq!(
            dxil_wave_lanes(&synthetic_dxbc_with_psv0(0, u32::MAX)),
            None
        );
        assert_eq!(
            dxil_wave_lanes(&synthetic_dxbc_with_psv0(64, 64)),
            Some((64, 64))
        );
        assert_eq!(
            dxil_wave_lanes(&synthetic_dxbc_with_psv0(32, 32)),
            Some((32, 32))
        );
    }

    /// Immediate console samplers can carry signed 14-bit 6.8 mip LOD biases
    /// outside D3D12's `[-16.0, 15.99]` static-sampler range; clamping lets
    /// `D3D12SerializeVersionedRootSignature` accept them.
    #[test]
    fn out_of_range_static_sampler_mip_lod_bias_serializes_for_host() {
        use victory_shader::emit::StaticSampler;

        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let raw_samplers = [
            StaticSampler::from_words([0x3088, 0x00FF_F000, 0x1400_2800, 0x8000_0000]), // -24.0
            StaticSampler::from_words([0x3088, 0x00FF_F000, 0x1400_2000, 0x8000_0000]), // -32.0
            StaticSampler::from_words([0x3088, 0x00FF_F000, 0x1400_1800, 0x8000_0000]), // +24.0
            StaticSampler::from_words([0x3088, 0x00FF_F000, 0x1400_1FFF, 0x8000_0000]), // +31.996
        ];
        let clamped: Vec<D3D12_STATIC_SAMPLER_DESC> = raw_samplers
            .iter()
            .enumerate()
            .map(|(n, s)| {
                let bias = s.mip_lod_bias_for_host();
                assert!(
                    (-16.0..=15.99).contains(&bias),
                    "clamped bias {bias} outside D3D12 bounds"
                );
                D3D12_STATIC_SAMPLER_DESC {
                    Filter: D3D12_FILTER(s.filter as i32),
                    AddressU: D3D12_TEXTURE_ADDRESS_MODE(s.address[0] as i32),
                    AddressV: D3D12_TEXTURE_ADDRESS_MODE(s.address[1] as i32),
                    AddressW: D3D12_TEXTURE_ADDRESS_MODE(s.address[2] as i32),
                    MipLODBias: bias,
                    MaxAnisotropy: s.max_anisotropy,
                    ComparisonFunc: D3D12_COMPARISON_FUNC(s.comparison as i32),
                    BorderColor: D3D12_STATIC_BORDER_COLOR(s.border as i32),
                    MinLOD: s.min_lod,
                    MaxLOD: s.max_lod,
                    ShaderRegister: n as u32,
                    RegisterSpace: SAMPLER_SPACE,
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                }
            })
            .collect();
        root_signature(&host.device, &[], &clamped, true)
            .expect("clamped static samplers serialize");

        // Confirm D3D12 itself rejects the unclamped -24.0 value.
        let mut unclamped = clamped[0];
        unclamped.MipLODBias = raw_samplers[0].mip_lod_bias;
        assert!(root_signature(&host.device, &[], &[unclamped], true).is_err());
    }
}
