// Reproduces, in native Metal, the GPU watchdog reset that WARNO's irradiance probe update triggers under DXMT.
//
// The game convolves its environment cube map with a pixel shader that takes 1000 x 250 cube map samples per pixel,
// one draw per 16x16 cube face. One invocation of that loop runs for more than ~50ms and the command buffer fails
// with "Caused GPU Hang Error (kIOGPUCommandBufferCallbackErrorHang)".
//
// build: swiftc -O irr.swift -o irr
// usage: irr [phi_iterations=1000] [faces=6] [passes_per_face=1] [target_size=16] [no_texture=0]
//
// Measured on a MacBook Pro M5 Max, macOS 27.2.1 (GPU time of the command buffer):
//   irr 700  1          completed, 42 ms          irr 1000 1 1 1     GPU Hang at 57 ms (1x1 target)
//   irr 800  1          completed, 42 ms          irr 1000 1 16      GPU Hang (16 one-row passes; the 2nd one faults)
//   irr 900  1          GPU Hang at 57 ms         irr 250  1 1 64    completed, 45 ms (4 tiles, 62500 samples/pixel)
//   irr 1000 1          GPU Hang at 57 ms         irr 3000 1 1 16 1  completed, 43 ms (no texture sample in the loop)
// The time of a pass does not depend on the number of pixels: what matters is how long one invocation runs.
import Metal
import Foundation

let args = CommandLine.arguments
let nphi = args.count > 1 ? Int(args[1])! : 1000
let faces = args.count > 2 ? Int(args[2])! : 6
let split = args.count > 3 ? Int(args[3])! : 1      // split each face into N passes (scissor strips)
let size = args.count > 4 ? Int(args[4])! : 16
let cheap = args.count > 5 ? Int(args[5])! : 0     // 1 = loop body without the texture sample
let dev = MTLCreateSystemDefaultDevice()!
let src = """
#include <metal_stdlib>
using namespace metal;
struct VOut { float4 pos [[position]]; float2 uv; };
vertex VOut vs(uint vid [[vertex_id]]) {
  VOut o; float2 p = float2((vid << 1) & 2, vid & 2);
  o.pos = float4(p * 2 - 1, 0, 1); o.uv = p; return o;
}
struct U { float delta; float phiMax; float cheap; float pad; };
fragment float4 fs(VOut in [[stage_in]], texturecube<float> t0 [[texture(0)]], sampler s0 [[sampler(0)]], constant U &u [[buffer(0)]]) {
  float3 N = normalize(float3(in.uv * 2 - 1, 1));
  float3 up = abs(N.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
  float3 right = normalize(cross(up, N)); up = cross(N, right);
  float3 irr = 0; float n = 0;
  for (float phi = 0; phi < u.phiMax; phi += u.delta) {
    float cp = cos(phi), sp = sin(phi);
    for (float theta = 0; theta < 1.5707963705; theta += u.delta) {
      float st = sin(theta), ct = cos(theta);
      float3 v = (st * cp) * right + (st * sp) * up + ct * N;
      irr += (u.cheap > 0.5 ? v : t0.sample(s0, v).rgb) * ct * st;
      n += 1;
    }
  }
  return float4(3.14159265 * irr / n, 1);
}
"""
let lib = try! dev.makeLibrary(source: src, options: nil)
let pd = MTLRenderPipelineDescriptor()
pd.vertexFunction = lib.makeFunction(name: "vs"); pd.fragmentFunction = lib.makeFunction(name: "fs")
pd.colorAttachments[0].pixelFormat = .rgba16Float
pd.inputPrimitiveTopology = .triangle
let pso = try! dev.makeRenderPipelineState(descriptor: pd)

let sd = MTLTextureDescriptor.textureCubeDescriptor(pixelFormat: .rgba16Float, size: 512, mipmapped: true)
sd.usage = [.shaderRead, .renderTarget]; sd.storageMode = .private
let srcTex = dev.makeTexture(descriptor: sd)!
let dd = MTLTextureDescriptor.textureCubeDescriptor(pixelFormat: .rgba16Float, size: size, mipmapped: false)
dd.usage = [.shaderRead, .renderTarget]; dd.storageMode = .private
let dst = dev.makeTexture(descriptor: dd)!
let smp = MTLSamplerDescriptor(); smp.minFilter = .linear; smp.magFilter = .linear; smp.mipFilter = .linear
let sampler = dev.makeSamplerState(descriptor: smp)!
var u: [Float] = [0.0062831854, Float(nphi) * 0.0062831854, Float(cheap), 0]
let ubuf = dev.makeBuffer(bytes: &u, length: 16, options: [])!
let q = dev.makeCommandQueue()!

func run(_ label: String, _ body: (MTLCommandBuffer) -> Void) {
  let d = MTLCommandBufferDescriptor(); d.errorOptions = .encoderExecutionStatus
  let cb = q.makeCommandBuffer(descriptor: d)!
  body(cb); cb.commit(); cb.waitUntilCompleted()
  let ms = (cb.gpuEndTime - cb.gpuStartTime) * 1000
  var extra = ""
  if let e = cb.error as NSError? {
    extra = "ERROR: " + e.localizedDescription
    if let infos = e.userInfo[MTLCommandBufferEncoderInfoErrorKey] as? [MTLCommandBufferEncoderInfo] {
      extra += " | " + infos.map { "\($0.label):\($0.errorState.rawValue)" }.joined(separator: " ")
    }
  }
  print(String(format: "%@: status=%d gpu=%.1f ms", label, cb.status.rawValue, ms), extra)
}
// fill the source cube map (clear + mips)
run("init") { cb in
  for f in 0..<6 {
    let rp = MTLRenderPassDescriptor(); rp.colorAttachments[0].texture = srcTex; rp.colorAttachments[0].slice = f
    rp.colorAttachments[0].loadAction = .clear; rp.colorAttachments[0].clearColor = MTLClearColor(red: 0.2 * Double(f), green: 0.5, blue: 0.1, alpha: 1)
    rp.colorAttachments[0].storeAction = .store
    cb.makeRenderCommandEncoder(descriptor: rp)!.endEncoding()
  }
  let b = cb.makeBlitCommandEncoder()!; b.generateMipmaps(for: srcTex); b.endEncoding()
}
run("irradiance phi=\(nphi) faces=\(faces) split=\(split) size=\(size)") { cb in
  for f in 0..<faces {
    let view = dst.makeTextureView(pixelFormat: .rgba16Float, textureType: .type2DArray, levels: 0..<1, slices: f..<6)!
    for s in 0..<split {
      let rp = MTLRenderPassDescriptor(); rp.colorAttachments[0].texture = view
      rp.colorAttachments[0].loadAction = .load; rp.colorAttachments[0].storeAction = .store
      rp.renderTargetArrayLength = 6 - f
      let e = cb.makeRenderCommandEncoder(descriptor: rp)!
      e.label = "irr face \(f) part \(s)"
      e.setRenderPipelineState(pso); e.setFragmentTexture(srcTex, index: 0); e.setFragmentSamplerState(sampler, index: 0)
      e.setFragmentBuffer(ubuf, offset: 0, index: 0)
      let h = size / split
      e.setScissorRect(MTLScissorRect(x: 0, y: s * h, width: size, height: s == split - 1 ? size - s * h : h))
      e.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
      e.endEncoding()
    }
  }
}
