// Banc synthétique : coût GPU d'une frame de draws « GS émulé » (mesh pipeline), par variante de mapping.
// usage: bench <workload A|B|AB> <mode 0..5> <payloadWords> <pack 0|1> [frames] [sommets par warp]
// modes 4 et 5 : routage par cascade en pipeline vertex (4 = toutes cascades instanciées, inexact ; 5 = cascades 0-2
// instanciées + cascade 3 non indexée avec la troncature, exact)
import Metal
import Foundation

let a = CommandLine.arguments
let workload = a[1], mode = Int(a[2])!, payloadWords = Int(a[3])!, pack = Int(a[4])!
let frames = a.count > 5 ? Int(a[5])! : 12
let warpV = a.count > 6 ? Int(a[6])! : 30   // expérience : sommets par warp (30 = DXMT)
let noRaster = ProcessInfo.processInfo.environment["NORASTER"] != nil   // coût de la géométrie seule
let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
let src = try! String(contentsOfFile: (a[0] as NSString).deletingLastPathComponent + "/gs.metal")

var pso3A: MTLRenderPipelineState! = nil, pso3B: MTLRenderPipelineState! = nil   // cascade 3 avec troncature (mode 5)
func pipeline(nreg: Int) -> MTLRenderPipelineState {
  let o = MTLCompileOptions()
  let payloadWords = payloadWords == 0 ? 4 + warpV * nreg * 4 + 4 : payloadWords   // 0 = au plus juste
  o.preprocessorMacros = ["NCASC": NSNumber(value: mode == 5 ? 3 : 4), "PAYLOAD_WORDS": NSNumber(value: payloadWords), "MESH_MODE": NSNumber(value: mode),
                          "PACK": NSNumber(value: pack), "NREG": NSNumber(value: nreg), "WARP_V": NSNumber(value: warpV)]
  o.languageVersion = .version3_2
  let lib = try! dev.makeLibrary(source: src, options: o)
  if mode >= 4 {
    let d = MTLRenderPipelineDescriptor()
    d.vertexFunction = lib.makeFunction(name: "vroute"); d.fragmentFunction = lib.makeFunction(name: "fragL")
    d.depthAttachmentPixelFormat = .depth32Float; d.inputPrimitiveTopology = .triangle
    let p = try! dev.makeRenderPipelineState(descriptor: d)
    d.vertexFunction = lib.makeFunction(name: "vroute3")
    let p3 = try! dev.makeRenderPipelineState(descriptor: d)
    if nreg == 1 { pso3A = p3 } else { pso3B = p3 }
    return p
  }
  let d = MTLMeshRenderPipelineDescriptor()
  d.objectFunction = lib.makeFunction(name: "obj")
  d.meshFunction = lib.makeFunction(name: "meshfn")
  d.fragmentFunction = lib.makeFunction(name: "frag")
  d.payloadMemoryLength = payloadWords * 4
  d.maxTotalThreadsPerObjectThreadgroup = warpV
  d.maxTotalThreadsPerMeshThreadgroup = mode >= 2 ? warpV / 3 : 1
  d.depthAttachmentPixelFormat = .depth32Float
  d.isRasterizationEnabled = !noRaster
  return try! dev.makeRenderPipelineState(descriptor: d, options: []).0
}

// Cible : atlas d'ombres 4 couches
let td = MTLTextureDescriptor()
td.textureType = .type2DArray; td.pixelFormat = .depth32Float; let rt = Int(ProcessInfo.processInfo.environment["RT"] ?? "4096")!   // RT=64 : remplissage négligeable, reste la géométrie
td.width = rt; td.height = rt
td.arrayLength = 4; td.usage = .renderTarget; td.storageMode = .private
let depth = dev.makeTexture(descriptor: td)!
let dsd = MTLDepthStencilDescriptor(); dsd.depthCompareFunction = .less; dsd.isDepthWriteEnabled = true
let dss = dev.makeDepthStencilState(descriptor: dsd)!

var rng = SystemRandomNumberGenerator()
func rnd() -> Float { Float.random(in: 0..<1, using: &rng) }

// Géométrie A : 1450 draws x 498 triangles indexés (petits triangles répartis sur [-1000,1000]^2)
let drawsA = 1450, trisA = 498
var vbA = [Float](), ibA = [UInt16]()
for _ in 0..<(trisA * 3 / 2) { vbA += [rnd() * 40 - 20, rnd() * 40 - 20, rnd() * 10] }   // ~0,5 sommet partagé par index
for _ in 0..<(trisA * 3) { ibA.append(UInt16(Int(rnd() * Float(trisA * 3 / 2)))) }
// Géométrie B : 1 quad, 203 draws x 5759 instances
let drawsB = 203, instB = 5759
let vbB: [Float] = [-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]
let ibB: [UInt16] = [0, 1, 2, 0, 2, 3]
var inst = [Float]()
for _ in 0..<(drawsB * instB + drawsA) { inst += [rnd() * 2000 - 1000, rnd() * 2000 - 1000, rnd() * 50, 1.0] }
// Cascades imbriquées : demi-côté 1000, 500, 250, 125
var casc = [Int32]()
for h in [Int32(1000), 500, 250, 125] { casc += [-h, -h, 0, h, h, 100, 0, 0] }
var uni: [Float] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 7095]
struct DrawArgs { var indexCount, instanceCount, startIndex: UInt32; var baseVertex: Int32; var startInstance: UInt32 }
var args = [DrawArgs]()
for i in 0..<drawsA { args.append(DrawArgs(indexCount: UInt32(trisA * 3), instanceCount: 1, startIndex: 0, baseVertex: 0, startInstance: UInt32(drawsB * instB + i))) }
for i in 0..<drawsB { args.append(DrawArgs(indexCount: 6, instanceCount: UInt32(instB), startIndex: 0, baseVertex: 0, startInstance: UInt32(i * instB))) }

func buf<T>(_ x: [T]) -> MTLBuffer { x.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! } }
let bVA = buf(vbA), bIA = buf(ibA), bVB = buf(vbB), bIB = buf(ibB), bInst = buf(inst), bCasc = buf(casc), bUni = buf(uni), bArgs = buf(args)
let psoA = pipeline(nreg: 1), psoB = pipeline(nreg: 2)
let meshThreads = MTLSize(width: mode >= 2 ? warpV / 3 : 1, height: 1, depth: 1)
let stride = MemoryLayout<DrawArgs>.stride

func frame() -> Double {
  let cb = q.makeCommandBuffer()!
  let rp = MTLRenderPassDescriptor()
  rp.depthAttachment.texture = depth; rp.depthAttachment.loadAction = .clear; rp.depthAttachment.storeAction = .store
  rp.depthAttachment.clearDepth = 1; rp.renderTargetArrayLength = 4
  let e = cb.makeRenderCommandEncoder(descriptor: rp)!
  e.setDepthStencilState(dss)
  e.setObjectBuffer(bInst, offset: 0, index: 3); e.setObjectBuffer(bUni, offset: 0, index: 4)
  e.setMeshBuffer(bCasc, offset: 0, index: 0); e.setMeshBuffer(bUni, offset: 0, index: 1)
  e.setObjectBuffer(bArgs, offset: 0, index: 2)
  if workload.contains("A") {
    e.setRenderPipelineState(psoA)
    e.setObjectBuffer(bIA, offset: 0, index: 0); e.setObjectBuffer(bVA, offset: 0, index: 1)
    if mode >= 4 {
      let nc = mode == 5 ? 3 : 4
      for b in [(bIA, 0), (bVA, 1), (bInst, 3), (bUni, 4), (bCasc, 5)] { e.setVertexBuffer(b.0, offset: 0, index: b.1) }
      for pass in 0..<(mode == 5 ? 2 : 1) {
        e.setRenderPipelineState(pass == 0 ? psoA : pso3A)
        for i in 0..<drawsA {
          e.setVertexBuffer(bArgs, offset: i * stride, index: 2)
          if pass == 0 { e.drawIndexedPrimitives(type: .triangle, indexCount: trisA * 3, indexType: .uint16, indexBuffer: bIA, indexBufferOffset: 0, instanceCount: nc) }
          else { e.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: trisA * 3, instanceCount: 1) }
        }
      }
    } else {
      for i in 0..<drawsA {
        e.setObjectBufferOffset(i * stride, index: 2)
        e.drawMeshThreadgroups(MTLSize(width: (trisA * 3 - 1) / warpV + 1, height: 1, depth: 1),
                               threadsPerObjectThreadgroup: MTLSize(width: warpV, height: 1, depth: 1), threadsPerMeshThreadgroup: meshThreads)
      }
    }
  }
  if workload.contains("B") {
    e.setRenderPipelineState(psoB)
    e.setObjectBuffer(bIB, offset: 0, index: 0); e.setObjectBuffer(bVB, offset: 0, index: 1)
    if mode >= 4 {
      let nc = mode == 5 ? 3 : 4
      for b in [(bIB, 0), (bVB, 1), (bInst, 3), (bUni, 4), (bCasc, 5)] { e.setVertexBuffer(b.0, offset: 0, index: b.1) }
      for pass in 0..<(mode == 5 ? 2 : 1) {
        e.setRenderPipelineState(pass == 0 ? psoB : pso3B)
        for i in 0..<drawsB {
          e.setVertexBuffer(bArgs, offset: (drawsA + i) * stride, index: 2)
          if pass == 0 { e.drawIndexedPrimitives(type: .triangle, indexCount: 6, indexType: .uint16, indexBuffer: bIB, indexBufferOffset: 0, instanceCount: instB * nc) }
          else { e.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6, instanceCount: instB) }
        }
      }
    } else {
      for i in 0..<drawsB {
        e.setObjectBufferOffset((drawsA + i) * stride, index: 2)
        let k = warpV / 6
        let groups = pack == 1 ? (instB + k - 1) / k : instB
        e.drawMeshThreadgroups(MTLSize(width: 1, height: groups, depth: 1),
                               threadsPerObjectThreadgroup: MTLSize(width: warpV, height: 1, depth: 1), threadsPerMeshThreadgroup: meshThreads)
      }
    }
  }
  e.endEncoding()
  cb.commit(); cb.waitUntilCompleted()
  if let err = cb.error { print("ERR", err) }
  return (cb.gpuEndTime - cb.gpuStartTime) * 1000
}
var t = (0..<frames).map { _ in frame() }
t.removeFirst(2); t.sort()
print(String(format: "%@ mode=%d payload=%d pack=%d  GPU médiane %.2f ms  (min %.2f, max %.2f)", workload, mode, payloadWords * 4, pack, t[t.count / 2], t.first!, t.last!))
