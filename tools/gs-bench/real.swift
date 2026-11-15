// Banc « vrais shaders » : exécute les shaders object (VS) + mesh (GS) de WARNO convertis par airconv,
// avec les conventions de liaison de DXMT, dans une passe profondeur 4 couches. Sort le temps GPU médian
// et une empreinte du tampon de profondeur (pour comparer l'ancien et le nouveau convertisseur au bit près).
// usage: real <libdir> <A|B> <legacy|new> [frames] [draws] [dump.raw]
import Metal
import Foundation

let a = CommandLine.arguments
let libdir = a[1], scen = a[2], isNew = a[3].hasPrefix("new")
let phase1 = a[3] == "new1"   // layout du commit 58b45832 (warps de 30, payload sans mots de validité), pour comparer
let frames = a.count > 4 ? Int(a[4])! : 12
let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!

let nreg = scen == "A" ? 1 : 2; let shaders = scen == "C" ? "B" : scen                                // registres de sortie du VS
let strip = scen == "S"                                       // S : shaders de A, topologie triangle strip
// miroir de airconv_public.h : SM50GSWarpScale, SM50GSPayloadSize, SM50GSMeshThreadsLimit, et de GetGeometryDrawLayout
let scale = isNew && !phase1 && !strip && 16 + 64 * (4 + 16 * nreg) <= 16256 ? 2 : 1
let payload = !isNew ? 16256 : phase1 ? 16 + 32 * 16 * nreg : min(16 + 32 * scale * (4 + 16 * nreg), 16256)
let warpInc = 30 * scale                                      // sommets consommés par warp
let objThreads = strip ? 32 : warpInc
let meshThreads = isNew ? min(256 / (9 + 1), strip ? 30 : warpInc / 3) : 1

func fn(_ name: String) -> MTLFunction {
  try! dev.makeLibrary(URL: URL(fileURLWithPath: "\(libdir)/\(shaders)_\(name).metallib")).makeFunction(name: "shader_main")!
}
let pd = MTLMeshRenderPipelineDescriptor()
pd.objectFunction = fn("obj"); pd.meshFunction = fn("mesh")
pd.payloadMemoryLength = payload
pd.depthAttachmentPixelFormat = .depth32Float
if ProcessInfo.processInfo.environment["NORASTER"] != nil { pd.isRasterizationEnabled = false }   // coût géométrie seul
let pso = try! dev.makeRenderPipelineState(descriptor: pd, options: []).0

let side = 2048
let td = MTLTextureDescriptor()
td.textureType = .type2DArray; td.pixelFormat = .depth32Float; td.width = side; td.height = side
td.arrayLength = 4; td.usage = .renderTarget; td.storageMode = .private
let depth = dev.makeTexture(descriptor: td)!
let dsd = MTLDepthStencilDescriptor(); dsd.depthCompareFunction = .less; dsd.isDepthWriteEnabled = true
let dss = dev.makeDepthStencilState(descriptor: dsd)!

var seed: UInt64 = 0x9E3779B97F4A7C15   // déterministe : mêmes données pour toutes les variantes
func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(seed >> 40) / Float(1 << 24) }
func buf<T>(_ x: [T]) -> MTLBuffer { x.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! } }

struct DrawArgs { var indexCount, instanceCount, startIndex: UInt32; var baseVertex: Int32; var startInstance: UInt32; var pad: (UInt32, UInt32, UInt32) = (0, 0, 0) }
struct VBEntry { var addr: UInt64; var stride: UInt32; var length: UInt32 }

// constantes : cb[0..2] = identité (VS A : monde ; GS A : lumière ; VS B : droite/haut), cb[3..5] = lumière du GS B (centre en 1000,1000)
let cb = buf([Float]([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, -1000, 0, 1, 0, -1000, 0, 0, 1, 0]))
// cascades imbriquées : (xmin, ymin, zmin + 7095, xmax, ymax, zmax - 1290, -, -)
var casc = [Int32]()
for h in [Int32(1000), 500, 250] { casc += [-h, -h, 7095, h, h, 100 - 1290, 0, 0] }
casc += [300, 300, 7095, 900, 900, 100 - 1290, 0, 0]   // la 4e cascade ne s'emboîte pas : couche 3 couverte sans dépasser maxvertexcount
let bCasc = buf(casc)
let tbd = MTLTextureDescriptor.textureBufferDescriptor(with: .r32Sint, width: casc.count, resourceOptions: .storageModeShared, usage: .shaderRead)
let tCasc = bCasc.makeTexture(descriptor: tbd, offset: 0, bytesPerRow: casc.count * 4)!
let meshTable = buf([UInt64]([tCasc.gpuResourceID._impl, UInt64(casc.count) << 32]))
let cbTable = buf([UInt64]([cb.gpuAddress]))

var args = [DrawArgs](), resources = [MTLResource](), objTable: MTLBuffer? = nil
var vbTable: MTLBuffer, ib: MTLBuffer
let draws: Int
if scen == "A" || strip {
  // 1450 draws x 498 petits triangles indexés 16 bits, sommets propres à chaque draw (BaseVertex)
  draws = a.count > 5 ? Int(a[5])! : 1450
  let tris = 498, vpd = tris * 3 / 2
  var vb = [Float](), idx = [UInt16]()
  for _ in 0..<draws {
    let cx = rnd() * 1900 - 950, cy = rnd() * 1900 - 950
    for _ in 0..<vpd { vb += [cx + rnd() * 40 - 20, cy + rnd() * 40 - 20, rnd() * 50] }
  }
  // strip : un index de coupure (0xFFFF) de temps en temps, pour le chemin « sommet invalide »
  for i in 0..<(tris * 3) { idx.append(strip && i % 53 == 52 ? 0xFFFF : UInt16(rnd() * Float(vpd))) }
  // strip : longueurs variées, pour les warps partiels
  for d in 0..<draws { args.append(DrawArgs(indexCount: UInt32(strip ? 3 + (d * 37) % 900 : tris * 3), instanceCount: 1, startIndex: 0, baseVertex: Int32(d * vpd), startInstance: 0)) }
  let bv = buf(vb); ib = buf(idx)
  vbTable = buf([VBEntry(addr: bv.gpuAddress, stride: 12, length: UInt32(bv.length))])
  resources = [bv]
} else {
  // B : 203 draws x 5759 instances d'un quad (billboards) : reg0 par sommet, reg1/reg2 par instance
  // C : mêmes shaders, nombres d'indices et d'instances variés (paquets d'instances, warps partiels)
  draws = a.count > 5 ? Int(a[5])! : (scen == "B" ? 203 : 600)
  let quad = buf([Float]([-1, -1, 1, -1, 1, 1, -1, 1]))
  var idx: [UInt16] = [0, 1, 2, 0, 2, 3], total = 0
  if scen == "B" {
    for d in 0..<draws { args.append(DrawArgs(indexCount: 6, instanceCount: 5759, startIndex: 0, baseVertex: 0, startInstance: UInt32(d * 5759))); total += 5759 }
  } else {
    for _ in 0..<60 { idx.append(UInt16(rnd() * 4)) }
    let counts: [UInt32] = [3, 6, 9, 12, 15, 18, 21, 30, 33, 7, 1, 60]
    for d in 0..<draws {
      let n = 1 + Int(rnd() * 300)
      args.append(DrawArgs(indexCount: counts[d % counts.count], instanceCount: UInt32(n), startIndex: UInt32(rnd() * 6), baseVertex: 0, startInstance: UInt32(total))); total += n
    }
  }
  let big: Float = scen == "B" ? 1 : 3
  var id = [Float]()
  for _ in 0..<total { id += [rnd() * 2000, rnd() * 2000, rnd() * 50, (0.5 + rnd() * 1.5) * big, 0.125, 0, rnd(), rnd()] }
  let bi = buf(id); ib = buf(idx)
  vbTable = buf([VBEntry(addr: quad.gpuAddress, stride: 8, length: 32), VBEntry(addr: bi.gpuAddress, stride: 32, length: UInt32(bi.length))])
  // table de liaison du VS : s0, cube_s0, meta_s0 (biais lod), t0, meta_t0
  let sd = MTLSamplerDescriptor(); sd.supportArgumentBuffers = true
  let smp = dev.makeSamplerState(descriptor: sd)!
  let t2 = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: 1, height: 1, mipmapped: false)
  let tex = dev.makeTexture(descriptor: t2)!
  var px: [Float] = [0.1, 0.2, 0.3, 0.4]
  tex.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &px, bytesPerRow: 16)
  objTable = buf([UInt64]([smp.gpuResourceID._impl, smp.gpuResourceID._impl, 0, tex.gpuResourceID._impl, 0]))
  resources = [quad, bi, tex]
}
let bArgs = buf(args)

func frame(readback: MTLBuffer?) -> Double {
  let cmd = q.makeCommandBuffer()!
  let rp = MTLRenderPassDescriptor()
  rp.depthAttachment.texture = depth; rp.depthAttachment.loadAction = .clear; rp.depthAttachment.storeAction = .store
  rp.depthAttachment.clearDepth = 1; rp.renderTargetArrayLength = 4
  let e = cmd.makeRenderCommandEncoder(descriptor: rp)!
  e.setDepthStencilState(dss); e.setRenderPipelineState(pso)
  for r in resources + [cb, tCasc, bCasc] as [MTLResource] { e.useResource(r, usage: .read, stages: [.object, .mesh]) }
  e.setObjectBuffer(cbTable, offset: 0, index: 0)
  if let t = objTable { e.setObjectBuffer(t, offset: 0, index: 1) }
  e.setObjectBuffer(vbTable, offset: 0, index: 2)
  e.setObjectBuffer(ib, offset: 0, index: 3)
  e.setObjectBuffer(bArgs, offset: 0, index: 4)
  e.setMeshBuffer(cbTable, offset: 0, index: 0)
  e.setMeshBuffer(meshTable, offset: 0, index: 1)
  for (i, d) in args.enumerated() {
    e.setObjectBufferOffset(i * MemoryLayout<DrawArgs>.stride, index: 4)
    let count = Int(d.indexCount), inst = Int(d.instanceCount)
    // comme GeometryDrawIndexed : SM50GSInstancesPerWarp
    let k = isNew && !strip && count <= objThreads / 2 && count % 3 == 0 ? objThreads / count : 1   // SM50GSInstancesPerWarp
    e.drawMeshThreadgroups(MTLSize(width: (count - 1) / warpInc + 1, height: (inst - 1) / k + 1, depth: 1),
                           threadsPerObjectThreadgroup: MTLSize(width: objThreads, height: 1, depth: 1),
                           threadsPerMeshThreadgroup: MTLSize(width: meshThreads, height: 1, depth: 1))
  }
  e.endEncoding()
  if let rb = readback {
    let b = cmd.makeBlitCommandEncoder()!
    for l in 0..<4 {
      b.copy(from: depth, sourceSlice: l, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0), sourceSize: MTLSize(width: side, height: side, depth: 1),
             to: rb, destinationOffset: l * side * side * 4, destinationBytesPerRow: side * 4, destinationBytesPerImage: side * side * 4)
    }
    b.endEncoding()
  }
  cmd.commit(); cmd.waitUntilCompleted()
  if let err = cmd.error { print("ERR", err.localizedDescription) }
  return (cmd.gpuEndTime - cmd.gpuStartTime) * 1000
}
var t = (0..<frames).map { _ in frame(readback: nil) }
t.removeFirst(min(2, frames - 1)); t.sort()
let rb = dev.makeBuffer(length: 4 * side * side * 4, options: .storageModeShared)!
_ = frame(readback: rb)
let px = rb.contents().bindMemory(to: UInt32.self, capacity: 4 * side * side)
var hash: UInt64 = 0xcbf29ce484222325, covered = [0, 0, 0, 0]
for i in 0..<(4 * side * side) { hash = (hash ^ UInt64(px[i])) &* 0x100000001b3; if px[i] != 0x3f800000 { covered[i / (side * side)] += 1 } }
if a.count > 6 { try! Data(bytes: rb.contents(), count: rb.length).write(to: URL(fileURLWithPath: a[6])) }
print(String(format: "%@ %@ %@  GPU médiane %.2f ms (min %.2f max %.2f)  profondeur %016llx  pixels couverts par couche %@",
             scen, (libdir as NSString).lastPathComponent, a[3], t[t.count / 2], t.first!, t.last!, hash, covered.description))
