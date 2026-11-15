// Modèle synthétique de l'émulation GS de DXMT (object = VS, mesh = GS « cascade d'ombres » de WARNO).
// Macros : PAYLOAD_WORDS, MESH_MODE (0 base, 1 grille = primitives valides, 2 un TG + atomiques, 3 un TG + préfixe 2 passes),
//          PACK (1 = plusieurs instances par object threadgroup quand IndexCount <= 15), NREG (1 ou 2)
#include <metal_stdlib>
using namespace metal;

#ifndef WARP_V
#define WARP_V 30
#endif
#define WARP_P (WARP_V / 3)
#define GS_MAXV 9

struct Payload { uint hdr[4]; float4 v[WARP_V * NREG]; uint pad[PAYLOAD_WORDS - 4 - WARP_V * NREG * 4]; };
struct DrawArgs { uint indexCount; uint instanceCount; uint startIndex; int baseVertex; uint startInstance; };
struct Uni { float4 row[6]; };

struct VOut { float4 pos [[position]];
#if NREG == 2
  float2 uv;
#endif
};
struct POut { uint layer [[render_target_array_index]]; };

#if MESH_MODE >= 2
#define MAXV (WARP_P * GS_MAXV)
#define MAXP (WARP_P * (GS_MAXV / 3))
#else
#define MAXV GS_MAXV
#define MAXP GS_MAXV
#endif
using Mesh = mesh<VOut, POut, MAXV, MAXP, topology::triangle>;

[[object]] void obj(object_data Payload &p [[payload]], mesh_grid_properties grid,
                    uint tid [[thread_index_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]],
                    const device ushort *ib [[buffer(0)]], const device packed_float3 *vb [[buffer(1)]],
                    constant DrawArgs &da [[buffer(2)]], const device float4 *inst [[buffer(3)]],
                    constant Uni &u [[buffer(4)]]) {
#if WARP_V <= 32
  threadgroup atomic_uint valid;
  atomic_fetch_and_explicit(&valid, 0u, memory_order_relaxed);
#endif
  uint iid = tg.y, gidx = tg.x * WARP_V + tid;
  bool ok = gidx < da.indexCount;
#if PACK
  uint k = da.indexCount <= WARP_V / 2 ? WARP_V / da.indexCount : 1;
  if (k > 1) {
    iid = tg.y * k + tid / da.indexCount;
    gidx = tid % da.indexCount;
    ok = iid < da.instanceCount && tid < k * da.indexCount;
  }
#endif
  if (ok) {
    uint vid = uint(int(ib[da.startIndex + gidx]) + da.baseVertex);
    float3 pos = float3(vb[vid]);
    float4 it = inst[da.startInstance + iid];
    float4 wp = float4(pos * it.w + it.xyz, 1.0);
    p.v[tid * NREG] = float4(dot(u.row[0], wp), dot(u.row[1], wp), dot(u.row[2], wp), 1.0);
#if NREG == 2
    p.v[tid * NREG + 1] = float4(pos.xy, 0, 0);
#endif
#if WARP_V <= 32
    atomic_fetch_or_explicit(&valid, 1u << tid, memory_order_relaxed);
#endif
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
#if WARP_V <= 32
    uint vm = atomic_load_explicit(&valid, memory_order_relaxed), pm = 0;
    for (uint i = 0; i < WARP_P; i++)
      pm |= uint(((vm >> (3 * i)) & 7u) == 7u) << i;
#else
    // expérience « grand warp » : seulement le cas paquets d'instances, validité par instance
    uint pm = 0;
    if (da.indexCount <= WARP_V / 2) {   // paquets d'instances : validité par instance
      uint kk = WARP_V / da.indexCount, ppi = da.indexCount / 3;
      for (uint i = 0; i < WARP_P; i++)
        pm |= uint(i / ppi < kk && tg.y * kk + i / ppi < da.instanceCount) << i;
    } else {
      for (uint i = 0; i < WARP_P; i++)
        pm |= uint(tg.x * WARP_V + 3 * i + 2 < da.indexCount) << i;
    }
#endif
    p.hdr[0] = iid; p.hdr[1] = pm; p.hdr[2] = tg.x;
#if MESH_MODE == 0
    grid.set_threadgroups_per_grid(uint3(WARP_P, 1, 1));
#elif MESH_MODE == 1
    grid.set_threadgroups_per_grid(uint3(popcount(pm), 1, 1));
#else
    grid.set_threadgroups_per_grid(uint3(pm != 0 ? 1 : 0, 1, 1));
#endif
  }
}

// GS de WARNO : bbox du triangle contre 4 cascades, émission dans la couche i (9 sommets max = 3 cascades)
struct Tri { float3 a, b, c; float2 lo, hi; };
static inline bool cascade(const device int *casc, uint i, Tri t, thread float3 &scale, thread float3 &bias) {
  float2 rlo = float2(casc[8 * i + 0], casc[8 * i + 1]), rhi = float2(casc[8 * i + 3], casc[8 * i + 4]);
  if (!(all(rhi >= t.lo) && all(t.hi >= rlo))) return false;
  float zlo = float(casc[8 * i + 2]) - 7095.0, zhi = float(casc[8 * i + 5]) + 1290.0;
  float3 d = float3(rhi, zhi) - float3(rlo, zlo);
  scale = float3(2.0, -2.0, 1.0) / d;
  bias = -scale * float3((rlo + rhi) * 0.5, zlo);
  return true;
}

[[mesh]] void meshfn(Mesh m, const object_data Payload &p [[payload]], uint3 tg [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]], const device int *casc [[buffer(0)]],
                     constant Uni &u [[buffer(1)]]) {
#if MESH_MODE == 0
  uint prim = tg.x;
#elif MESH_MODE == 1
  uint pmk = p.hdr[1];
  for (uint i = 0; i < tg.x; i++) pmk &= pmk - 1;
  uint prim = ctz(pmk);
#else
  uint prim = tid;
#endif
  bool active = (p.hdr[1] >> prim) & 1u;
  Tri t;
  uint nv = 0, np = 0;
#if MESH_MODE >= 2
  threadgroup atomic_uint cv, cp;
  threadgroup uint cnt[WARP_P];
  if (tid == 0) { atomic_store_explicit(&cv, 0u, memory_order_relaxed); atomic_store_explicit(&cp, 0u, memory_order_relaxed); }
  threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
  if (active) {
    float4 a = p.v[(3 * prim) * NREG], b = p.v[(3 * prim + 1) * NREG], c = p.v[(3 * prim + 2) * NREG];
    t.a = float3(dot(u.row[3], a), dot(u.row[4], a), dot(u.row[5], a));
    t.b = float3(dot(u.row[3], b), dot(u.row[4], b), dot(u.row[5], b));
    t.c = float3(dot(u.row[3], c), dot(u.row[4], c), dot(u.row[5], c));
    t.lo = min(t.a.xy, min(t.b.xy, t.c.xy));
    t.hi = max(t.a.xy, max(t.b.xy, t.c.xy));
  }
#if MESH_MODE == 3
  // passe 1 : compter ; préfixe déterministe
  uint n = 0;
  if (active)
    for (uint i = 0; i < 4 && n < GS_MAXV / 3; i++) { float3 s, bi; n += cascade(casc, i, t, s, bi); }
  cnt[tid] = n;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint base = 0, total = 0;
  for (uint i = 0; i < WARP_P; i++) { uint ci = cnt[i]; base += i < tid ? ci : 0; total += ci; }
  if (tid == 0) m.set_primitive_count(total);
  nv = base * 3; np = base;
#endif
  if (active) {
    uint emitted = 0;
    for (uint i = 0; i < 4 && emitted < GS_MAXV / 3; i++) {
      float3 s, bi;
      if (!cascade(casc, i, t, s, bi)) continue;
      emitted++;
#if MESH_MODE == 2
      nv = atomic_fetch_add_explicit(&cv, 3u, memory_order_relaxed);
      np = atomic_fetch_add_explicit(&cp, 1u, memory_order_relaxed);
#endif
      float3 pa = t.a * s + bi, pb = t.b * s + bi, pc = t.c * s + bi;
      VOut o;
      o.pos = float4(pa, 1.0);
#if NREG == 2
      o.uv = p.v[(3 * prim) * NREG + 1].xy;
#endif
      m.set_vertex(nv, o);
      o.pos = float4(pb, 1.0);
#if NREG == 2
      o.uv = p.v[(3 * prim + 1) * NREG + 1].xy;
#endif
      m.set_vertex(nv + 1, o);
      o.pos = float4(pc, 1.0);
#if NREG == 2
      o.uv = p.v[(3 * prim + 2) * NREG + 1].xy;
#endif
      m.set_vertex(nv + 2, o);
      m.set_index(np * 3, nv); m.set_index(np * 3 + 1, nv + 1); m.set_index(np * 3 + 2, nv + 2);
      POut po; po.layer = i;
      m.set_primitive(np, po);
      nv += 3; np += 1;
    }
  }
#if MESH_MODE < 2
  m.set_primitive_count(np);
#elif MESH_MODE == 2
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) m.set_primitive_count(atomic_load_explicit(&cp, memory_order_relaxed));
#endif
}

#if NREG == 2
[[fragment]] void frag(VOut in [[stage_in]]) { if (fract(in.uv.x * 7.0) < 0.05) discard_fragment(); }
#else
[[fragment]] void frag(VOut in [[stage_in]]) {}
#endif

// ---- Expérience « routage par cascade en pipeline vertex » (sans object ni mesh shader) ----
// vroute  : draw indexé, instancié x cascades. Le sommet sort dans la couche de sa cascade ; les triangles hors de la
//           cascade sont éliminés par le découpage. Pas de test de boîte, pas de troncature maxvertexcount (inexact à 4 cascades).
// vroute3 : cascade 3 seule, draw non indexé. Chaque sommet recalcule les 3 sommets de son triangle pour appliquer
//           la troncature : rejeté s'il touche les cascades 0, 1 et 2 (les 9 sommets du GS sont alors épuisés).
struct VOutL { float4 pos [[position]]; uint layer [[render_target_array_index]];
#if NREG == 2
  float2 uv;
#endif
};
static inline float3 vs_world(const device packed_float3 *vb, const device float4 *inst, constant Uni &u, uint vid, uint iid) {
  float4 it = inst[iid];
  float4 wp = float4(float3(vb[vid]) * it.w + it.xyz, 1.0);
  float4 p = float4(dot(u.row[0], wp), dot(u.row[1], wp), dot(u.row[2], wp), 1.0);
  return float3(dot(u.row[3], p), dot(u.row[4], p), dot(u.row[5], p));
}
static inline float4 to_cascade(const device int *casc, uint i, float3 p) {
  float2 rlo = float2(casc[8 * i + 0], casc[8 * i + 1]), rhi = float2(casc[8 * i + 3], casc[8 * i + 4]);
  float zlo = float(casc[8 * i + 2]) - 7095.0, zhi = float(casc[8 * i + 5]) + 1290.0;
  float3 scale = float3(2.0, -2.0, 1.0) / (float3(rhi, zhi) - float3(rlo, zlo));
  return float4(p * scale - scale * float3((rlo + rhi) * 0.5, zlo), 1.0);
}
[[vertex]] VOutL vroute(uint vid [[vertex_id]], uint iid [[instance_id]],
                        const device packed_float3 *vb [[buffer(1)]], constant DrawArgs &da [[buffer(2)]],
                        const device float4 *inst [[buffer(3)]], constant Uni &u [[buffer(4)]],
                        const device int *casc [[buffer(5)]]) {
  uint c = iid % NCASC;
  VOutL o;
  o.pos = to_cascade(casc, c, vs_world(vb, inst, u, vid, da.startInstance + iid / NCASC));
  o.layer = c;
#if NREG == 2
  o.uv = float3(vb[vid]).xy;
#endif
  return o;
}
[[vertex]] VOutL vroute3(uint vid [[vertex_id]], uint iid [[instance_id]], const device ushort *ib [[buffer(0)]],
                         const device packed_float3 *vb [[buffer(1)]], constant DrawArgs &da [[buffer(2)]],
                         const device float4 *inst [[buffer(3)]], constant Uni &u [[buffer(4)]],
                         const device int *casc [[buffer(5)]]) {
  uint tri = vid / 3, corner = vid % 3;
  float3 p[3];
  uint own = 0;
  for (uint k = 0; k < 3; k++) {
    uint v = uint(int(ib[da.startIndex + 3 * tri + k]) + da.baseVertex);
    p[k] = vs_world(vb, inst, u, v, da.startInstance + iid);
    if (k == corner) own = v;
  }
  float2 lo = min(p[0].xy, min(p[1].xy, p[2].xy)), hi = max(p[0].xy, max(p[1].xy, p[2].xy));
  uint hits = 0;
  for (uint i = 0; i < 3; i++) {
    float2 rlo = float2(casc[8 * i + 0], casc[8 * i + 1]), rhi = float2(casc[8 * i + 3], casc[8 * i + 4]);
    hits += uint(all(rhi >= lo) && all(hi >= rlo));
  }
  VOutL o;
  o.pos = hits == 3 ? float4(0, 0, 2, 1) : to_cascade(casc, 3, p[corner]);   // tronqué : hors du volume, éliminé
  o.layer = 3;
#if NREG == 2
  o.uv = float3(vb[own]).xy;
#endif
  return o;
}
#if NREG == 2
[[fragment]] void fragL(VOutL in [[stage_in]]) { if (fract(in.uv.x * 7.0) < 0.05) discard_fragment(); }
#else
[[fragment]] void fragL(VOutL in [[stage_in]]) {}
#endif
