#include "air_signature.hpp"
#include "airconv_error.hpp"
#include "dxbc_converter.hpp"
#include "dxil_converter.hpp"
#include "nt/air_builder.hpp"
#include "llvm/IR/Constants.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <format>
#include <optional>
#include <utility>

namespace dxmt::dxbc {

using namespace microsoft;

std::tuple<uint32_t, uint32_t, uint32_t>
get_vertex_primitive_count_in_warp(D3D10_SB_PRIMITIVE primitive, bool strip) {
  switch (primitive) {
  case D3D10_SB_PRIMITIVE_POINT:
    return {32, 32, 1};
  case D3D10_SB_PRIMITIVE_LINE:
    if (strip)
      return {31, 31, 2};
    return {32, 16, 2};
  case D3D10_SB_PRIMITIVE_TRIANGLE:
    if (strip)
      return {30, 30, 3};
    return {30, 10, 3};
  case D3D10_SB_PRIMITIVE_LINE_ADJ:
    if (strip)
      return {29, 29, 4};
    return {32, 8, 4};
  case D3D10_SB_PRIMITIVE_TRIANGLE_ADJ:
    if (strip)
      return {28, 14, 6};
    return {30, 5, 6};
  case D3D11_SB_PRIMITIVE_1_CONTROL_POINT_PATCH:
    return {32, 32, 1};
  case D3D11_SB_PRIMITIVE_2_CONTROL_POINT_PATCH:
    return {32, 16, 2};
  case D3D11_SB_PRIMITIVE_3_CONTROL_POINT_PATCH:
    return {30, 10, 3};
  case D3D11_SB_PRIMITIVE_4_CONTROL_POINT_PATCH:
    return {32, 8, 4};
  case D3D11_SB_PRIMITIVE_5_CONTROL_POINT_PATCH:
    return {30, 6, 5};
  case D3D11_SB_PRIMITIVE_6_CONTROL_POINT_PATCH:
    return {30, 5, 6};
  case D3D11_SB_PRIMITIVE_7_CONTROL_POINT_PATCH:
    return {28, 4, 7};
  case D3D11_SB_PRIMITIVE_8_CONTROL_POINT_PATCH:
    return {32, 4, 8};
  case D3D11_SB_PRIMITIVE_9_CONTROL_POINT_PATCH:
    return {27, 3, 9};
  case D3D11_SB_PRIMITIVE_10_CONTROL_POINT_PATCH:
    return {30, 3, 10};
  case D3D11_SB_PRIMITIVE_11_CONTROL_POINT_PATCH:
    return {22, 2, 11};
  case D3D11_SB_PRIMITIVE_12_CONTROL_POINT_PATCH:
    return {24, 2, 12};
  case D3D11_SB_PRIMITIVE_13_CONTROL_POINT_PATCH:
    return {26, 2, 13};
  case D3D11_SB_PRIMITIVE_14_CONTROL_POINT_PATCH:
    return {28, 2, 14};
  case D3D11_SB_PRIMITIVE_15_CONTROL_POINT_PATCH:
    return {30, 2, 15};
  case D3D11_SB_PRIMITIVE_16_CONTROL_POINT_PATCH:
    return {32, 2, 16};
  case D3D11_SB_PRIMITIVE_17_CONTROL_POINT_PATCH:
  case D3D11_SB_PRIMITIVE_18_CONTROL_POINT_PATCH:
  case D3D11_SB_PRIMITIVE_19_CONTROL_POINT_PATCH:
  case D3D11_SB_PRIMITIVE_20_CONTROL_POINT_PATCH:
  case D3D11_SB_PRIMITIVE_21_CONTROL_POINT_PATCH:
  case D3D11_SB_PRIMITIVE_22_CONTROL_POINT_PATCH:
  case D3D11_SB_PRIMITIVE_23_CONTROL_POINT_PATCH:
  case D3D11_SB_PRIMITIVE_24_CONTROL_POINT_PATCH:
  case D3D11_SB_PRIMITIVE_25_CONTROL_POINT_PATCH:
  case D3D11_SB_PRIMITIVE_26_CONTROL_POINT_PATCH:
  case D3D11_SB_PRIMITIVE_27_CONTROL_POINT_PATCH:
  case D3D11_SB_PRIMITIVE_28_CONTROL_POINT_PATCH:
  case D3D11_SB_PRIMITIVE_29_CONTROL_POINT_PATCH:
  case D3D11_SB_PRIMITIVE_30_CONTROL_POINT_PATCH:
  case D3D11_SB_PRIMITIVE_31_CONTROL_POINT_PATCH:
  case D3D11_SB_PRIMITIVE_32_CONTROL_POINT_PATCH:
    return {
        uint32_t(primitive) - uint32_t(D3D11_SB_PRIMITIVE_1_CONTROL_POINT_PATCH) + 1, 1,
        uint32_t(primitive) - uint32_t(D3D11_SB_PRIMITIVE_1_CONTROL_POINT_PATCH) + 1
    };
  default:
    break;
  }

  return {0, 0, 0};
}

uint32_t
gs_mesh_threads_limit(const SM50ShaderInternal *pGeometryStage) {
  // ponytail: a DXIL geometry shader keeps one invocation per mesh threadgroup, the two-pass
  // conversion below has only been exercised with DXBC. Batch them when a DXIL GS workload shows up.
  if (pGeometryStage->dxil)
    return 1;
  // the GS body runs twice (count, then write): only if it has no side effect but its emits
  if (!pGeometryStage->shader_info.uavMap.empty())
    return 1;
  return SM50GSMeshThreadsLimit(pGeometryStage->gs_max_vertex_output);
}

/*
 * Payload slot of each vertex of primitive `primitive_id` of the warp: load_vertex(slot, vertex of the primitive)
 */
static llvm::Error
gs_primitive_vertices(
    llvm::IRBuilder<> &builder, D3D10_SB_PRIMITIVE primitive, bool is_strip, uint32_t vertex_per_primitive,
    pvalue primitive_id, const std::function<void(pvalue, uint32_t)> &load_vertex
) {
  auto const one_const = builder.getInt32(1);
  auto const two_const = builder.getInt32(2);
  if (is_strip && primitive != D3D10_SB_PRIMITIVE_POINT) {
    auto leading_vertex_index = primitive_id;
    switch (primitive) {
    case microsoft::D3D10_SB_PRIMITIVE_TRIANGLE: {
      /*
      primitive 0: {0, 1, 2}
      primitive 1: {1, 3, 2}
      ...
      primitive n: {n, n + 1 + (n & 1), n + 2 - (n & 1)}

      I know primitive restart can be broken! just assume the 1st triangle of the thunk is at correct winding
      */
      auto odd_bit = builder.CreateAnd(leading_vertex_index, builder.getInt32(1));
      load_vertex(leading_vertex_index, 0);
      load_vertex(builder.CreateAdd(builder.CreateAdd(leading_vertex_index, builder.getInt32(1)), odd_bit), 1);
      load_vertex(builder.CreateSub(builder.CreateAdd(leading_vertex_index, builder.getInt32(2)), odd_bit), 2);
      break;
    }
    case microsoft::D3D10_SB_PRIMITIVE_LINE: {
      /*
      primitive 0: {0, 1}
      primitive 1: {1, 2}
      ...
      primitive n: {n, n+1}
      */
      load_vertex(leading_vertex_index, 0);
      load_vertex(builder.CreateAdd(leading_vertex_index, builder.getInt32(1)), 1);
      break;
    }
    case microsoft::D3D10_SB_PRIMITIVE_LINE_ADJ: {
      /*
      primitive 0: {0, 1, 2, 3}
      primitive 1: {1, 2, 3, 4}
      ...
      primitive n: {n, n+1, n+2, n+3}
      */
      load_vertex(leading_vertex_index, 0);
      load_vertex(builder.CreateAdd(leading_vertex_index, builder.getInt32(1)), 1);
      load_vertex(builder.CreateAdd(leading_vertex_index, builder.getInt32(2)), 2);
      load_vertex(builder.CreateAdd(leading_vertex_index, builder.getInt32(3)), 3);
      break;
    }
    case microsoft::D3D10_SB_PRIMITIVE_TRIANGLE_ADJ: {
      /*
      assume 16 vertices for 6 primitives:
      primitive 0: {0, (-2), 2, 6, 4, 3}
      primitive 1: {2, 5, 6, 8, 4, 0}
      primitive 2: {4, 2, 6,10, 8, 7}
      primitive 3: {6, 9,10,12, 8, 4}
      primitive 4: {8, 6, 10,14,12,11}
      primitive 5: {10,13,14,(16),12,8}
      ...
      primitive n (inter): {2n, -, 2(n + 1 + (n & 1)), -, 2(n + 2 - (n & 1)), -}
      primitive n (outer1): {-, is_even(n) ? 2n - 2 :2n + 3 : , -, -, -, -} mod 17 (-2 mapped to 15)
      primitive n (outer2): {-, -, -, 6 + 2n, -, -} mod 15 (16 mapped to 1)
      primitive n (outer3): {-, -, -, -, -, is_even(n) ? 2n + 3: 2n - 2}
      */
      auto double_ = [&](pvalue val) { return builder.CreateMul(two_const, val);};
      auto odd_bit = builder.CreateAnd(leading_vertex_index, one_const);
      auto _2n = double_(leading_vertex_index);
      load_vertex(_2n, 0);
      load_vertex(double_(builder.CreateAdd(builder.CreateAdd(leading_vertex_index, one_const), odd_bit)), 2);
      load_vertex(double_(builder.CreateSub(builder.CreateAdd(leading_vertex_index, two_const), odd_bit)), 4);
      auto _2np3 = builder.CreateAdd(_2n, builder.getInt32(3));
      auto _2nm2 = builder.CreateSub(_2n, two_const);
      auto is_odd = builder.CreateICmpEQ(odd_bit, one_const);
      load_vertex(builder.CreateURem(builder.CreateSelect(is_odd, _2np3, _2nm2), builder.getInt32(33)), 1);
      load_vertex(builder.CreateURem(builder.CreateAdd(_2n, builder.getInt32(6)), builder.getInt32(31)), 3);
      load_vertex(builder.CreateSelect(is_odd, _2nm2, _2np3), 5);
      break;
    }
    default:
      return llvm::make_error<UnsupportedFeature>(std::format(
          "unhandled geometry shader input primitive strip: {}", (uint32_t)primitive
      ));
    }
  } else {
    auto leading_vertex_index = builder.CreateMul(builder.getInt32(vertex_per_primitive), primitive_id);
    for (uint32_t vid = 0; vid < vertex_per_primitive; vid++) {
      load_vertex(builder.CreateAdd(leading_vertex_index, builder.getInt32(vid)), vid);
    }
  }
  return llvm::Error::success();
}

/*
 * What a GS invocation has emitted so far: D3D strips turned into the vertices and primitives of a mesh.
 * Vertices of the current strip are provisional until the strip is cut with enough of them.
 */
struct GSEmitCounters {
  llvm::IRBuilder<> &builder;
  llvm::Type *int_ty;
  air::MeshOutputTopology topology;
  uint32_t max_vertex_out;
  bool limit; // a slice holds max_vertex_out vertices: as D3D does, ignore what is emitted past it
  llvm::AllocaInst *next_write_vertex; // vertex of the current strip (of the invocation for points)
  llvm::AllocaInst *vertex_offset;     // first vertex of the current strip
  llvm::AllocaInst *primitive_count;
  llvm::AllocaInst *emitted_vertices = nullptr;
  pvalue accepted = nullptr; // `limit` only: the emit being converted is within max_vertex_out

  GSEmitCounters(
      llvm::IRBuilder<> &builder, llvm::Type *int_ty, air::MeshOutputTopology topology, uint32_t max_vertex_out,
      bool limit
  ) : builder(builder), int_ty(int_ty), topology(topology), max_vertex_out(max_vertex_out), limit(limit) {
    next_write_vertex = builder.CreateAlloca(int_ty, nullptr, "next_write_vertex");
    vertex_offset = builder.CreateAlloca(int_ty, nullptr, "vertex_offset");
    primitive_count = builder.CreateAlloca(int_ty, nullptr, "primitive_count");
    if (limit)
      emitted_vertices = builder.CreateAlloca(int_ty, nullptr, "emitted_vertices");
    reset(builder.getInt32(0), builder.getInt32(0));
  }

  void
  reset(pvalue first_vertex, pvalue first_primitive) {
    builder.CreateStore(builder.getInt32(0), next_write_vertex);
    builder.CreateStore(first_vertex, vertex_offset);
    builder.CreateStore(first_primitive, primitive_count);
    if (limit)
      builder.CreateStore(builder.getInt32(0), emitted_vertices);
  }

  /* Returns the increment of the vertex counters for this emit */
  pvalue
  begin_emit() {
    if (!limit)
      return builder.getInt32(1);
    auto emitted = builder.CreateLoad(int_ty, emitted_vertices);
    accepted = builder.CreateICmpULT(emitted, builder.getInt32(max_vertex_out));
    auto increment = builder.CreateZExt(accepted, int_ty);
    builder.CreateStore(builder.CreateAdd(emitted, increment), emitted_vertices);
    return increment;
  }

  /* An emit that only counts */
  void
  count_emit() {
    auto current_write_vertex = builder.CreateLoad(int_ty, next_write_vertex);
    builder.CreateStore(builder.CreateAdd(begin_emit(), current_write_vertex), next_write_vertex);
  }

  void
  cut() {
    if (topology == air::MeshOutputTopology::Point)
      return; // there is nothing to cut!
    // a strip needs 3 vertices (2 for lines) to count
    const uint32_t first_primitive_at = topology == air::MeshOutputTopology::Triangle ? 2 : 1;
    auto const zero_const = builder.getInt32(0);
    auto current_write_vertex = builder.CreateLoad(int_ty, next_write_vertex);
    builder.CreateStore(zero_const, next_write_vertex);

    auto has_valid_primitive = builder.CreateICmpUGT(current_write_vertex, builder.getInt32(first_primitive_at));
    auto add_primitive_count = builder.CreateSelect(
        has_valid_primitive, builder.CreateSub(current_write_vertex, builder.getInt32(first_primitive_at)), zero_const
    );
    auto add_vertex_count = builder.CreateSelect(has_valid_primitive, current_write_vertex, zero_const);
    builder.CreateStore(builder.CreateAdd(builder.CreateLoad(int_ty, vertex_offset), add_vertex_count), vertex_offset);
    builder.CreateStore(
        builder.CreateAdd(builder.CreateLoad(int_ty, primitive_count), add_primitive_count), primitive_count
    );
  }

  /* vertices | primitives << 16, once the last strip is cut */
  pvalue
  emitted() {
    auto vertices = builder.CreateLoad(int_ty, topology == air::MeshOutputTopology::Point ? next_write_vertex : vertex_offset);
    auto primitives =
        topology == air::MeshOutputTopology::Point ? vertices : builder.CreateLoad(int_ty, primitive_count);
    return builder.CreateOr(vertices, builder.CreateShl(primitives, 16));
  }
};

struct GSWarpLayout {
  uint32_t vertex_count;         // threads of the object threadgroup
  uint32_t primitive_count;
  uint32_t vertex_per_primitive;
  uint32_t scale;                // see SM50GSWarpScale()
  uint32_t mesh_threads;         // GS invocations per mesh threadgroup
  /*
   * The object stage runs the counting pass of the GS, for each primitive of its warp: a warp
   * that emits nothing launches no mesh threadgroup, and the mesh stage needs no pass of its own
   * to lay out its output.
   */
  bool precount;
  uint32_t payload_size;
  // payload, in words: 4 of header, a validity word per vertex, 32 words of emitted counts, the vertices
  uint32_t payload_count_offset;
  uint32_t payload_vertex_offset;
};

static GSWarpLayout
gs_warp_layout(
    const SM50ShaderInternal *pVertexStage, const SM50ShaderInternal *pGeometryStage, bool strip, bool rootsig
) {
  auto [vertex_count, primitive_count, vertex_per_primitive] =
      get_vertex_primitive_count_in_warp(pGeometryStage->gs_input_primitive, strip);
  auto scale = SM50GSWarpScale(pVertexStage->max_output_register, vertex_per_primitive, strip);
  auto mesh_threads = std::clamp(gs_mesh_threads_limit(pGeometryStage), 1u, std::max(1u, primitive_count * scale));
  return {
      .vertex_count = vertex_count * scale,
      .primitive_count = primitive_count * scale,
      .vertex_per_primitive = vertex_per_primitive,
      .scale = scale,
      .mesh_threads = mesh_threads,
      // ponytail: one count per primitive, so a single GS instance; and the GS tables of a root
      // signature (d3d12) are not bound to the object stage. Both keep counting in the mesh stage.
      .precount = mesh_threads > 1 && pGeometryStage->gs_instance_count == 1 && !rootsig,
      .payload_size = SM50GSPayloadSize(pVertexStage->max_output_register, scale),
      .payload_count_offset = 4 + 32 * scale,
      .payload_vertex_offset = 4 + 32 * scale + 32,
  };
}

llvm::Error
convert_dxbc_geometry_shader(
    SM50ShaderInternal *pShaderInternal, const char *name, SM50ShaderInternal *pVertexStage, llvm::LLVMContext &context,
    llvm::Module &module, SM50_SHADER_COMPILATION_ARGUMENT_DATA *pArgs
) {
  auto func_signature = pShaderInternal->func_signature; // copy
  auto shader_info = &(pShaderInternal->shader_info);

  bool is_strip = false;

  uint32_t max_output_register = pShaderInternal->max_output_register;
  SM50_SHADER_PSO_GEOMETRY_SHADER_DATA *pso_data = nullptr;
  if (args_get_data<SM50_SHADER_PSO_GEOMETRY_SHADER, SM50_SHADER_PSO_GEOMETRY_SHADER_DATA>(pArgs, &pso_data)) {
    is_strip = pso_data->strip_topology;
  }
  SM50_SHADER_METAL_VERSION metal_version = SM50_SHADER_METAL_310;
  SM50_SHADER_FLAG shader_flags = {};
  SM50_SHADER_COMMON_DATA *sm50_common = nullptr;
  if (args_get_data<SM50_SHADER_COMMON, SM50_SHADER_COMMON_DATA>(pArgs, &sm50_common)) {
    metal_version = sm50_common->metal_version;
    shader_flags = sm50_common->flags;
  }
  SM50_SHADER_ROOT_SIGNATURE_DATA *rootsig = nullptr;
  args_get_data<SM50_SHADER_ROOT_SIGNATURE, SM50_SHADER_ROOT_SIGNATURE_DATA>(pArgs, &rootsig);

  Prologues prologue;
  Epilogues epilogue;

  io_binding_map resource_map;
  air::AirType types(context);

  {
    SignatureContext sig_ctx(prologue, epilogue, func_signature, resource_map);
    for (auto &p : pShaderInternal->signature_handlers) {
      p(sig_ctx);
    }
  }
  auto& gs_output_handlers = pShaderInternal->mesh_output_handlers;

  auto binding_map = rootsig ? setup_binding_rootsig(
                                   shader_info, func_signature, module, D3D10_SB_GEOMETRY_SHADER, rootsig->bytecode,
                                   rootsig->bytecode_length
                               )
                             : setup_binding_table2(shader_info, func_signature, module);

  auto gs_output_topology = pShaderInternal->gs_output_topology;
  int32_t max_vertex_out = pShaderInternal->gs_max_vertex_output;
  air::MeshOutputTopology topology = air::MeshOutputTopology::Point;
  switch (gs_output_topology) {
  case microsoft::D3D10_SB_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP:
    topology = air::MeshOutputTopology::Triangle;
    break;
  case microsoft::D3D10_SB_PRIMITIVE_TOPOLOGY_LINESTRIP:
    topology = air::MeshOutputTopology::Line;
    break;
  case microsoft::D3D10_SB_PRIMITIVE_TOPOLOGY_POINTLIST:
    func_signature.DefineMeshVertexOutput(air::OutputPointSize{});
    break;
  default:
    return llvm::make_error<UnsupportedFeature>("unsupported geometry shader output topology");
  }

  const auto warp = gs_warp_layout(pVertexStage, pShaderInternal, is_strip, rootsig != nullptr);
  const uint32_t warp_primitive_count = warp.primitive_count, vertex_per_primitive = warp.vertex_per_primitive;

  /*
   * Batched layout: a mesh threadgroup runs `mesh_threads` GS invocations, one per thread.
   * A first pass over the GS body counts what each invocation emits, a prefix sum over the
   * threadgroup gives each thread its slice of the mesh, a second pass writes it. The mesh
   * stays compact and in input primitive order.
   * The counting pass runs here, or in the object stage which then leaves the counts in the
   * payload (warp.precount).
   */
  const uint32_t mesh_threads = warp.mesh_threads;
  const bool batched = mesh_threads > 1;
  // a primitive slot is (speculatively) written for each emitted vertex, see call_emit
  uint32_t max_primitive_out = max_vertex_out;
  if (batched && topology == air::MeshOutputTopology::Triangle)
    max_primitive_out = std::max(max_vertex_out, 2) - 2;
  if (batched && topology == air::MeshOutputTopology::Line)
    max_primitive_out = std::max(max_vertex_out, 1) - 1;
  // batched: speculative writes that fall out of a thread's slice go to its scratch vertex/primitive
  const uint32_t scratch_vertex_base = mesh_threads * max_vertex_out;
  const uint32_t scratch_primitive_base = mesh_threads * max_primitive_out;

  uint32_t payload_idx = func_signature.DefineInput(air::InputPayload{.size = warp.payload_size});
  auto mesh_idx = func_signature.DefineInput(
      batched ? air::InputMesh{scratch_vertex_base + mesh_threads, scratch_primitive_base + mesh_threads, topology}
              // it's intended to declare a bit more max primitive count
              : air::InputMesh{(uint32_t)max_vertex_out, (uint32_t)max_vertex_out, topology}
  );
  uint32_t tg_in_grid_idx = func_signature.DefineInput(air::InputThreadgroupPositionInGrid{});
  uint32_t thread_idx = batched ? func_signature.DefineInput(air::InputThreadPositionInThreadgroup{}) : ~0u;

  if (pShaderInternal->clip_distance_scalars.size() > 0) {
    func_signature.DefineMeshVertexOutput(
        air::OutputClipDistance{.count = pShaderInternal->clip_distance_scalars.size()}
    );
  }

  auto [function, function_metadata] = func_signature.CreateFunction(name, context, module, 0, false);

  auto entry_bb = llvm::BasicBlock::Create(context, "entry", function);
  auto active_ = llvm::BasicBlock::Create(context, "active", function);
  auto epilogue_bb = llvm::BasicBlock::Create(context, "epilogue", function);
  llvm::IRBuilder<> builder(entry_bb);
  llvm::raw_null_ostream nulldbg{};
  llvm::air::AIRBuilder air(
      {
          .sampleNaNToZero = bool(shader_flags & SM50_SHADER_FLAG_SAMPLE_NAN_TO_ZERO),
          .defuseFma = bool(shader_flags & SM50_SHADER_FLAG_DEFUSE_FMA),
      },
      builder, nulldbg
  );
  setup_metal_version(module, metal_version);

  auto tg_position_in_grid = function->getArg(tg_in_grid_idx);
  pvalue primitive_id_in_warp = builder.CreateExtractElement(tg_position_in_grid, (uint32_t)0);
  pvalue thread_id = builder.getInt32(0);
  if (batched) {
    thread_id = builder.CreateExtractElement(function->getArg(thread_idx), (uint32_t)0);
    primitive_id_in_warp =
        builder.CreateAdd(builder.CreateMul(primitive_id_in_warp, builder.getInt32(mesh_threads)), thread_id);
  }
  resource_map.gs_instance_id = builder.CreateExtractElement(tg_position_in_grid, (uint32_t)1);

  auto const zero_const = builder.getInt32(0);
  auto const one_const = builder.getInt32(1);
  auto const two_const = builder.getInt32(2);
  GSEmitCounters counters(builder, types._int, topology, max_vertex_out, batched);
  auto next_write_vertex = counters.next_write_vertex, vertex_offset = counters.vertex_offset,
       primitive_count = counters.primitive_count;

  auto mesh_ptr = function->getArg(mesh_idx);

  resource_map.mesh = mesh_ptr;

  auto emit_clip_distances = [&](llvm::Value *vertex) {
    for (auto x : llvm::enumerate(pShaderInternal->clip_distance_scalars)) {
      if (x.value().reg >= max_output_register)
        continue;
      auto src_ptr = builder.CreateGEP(
          llvm::ArrayType::get(types._float4, max_output_register), resource_map.output.ptr_float4,
          {builder.getInt32(0), builder.getInt32(x.value().reg), builder.getInt32(x.value().component)}
      );
      air.CreateSetMeshClipDistance(vertex, builder.getInt32(x.index()), builder.CreateLoad(types._float, src_ptr));
    }
  };

  IOHelper io(air, function, types, resource_map);

  // batched only
  bool count_only = false; // converting the counting pass: emit/cut only maintain the counters
  pvalue end_vertex = nullptr, end_primitive = nullptr; // end of the thread's slice of the mesh
  pvalue scratch_vertex = nullptr, scratch_primitive = nullptr;
  auto begin_emit = [&]() { return counters.begin_emit(); };
  /*
   * A vertex is written before its strip is known to be valid, and so is the primitive it leads.
   * Within the slice, such a write is overwritten by the next valid strip; past its end it would
   * land in the next thread's slice: redirect it to the scratch slot.
   */
  auto vertex_slot = [&](pvalue vertex) {
    return batched ? builder.CreateSelect(
                         builder.CreateAnd(counters.accepted, builder.CreateICmpULT(vertex, end_vertex)), vertex,
                         scratch_vertex
                     )
                   : vertex;
  };
  auto primitive_slot = [&](pvalue primitive) {
    return batched ? builder.CreateSelect(
                         builder.CreateAnd(counters.accepted, builder.CreateICmpULT(primitive, end_primitive)), primitive,
                         scratch_primitive
                     )
                   : primitive;
  };

  if (topology == air::MeshOutputTopology::Triangle) {
    resource_map.call_emit = [&]() {
      auto current_write_vertex = builder.CreateLoad(types._int, next_write_vertex);
      builder.CreateStore(builder.CreateAdd(begin_emit(), current_write_vertex), next_write_vertex);
      if (count_only)
        return;
      auto current_vertex_offset = builder.CreateLoad(types._int, vertex_offset);
      auto current_vertex_with_offset = builder.CreateAdd(current_vertex_offset, current_write_vertex);

      auto current_primitive_count = builder.CreateLoad(types._int, primitive_count);
      auto current_primitive_idx = primitive_slot(builder.CreateAdd(current_primitive_count, current_write_vertex));

      auto even_winding = builder.CreateZExt(
          builder.CreateICmpEQ(zero_const, builder.CreateAnd(current_write_vertex, one_const)), types._int
      );

      auto current_vertex_slot = vertex_slot(current_vertex_with_offset);
      MeshOutputContext gs_out_ctx{current_vertex_slot, current_primitive_idx};
      for (auto &h : gs_output_handlers) {
        h(gs_out_ctx, io);
      }
      emit_clip_distances(current_vertex_slot);

      auto triple_primitive_idx = builder.CreateMul(current_primitive_idx, builder.getInt32(3));
      air.CreateSetMeshIndex(
          builder.CreateAdd(triple_primitive_idx, zero_const), current_vertex_with_offset
      );
      air.CreateSetMeshIndex(
          builder.CreateAdd(triple_primitive_idx, one_const),
          builder.CreateSub(builder.CreateAdd(current_vertex_with_offset, two_const), even_winding)
      );
      air.CreateSetMeshIndex(
          builder.CreateAdd(triple_primitive_idx, two_const),
          builder.CreateAdd(builder.CreateAdd(current_vertex_with_offset, one_const), even_winding)
      );
    };
  } else if (topology == air::MeshOutputTopology::Line) {
    resource_map.call_emit = [&]() {
      auto current_write_vertex = builder.CreateLoad(types._int, next_write_vertex);
      builder.CreateStore(builder.CreateAdd(begin_emit(), current_write_vertex), next_write_vertex);
      if (count_only)
        return;

      auto current_vertex_offset = builder.CreateLoad(types._int, vertex_offset);
      auto current_vertex_with_offset = builder.CreateAdd(current_vertex_offset, current_write_vertex);

      auto current_primitive_count = builder.CreateLoad(types._int, primitive_count);
      auto current_primitive_idx = primitive_slot(builder.CreateAdd(current_primitive_count, current_write_vertex));

      auto current_vertex_slot = vertex_slot(current_vertex_with_offset);
      MeshOutputContext gs_out_ctx{current_vertex_slot, current_primitive_idx};
      for (auto &h : gs_output_handlers) {
        h(gs_out_ctx, io);
      }
      emit_clip_distances(current_vertex_slot);

      auto double_primitive_idx = builder.CreateMul(current_primitive_idx, builder.getInt32(2));
      air.CreateSetMeshIndex(
          builder.CreateAdd(double_primitive_idx, zero_const), current_vertex_with_offset
      );
      air.CreateSetMeshIndex(
          builder.CreateAdd(double_primitive_idx, one_const),
          builder.CreateAdd(current_vertex_with_offset, one_const)
      );
    };
  } else {
    resource_map.call_emit = [&]() {
      // only one accumulator to maintain, simple one ~
      auto current_write_vertex = builder.CreateLoad(types._int, next_write_vertex);
      builder.CreateStore(builder.CreateAdd(begin_emit(), current_write_vertex), next_write_vertex);
      if (count_only)
        return;

      // a point is its own primitive: vertex_offset is the start of the thread's slice (0 if not batched)
      auto current_point = vertex_slot(builder.CreateAdd(builder.CreateLoad(types._int, vertex_offset), current_write_vertex));
      MeshOutputContext gs_out_ctx{current_point, current_point};
      for (auto &h : gs_output_handlers) {
        h(gs_out_ctx, io);
      }
      emit_clip_distances(current_point);
      air.CreateSetMeshPointSize(current_point, air.getFloat(1.0));
      air.CreateSetMeshIndex(current_point, current_point);
    };
  }
  resource_map.call_cut = [&]() { counters.cut(); };

  // payload: (primitive id base, primitive id modulus, -, -), a validity word per vertex, the vertices
  auto payload = function->getArg(payload_idx);
  auto primitive_id_base =
      builder.CreateLoad(types._int, builder.CreateConstInBoundsGEP1_32(types._int, payload, 0));
  // primitives per instance when the object shader packs several instances in the warp, ~0u otherwise
  auto primitive_id_modulus =
      builder.CreateLoad(types._int, builder.CreateConstInBoundsGEP1_32(types._int, payload, 1));
  auto vertex_valid_ptr = builder.CreateConstInBoundsGEP1_32(types._int, payload, 4);
  auto emitted_count_ptr = builder.CreateConstInBoundsGEP1_32(types._int, payload, warp.payload_count_offset);

  resource_map.patch_id =
      builder.CreateAdd(primitive_id_base, builder.CreateURem(primitive_id_in_warp, primitive_id_modulus));

  auto input_per_vertex_type = llvm::ArrayType::get(types._int4, pVertexStage->max_output_register);
  auto input_per_vertex_type_float = llvm::ArrayType::get(types._float4, pVertexStage->max_output_register);

  auto input_ptr_int4_type = llvm::ArrayType::get(input_per_vertex_type,  vertex_per_primitive);
  resource_map.input.ptr_int4 = builder.CreateAlloca(input_ptr_int4_type);
  resource_map.input.ptr_float4 = builder.CreateBitCast(
      resource_map.input.ptr_int4,
      llvm::ArrayType::get(input_per_vertex_type_float, vertex_per_primitive)->getPointerTo()
  );
  resource_map.input_element_count = pVertexStage->max_output_register;

  resource_map.output.ptr_int4 = builder.CreateAlloca(llvm::ArrayType::get(types._int4, max_output_register));
  resource_map.output.ptr_float4 = builder.CreateBitCast(
      resource_map.output.ptr_int4, llvm::ArrayType::get(types._float4, max_output_register)->getPointerTo()
  );
  resource_map.output_element_count = max_output_register;

  setup_temp_register(shader_info, resource_map, types, module, builder);
  setup_immediate_constant_buffer(shader_info, resource_map, types, module, builder);

  auto sync_bb = batched && !warp.precount ? llvm::BasicBlock::Create(context, "sync", function) : nullptr;
  auto inactive_bb = sync_bb ? sync_bb : epilogue_bb;
  auto assemble_bb = llvm::BasicBlock::Create(context, "assemble", function);
  // the last threadgroup of the warp may run more threads than there are primitives left
  auto in_warp = builder.CreateICmpULT(primitive_id_in_warp, builder.getInt32(warp_primitive_count));
  auto entry_exit_bb = builder.GetInsertBlock();
  // precount: (vertices | primitives << 16) emitted by this thread, by the threads before it, by the threadgroup
  pvalue own_count = nullptr, count_before = zero_const, count_total = zero_const;
  if (warp.precount) {
    auto emitted_count = [&](pvalue primitive) {
      auto exists = builder.CreateICmpULT(primitive, builder.getInt32(warp_primitive_count));
      return builder.CreateSelect(
          exists,
          builder.CreateLoad(
              types._int,
              builder.CreateGEP(types._int, emitted_count_ptr, {builder.CreateSelect(exists, primitive, zero_const)})
          ),
          zero_const
      );
    };
    own_count = emitted_count(primitive_id_in_warp);
    auto first_primitive_of_group = builder.CreateSub(primitive_id_in_warp, thread_id);
    for (uint32_t thread = 0; thread < mesh_threads; thread++) {
      auto count = emitted_count(builder.CreateAdd(first_primitive_of_group, builder.getInt32(thread)));
      count_total = builder.CreateAdd(count_total, count);
      count_before = builder.CreateAdd(
          count_before, builder.CreateSelect(builder.CreateICmpULT(builder.getInt32(thread), thread_id), count, zero_const)
      );
    }
    auto set_count_bb = llvm::BasicBlock::Create(context, "set_count", function);
    auto counted_bb = llvm::BasicBlock::Create(context, "counted", function);
    builder.CreateCondBr(builder.CreateICmpEQ(thread_id, zero_const), set_count_bb, counted_bb);
    builder.SetInsertPoint(set_count_bb);
    air.CreateSetMeshPrimitiveCount(builder.CreateLShr(count_total, 16));
    builder.CreateBr(counted_bb);
    builder.SetInsertPoint(counted_bb);
    // a primitive that emits nothing has nothing to write (and an invalid one was counted as such)
    builder.CreateCondBr(builder.CreateICmpNE(own_count, zero_const), assemble_bb, epilogue_bb);
  } else {
    builder.CreateCondBr(in_warp, assemble_bb, inactive_bb);
  }
  builder.SetInsertPoint(assemble_bb);

  auto vertices_ptr = builder.CreateBitCast(
      builder.CreateConstInBoundsGEP1_32(types._int, function->getArg(payload_idx), warp.payload_vertex_offset),
      types._int4->getPointerTo((uint32_t)air::AddressSpace::object_data)
  );

  // the primitive is valid if all its vertices are (in the draw range, not a strip cut index)
  std::vector<pvalue> primitive_vertices(vertex_per_primitive);
  pvalue is_valid = builder.getInt1(true);
  auto load_vertex = [&](pvalue vertex_index, uint32_t vid) {
    primitive_vertices[vid] = vertex_index;
    if (warp.precount)
      return; // already checked by the object stage
    is_valid = builder.CreateAnd(
        is_valid,
        builder.CreateICmpNE(
            builder.CreateLoad(types._int, builder.CreateGEP(types._int, vertex_valid_ptr, {vertex_index})), zero_const
        )
    );
  };

  if (auto err = gs_primitive_vertices(
          builder, pShaderInternal->gs_input_primitive, is_strip, vertex_per_primitive, primitive_id_in_warp, load_vertex
      )) {
    return err;
  }

  auto assemble_exit_bb = builder.GetInsertBlock();
  builder.CreateCondBr(is_valid, active_, inactive_bb);
  builder.SetInsertPoint(active_);
  for (uint32_t vid = 0; vid < vertex_per_primitive; vid++) {
    for (unsigned reg = 0; reg < pVertexStage->max_output_register; reg++) {
      builder.CreateStore(
          builder.CreateLoad(
              types._int4,
              builder.CreateGEP(
                  types._int4, vertices_ptr,
                  {builder.CreateAdd(
                      builder.CreateMul(primitive_vertices[vid], builder.getInt32(pVertexStage->max_output_register)),
                      builder.getInt32(reg)
                  )}
              )
          ),
          builder.CreateGEP(
              input_ptr_int4_type, resource_map.input.ptr_int4,
              {zero_const, builder.getInt32(vid), builder.getInt32(reg)}
          )
      );
    }
  }

  struct context ctx{
      .builder = builder,
      .air = air,
      .binding = *binding_map,
      .llvm = context,
      .module = module,
      .function = function,
      .resource = resource_map,
      .types = types,
      .pso_sample_mask = 0xffffffff,
      .shader_type = pShaderInternal->shader_type,
      .metal_version = metal_version,
  };

  if (auto err = prologue.Run(io)) {
    return err;
  }

  auto convert_body = [&](llvm::BasicBlock *return_bb) {
    return pShaderInternal->dxil ? dxil::convert_basicblocks(pShaderInternal->dxil_entry, ctx, return_bb)
                                 : convert_basicblocks(pShaderInternal->entry(), ctx, return_bb);
  };

  // start of the thread's slice of the mesh, and what goes in it
  auto begin_write_pass = [&](pvalue before, pvalue own) {
    auto first_vertex = builder.CreateAnd(before, 0xffff);
    auto first_primitive = builder.CreateLShr(before, 16);
    end_vertex = builder.CreateAdd(first_vertex, builder.CreateAnd(own, 0xffff));
    end_primitive = builder.CreateAdd(first_primitive, builder.CreateLShr(own, 16));
    scratch_vertex = builder.CreateAdd(thread_id, builder.getInt32(scratch_vertex_base));
    scratch_primitive = builder.CreateAdd(thread_id, builder.getInt32(scratch_primitive_base));
    counters.reset(first_vertex, first_primitive);
  };

  if (warp.precount) {
    begin_write_pass(count_before, own_count);
  } else if (batched) {
    // pass 1: count. The GS has no side effect but its emits, so what it computes for them is dead code here.
    auto count_end_bb = llvm::BasicBlock::Create(context, "count_end", function);
    auto set_count_bb = llvm::BasicBlock::Create(context, "set_count", function);
    auto synced_bb = llvm::BasicBlock::Create(context, "synced", function);
    auto write_bb = llvm::BasicBlock::Create(context, "write", function);

    count_only = true;
    auto count_entry = convert_body(count_end_bb);
    count_only = false;
    if (auto err = count_entry.takeError()) {
      return err;
    }
    builder.CreateBr(count_entry.get());

    builder.SetInsertPoint(count_end_bb);
    counters.cut();
    // at most maxvertexcount vertices (see begin_emit), so at most max_primitive_out primitives: it fits the slice
    auto emitted = counters.emitted();
    auto count_exit_bb = builder.GetInsertBlock();
    builder.CreateBr(sync_bb);

    // prefix sum of (vertex | primitive << 16) counts over the threadgroup
    builder.SetInsertPoint(sync_bb);
    auto emitted_phi = builder.CreatePHI(types._int, 3);
    emitted_phi->addIncoming(zero_const, entry_exit_bb);
    emitted_phi->addIncoming(zero_const, assemble_exit_bb);
    emitted_phi->addIncoming(emitted, count_exit_bb);
    own_count = emitted_phi;
    auto counts_type = llvm::ArrayType::get(types._int, mesh_threads);
    auto counts = new llvm::GlobalVariable(
        module, counts_type, false, llvm::GlobalValue::InternalLinkage, llvm::Constant::getNullValue(counts_type),
        "gs_emit_counts", nullptr, llvm::GlobalValue::NotThreadLocal, (uint32_t)air::AddressSpace::threadgroup
    );
    builder.CreateStore(own_count, builder.CreateGEP(counts_type, counts, {zero_const, thread_id}));
    air.CreateBarrier(llvm::air::MemFlags::Threadgroup);
    for (uint32_t thread = 0; thread < mesh_threads; thread++) {
      auto count =
          builder.CreateLoad(types._int, builder.CreateGEP(counts_type, counts, {zero_const, builder.getInt32(thread)}));
      count_total = builder.CreateAdd(count_total, count);
      count_before = builder.CreateAdd(
          count_before, builder.CreateSelect(builder.CreateICmpULT(builder.getInt32(thread), thread_id), count, zero_const)
      );
    }
    builder.CreateCondBr(builder.CreateICmpEQ(thread_id, zero_const), set_count_bb, synced_bb);

    builder.SetInsertPoint(set_count_bb);
    air.CreateSetMeshPrimitiveCount(builder.CreateLShr(count_total, 16));
    builder.CreateBr(synced_bb);

    // a thread that emits nothing has nothing to write
    builder.SetInsertPoint(synced_bb);
    builder.CreateCondBr(builder.CreateICmpNE(own_count, zero_const), write_bb, epilogue_bb);

    // pass 2: write
    builder.SetInsertPoint(write_bb);
    begin_write_pass(count_before, own_count);
  }

  auto real_entry = convert_body(epilogue_bb);
  if (auto err = real_entry.takeError()) {
    return err;
  }
  builder.CreateBr(real_entry.get());
  builder.SetInsertPoint(epilogue_bb);

  auto epilogue_result = epilogue.Run(function->getReturnType(), io);
  if (auto err = epilogue_result.takeError()) {
    return err;
  }
  if (!batched) {
    resource_map.call_cut();
    if (topology == air::MeshOutputTopology::Point)
      air.CreateSetMeshPrimitiveCount(builder.CreateLoad(types._int, next_write_vertex));
    else
      air.CreateSetMeshPrimitiveCount(builder.CreateLoad(types._int, primitive_count));
  }

  builder.CreateRetVoid();
  module.getOrInsertNamedMetadata("air.mesh")->addOperand(function_metadata);

  return llvm::Error::success();
}

llvm::Error
convert_dxbc_vertex_for_geometry_shader(
    const SM50ShaderInternal *pShaderInternal, const char *name, const SM50ShaderInternal *pGeometryStage,
    llvm::LLVMContext &context, llvm::Module &module, SM50_SHADER_COMPILATION_ARGUMENT_DATA *pArgs
) {
  auto func_signature = pShaderInternal->func_signature; // copy
  auto shader_info = &(pShaderInternal->shader_info);

  bool is_strip = false;

  uint32_t max_input_register = pShaderInternal->max_input_register;
  uint32_t max_output_register = pShaderInternal->max_output_register;
  SM50_SHADER_IA_INPUT_LAYOUT_DATA *ia_layout = nullptr;
  SM50_SHADER_PSO_GEOMETRY_SHADER_DATA *pso_data = nullptr;
  args_get_data<SM50_SHADER_IA_INPUT_LAYOUT, SM50_SHADER_IA_INPUT_LAYOUT_DATA>(pArgs, &ia_layout);
  if (args_get_data<SM50_SHADER_PSO_GEOMETRY_SHADER, SM50_SHADER_PSO_GEOMETRY_SHADER_DATA>(pArgs, &pso_data)) {
    is_strip = pso_data->strip_topology;
  }
  SM50_SHADER_METAL_VERSION metal_version = SM50_SHADER_METAL_310;
  SM50_SHADER_FLAG shader_flags = {};
  SM50_SHADER_COMMON_DATA *sm50_common = nullptr;
  if (args_get_data<SM50_SHADER_COMMON, SM50_SHADER_COMMON_DATA>(pArgs, &sm50_common)) {
    metal_version = sm50_common->metal_version;
    shader_flags = sm50_common->flags;
  }
  SM50_SHADER_ROOT_SIGNATURE_DATA *rootsig = nullptr;
  args_get_data<SM50_SHADER_ROOT_SIGNATURE, SM50_SHADER_ROOT_SIGNATURE_DATA>(pArgs, &rootsig);

  bool is_triadj_strip = is_strip && pGeometryStage->gs_input_primitive == D3D10_SB_PRIMITIVE_TRIANGLE_ADJ;
  bool is_indexed_draw = ia_layout && ia_layout->index_buffer_format > 0;
  bool is_dyanmic_ib = is_indexed_draw && ia_layout->index_buffer_format == SM50_INDEX_BUFFER_FORMAT_DYNAMIC;

  Prologues prologue;
  Epilogues epilogue;

  io_binding_map resource_map;
  air::AirType types(context);

  {
    SignatureContext sig_ctx(prologue, epilogue, func_signature, resource_map);
    sig_ctx.ia_layout = ia_layout;
    sig_ctx.skip_vertex_output = true;
    for (auto &p : pShaderInternal->signature_handlers) {
      p(sig_ctx);
    }
  }

  auto binding_map = rootsig ? setup_binding_rootsig(
                                   shader_info, func_signature, module, D3D10_SB_VERTEX_SHADER, rootsig->bytecode,
                                   rootsig->bytecode_length
                               )
                             : setup_binding_table2(shader_info, func_signature, module);

  const auto warp = gs_warp_layout(pShaderInternal, pGeometryStage, is_strip, rootsig != nullptr);
  const uint32_t warp_vertex_count = warp.vertex_count, warp_primitive_count = warp.primitive_count,
                 vertex_per_primitive = warp.vertex_per_primitive;

  uint32_t payload_idx = func_signature.DefineInput(air::InputPayload{.size = warp.payload_size});
  // (warp_size, 1, 1)
  uint32_t thread_id_idx = func_signature.DefineInput(air::InputThreadPositionInThreadgroup{});
  // (warp_count, instance_count, 1)
  // warp_count = ceil(index_count / warp_size)
  uint32_t tg_id_idx = func_signature.DefineInput(air::InputThreadgroupPositionInGrid{});
  func_signature.DefineInput(air::InputMeshGridProperties{});
  uint32_t draw_argument_idx = func_signature.DefineInput(air::ArgumentBindingBuffer{
      .buffer_size = {},
      .location_index = SM50_BINDING_INDEX_DRAW_ARGUMENTS,
      .array_size = 0,
      .memory_access = air::MemoryAccess::read,
      .address_space = air::AddressSpace::constant,
      .type = is_indexed_draw //
          ? is_dyanmic_ib
              ? air::MSLWhateverStruct{"draw_indexed_dynamic_arguments", types._dxmt_draw_dyn_ib_arguments}
              : air::MSLWhateverStruct{"draw_indexed_arguments", types._dxmt_draw_indexed_arguments}
          : air::MSLWhateverStruct{"draw_arguments", types._dxmt_draw_arguments},
      .arg_name = "draw_arguments",
      .raster_order_group = {}
  });

  uint32_t index_buffer_idx = ~0u;
  if (is_indexed_draw && !is_dyanmic_ib) {
    index_buffer_idx = func_signature.DefineInput(air::ArgumentBindingBuffer{
        .buffer_size = {},
        .location_index = SM50_BINDING_INDEX_INDEX_BUFFER,
        .array_size = 0,
        .memory_access = air::MemoryAccess::read,
        .address_space = air::AddressSpace::device,
        .type = ia_layout->index_buffer_format == 1 ? air::MSLRepresentableType(air::MSLUshort{})
                                                    : air::MSLRepresentableType(air::MSLUint{}),
        .raster_order_group = {}
    });
  }

  // precount: the counting pass of the GS runs here, with the GS resources bound to the object stage
  io_binding_map gs_resource_map;
  auto gs_binding_map = warp.precount ? setup_binding_table2(
                                            &pGeometryStage->shader_info, func_signature, module,
                                            SM50_BINDING_INDEX_CONSTANT_BUFFER2, SM50_BINDING_INDEX_ARGUMENT_TABLE2
                                        )
                                      : nullptr;

  auto [function, function_metadata] = func_signature.CreateFunction(name, context, module, 0, true);

  auto entry_bb = llvm::BasicBlock::Create(context, "entry", function);
  auto epilogue_bb = llvm::BasicBlock::Create(context, "epilogue", function);
  llvm::IRBuilder<> builder(entry_bb);
  llvm::raw_null_ostream nulldbg{};
  llvm::air::AIRBuilder air(
      {
          .sampleNaNToZero = bool(shader_flags & SM50_SHADER_FLAG_SAMPLE_NAN_TO_ZERO),
          .defuseFma = bool(shader_flags & SM50_SHADER_FLAG_DEFUSE_FMA),
      },
      builder, nulldbg
  );

  setup_metal_version(module, metal_version);

  auto index_check = llvm::BasicBlock::Create(context, "index_check", function);
  auto active = llvm::BasicBlock::Create(context, "active", function);
  auto will_dispatch = llvm::BasicBlock::Create(context, "will_dispatch", function);
  auto dispatch = llvm::BasicBlock::Create(context, "dispatch", function);
  auto return_ = llvm::BasicBlock::Create(context, "return", function);

  auto payload = function->getArg(payload_idx);
  auto thread_position_in_group = function->getArg(thread_id_idx);
  auto warp_vertex_id = builder.CreateExtractElement(thread_position_in_group, (uint32_t)0);

  auto threadgroup_position_in_grid = function->getArg(tg_id_idx);
  auto warp_id = builder.CreateExtractElement(threadgroup_position_in_grid, (uint32_t)0);
  pvalue instance_id = builder.CreateExtractElement(threadgroup_position_in_grid, (uint32_t)1);

  auto draw_arguments = builder.CreateLoad(
    is_indexed_draw ? is_dyanmic_ib ? types._dxmt_draw_dyn_ib_arguments : types._dxmt_draw_indexed_arguments
                    : types._dxmt_draw_arguments,
    function->getArg(draw_argument_idx)
  );

  auto vertex_count = builder.CreateExtractValue(draw_arguments, 0);

  resource_map.input.ptr_int4 = builder.CreateAlloca(llvm::ArrayType::get(types._int4, max_input_register));
  resource_map.input.ptr_float4 = builder.CreateBitCast(
      resource_map.input.ptr_int4, llvm::ArrayType::get(types._float4, max_input_register)->getPointerTo()
  );
  resource_map.input_element_count = max_input_register;

  // payload: (primitive id base, primitive id modulus, -, -), a validity word per vertex, the vertices
  auto vertex_valid_ptr =
      builder.CreateGEP(types._int, builder.CreateConstInBoundsGEP1_32(types._int, payload, 4), {warp_vertex_id});
  auto payload_output_ptr = builder.CreateConstInBoundsGEP1_32(types._int, payload, warp.payload_vertex_offset);

  resource_map.output.ptr_int4 = builder.CreateBitCast(
      payload_output_ptr,
      llvm::ArrayType::get(types._int4, max_output_register)->getPointerTo((uint32_t)air::AddressSpace::object_data)
  );
  resource_map.output.ptr_int4 = builder.CreateGEP(
      resource_map.output.ptr_int4->getType()->getNonOpaquePointerElementType(), resource_map.output.ptr_int4,
      {warp_vertex_id}
  );
  resource_map.output.ptr_float4 = builder.CreateBitCast(
      payload_output_ptr,
      llvm::ArrayType::get(types._float4, max_output_register)->getPointerTo((uint32_t)air::AddressSpace::object_data)
  );
  resource_map.output.ptr_float4 = builder.CreateGEP(
      resource_map.output.ptr_float4->getType()->getNonOpaquePointerElementType(), resource_map.output.ptr_float4,
      {warp_vertex_id}
  );

  resource_map.output_element_count = max_output_register;

  setup_temp_register(shader_info, resource_map, types, module, builder);
  setup_immediate_constant_buffer(shader_info, resource_map, types, module, builder);

  struct context ctx{
      .builder = builder,
      .air = air,
      .binding = *binding_map,
      .llvm = context,
      .module = module,
      .function = function,
      .resource = resource_map,
      .types = types,
      .pso_sample_mask = 0xffffffff,
      .shader_type = pShaderInternal->shader_type,
      .metal_version = metal_version,
  };

  std::optional<GSEmitCounters> gs_counters;
  auto gs_input_type =
      llvm::ArrayType::get(llvm::ArrayType::get(types._int4, max_output_register), vertex_per_primitive);
  if (warp.precount) {
    air::MeshOutputTopology gs_topology = air::MeshOutputTopology::Point;
    switch (pGeometryStage->gs_output_topology) {
    case microsoft::D3D10_SB_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP:
      gs_topology = air::MeshOutputTopology::Triangle;
      break;
    case microsoft::D3D10_SB_PRIMITIVE_TOPOLOGY_LINESTRIP:
      gs_topology = air::MeshOutputTopology::Line;
      break;
    case microsoft::D3D10_SB_PRIMITIVE_TOPOLOGY_POINTLIST:
      break;
    default:
      return llvm::make_error<UnsupportedFeature>("unsupported geometry shader output topology");
    }
    gs_counters.emplace(builder, types._int, gs_topology, pGeometryStage->gs_max_vertex_output, true);
    gs_resource_map.input.ptr_int4 = builder.CreateAlloca(gs_input_type);
    gs_resource_map.input.ptr_float4 = builder.CreateBitCast(
        gs_resource_map.input.ptr_int4,
        llvm::ArrayType::get(llvm::ArrayType::get(types._float4, max_output_register), vertex_per_primitive)
            ->getPointerTo()
    );
    gs_resource_map.input_element_count = max_output_register;
    gs_resource_map.output.ptr_int4 =
        builder.CreateAlloca(llvm::ArrayType::get(types._int4, pGeometryStage->max_output_register));
    gs_resource_map.output.ptr_float4 = builder.CreateBitCast(
        gs_resource_map.output.ptr_int4,
        llvm::ArrayType::get(types._float4, pGeometryStage->max_output_register)->getPointerTo()
    );
    gs_resource_map.output_element_count = pGeometryStage->max_output_register;
    gs_resource_map.gs_instance_id = builder.getInt32(0);
    gs_resource_map.call_emit = [&]() { gs_counters->count_emit(); };
    gs_resource_map.call_cut = [&]() { gs_counters->cut(); };
    setup_temp_register(&pGeometryStage->shader_info, gs_resource_map, types, module, builder);
    setup_immediate_constant_buffer(&pGeometryStage->shader_info, gs_resource_map, types, module, builder);
  }

  pvalue global_index_id =
      builder.CreateAdd(builder.CreateMul(warp_id, builder.getInt32(warp_vertex_count)), warp_vertex_id);
  pvalue in_draw_range = nullptr; // defaults to global_index_id < vertex_count
  pvalue primitive_id_modulus = builder.getInt32(~0u);

  if (!is_strip && vertex_per_primitive) {
    /*
     * Instance packing, see SM50GSInstancesPerWarp(): a draw of few vertices and many instances
     * (instanced quads...) would run one mostly idle threadgroup per instance. Fill the warp with
     * several instances instead: thread -> (instance, vertex), the grid being (1, ceil(instances / packed)).
     */
    auto instance_count = builder.CreateExtractValue(draw_arguments, 1);
    auto packable = builder.CreateAnd(
        builder.CreateAnd(
            builder.CreateICmpNE(vertex_count, builder.getInt32(0)),
            builder.CreateICmpULE(vertex_count, builder.getInt32(warp_vertex_count / 2))
        ),
        builder.CreateICmpEQ(builder.CreateURem(vertex_count, builder.getInt32(vertex_per_primitive)), builder.getInt32(0))
    );
    auto divisor = builder.CreateSelect(packable, vertex_count, builder.getInt32(warp_vertex_count));
    auto packed_instance_count = builder.CreateUDiv(builder.getInt32(warp_vertex_count), divisor); // 1 if not packable
    auto instance_in_warp = builder.CreateUDiv(warp_vertex_id, divisor);
    auto packed_instance_id =
        builder.CreateAdd(builder.CreateMul(instance_id, packed_instance_count), instance_in_warp);
    in_draw_range = builder.CreateSelect(
        packable,
        builder.CreateAnd(
            builder.CreateICmpULT(instance_in_warp, packed_instance_count),
            builder.CreateICmpULT(packed_instance_id, instance_count)
        ),
        builder.CreateICmpULT(global_index_id, vertex_count)
    );
    global_index_id = builder.CreateSelect(
        packable, builder.CreateSub(warp_vertex_id, builder.CreateMul(instance_in_warp, divisor)), global_index_id
    );
    instance_id = builder.CreateSelect(packable, packed_instance_id, instance_id);
    // SV_PrimitiveID restarts with each instance
    primitive_id_modulus = builder.CreateSelect(
        packable, builder.CreateUDiv(divisor, builder.getInt32(vertex_per_primitive)), primitive_id_modulus
    );
  }

  if (is_triadj_strip) {
    // 2nd and 32th vertex are not read but 33th and -2th (or 1st) are.
    auto const one_const = builder.getInt32(1);
    auto signed_max = [&](pvalue a, pvalue b) {
      return builder.CreateSelect(builder.CreateICmpSLT(a, b), b, a);
    };
    auto signed_min = [&](pvalue a, pvalue b) {
      return builder.CreateSelect(builder.CreateICmpSLT(a, b), a, b);
    };

    auto is_second_thread = builder.CreateICmpEQ(warp_vertex_id, one_const);
    global_index_id = builder.CreateSelect(
      is_second_thread,
      builder.CreateAdd(global_index_id, builder.getInt32(31)), global_index_id
    );
    auto is_last_thread = builder.CreateICmpEQ(warp_vertex_id, builder.getInt32(31));
    global_index_id = builder.CreateSelect(
      is_last_thread,
      signed_max(one_const, builder.CreateSub(global_index_id, builder.getInt32(33))),
      signed_min(global_index_id, builder.CreateSub(vertex_count, one_const)) // may degenerate
    );
  }

  // invalid until proven otherwise: out of the draw range, or a strip cut index
  builder.CreateStore(builder.getInt32(0), vertex_valid_ptr);

  if (!in_draw_range)
    in_draw_range = builder.CreateICmp(llvm::CmpInst::ICMP_ULT, global_index_id, vertex_count);
  builder.CreateCondBr(in_draw_range, index_check, will_dispatch);
  builder.SetInsertPoint(index_check);

  if (is_dyanmic_ib) {
    auto start_index = builder.CreateExtractValue(draw_arguments, 2);
    auto index_format = builder.CreateExtractValue(draw_arguments, 5);
    auto index_buffer = builder.CreateExtractValue(draw_arguments, 6);

    auto no_index = llvm::BasicBlock::Create(context, "no_index", function);
    auto r16_index = llvm::BasicBlock::Create(context, "r16_index", function);
    auto r32_index = llvm::BasicBlock::Create(context, "r32_index", function);
    auto finalize_index = llvm::BasicBlock::Create(context, "finalize_index", function);

    auto switch_ = builder.CreateSwitch(index_format, no_index, 2);
    switch_->addCase(air.getInt(42), r32_index);
    switch_->addCase(air.getInt(57), r16_index);

    builder.SetInsertPoint(finalize_index);
    auto vertex_id = builder.CreatePHI(air.getIntTy(), 3);
    resource_map.vertex_id = vertex_id;
    builder.CreateCondBr(
        builder.CreateICmp(llvm::CmpInst::ICMP_NE, builder.getInt32(-1), vertex_id), active, will_dispatch
    );
    {
      builder.SetInsertPoint(no_index);
      builder.CreateBr(finalize_index);
      vertex_id->addIncoming(global_index_id, no_index);
    }
    {
      builder.SetInsertPoint(r32_index);
      auto loaded_r32 = builder.CreateLoad(
          builder.getInt32Ty(),
          builder.CreateGEP(
              builder.getInt32Ty(),
              builder.CreatePointerCast(
                  index_buffer, builder.getInt32Ty()->getPointerTo((unsigned)air::AddressSpace::device)
              ),
              {builder.CreateAdd(start_index, global_index_id)}
          )
      );
      vertex_id->addIncoming(loaded_r32, r32_index);
      builder.CreateBr(finalize_index);
    }
    {
      builder.SetInsertPoint(r16_index);
      auto loaded_r16 = builder.CreateLoad(
          builder.getInt16Ty(),
          builder.CreateGEP(
              builder.getInt16Ty(),
              builder.CreatePointerCast(
                  index_buffer, builder.getInt16Ty()->getPointerTo((unsigned)air::AddressSpace::device)
              ),
              {builder.CreateAdd(start_index, global_index_id)}
          )
      );
      vertex_id->addIncoming(
          builder.CreateSub(
              builder.CreateZExt(builder.CreateAdd(loaded_r16, builder.getInt16(1)), types._int), air.getInt(1)
          ),
          r16_index
      );
      builder.CreateBr(finalize_index);
    }

  } else if (index_buffer_idx != ~0u) {
    auto start_index = builder.CreateExtractValue(draw_arguments, 2);
    auto index_buffer = function->getArg(index_buffer_idx);
    auto index_buffer_element_type = index_buffer->getType()->getNonOpaquePointerElementType();
    auto vertex_id = builder.CreateLoad(
        index_buffer_element_type,
        builder.CreateGEP(index_buffer_element_type, index_buffer, {builder.CreateAdd(start_index, global_index_id)})
    );
    // so 0xFFFF is mapped to 0xFFFFFFFF, other values are zero-extended
    resource_map.vertex_id = builder.CreateSub(
      builder.CreateZExt(
        builder.CreateAdd(
          vertex_id, llvm::ConstantInt::get(index_buffer_element_type, 1)
        ),
        types._int
      ),
      llvm::ConstantInt::get(types._int, 1)
    );

    builder.CreateCondBr(
        builder.CreateICmp(llvm::CmpInst::ICMP_NE, builder.getInt32(-1), resource_map.vertex_id), active, will_dispatch
    );
  } else {
    resource_map.vertex_id = global_index_id;
    builder.CreateBr(active);
  }

  builder.SetInsertPoint(active);

  builder.CreateStore(builder.getInt32(1), vertex_valid_ptr);

  resource_map.base_vertex_id = builder.CreateExtractValue(draw_arguments, is_indexed_draw ? 3 : 2);
  resource_map.instance_id = instance_id;
  resource_map.vertex_id_with_base = builder.CreateAdd(resource_map.vertex_id, resource_map.base_vertex_id);
  resource_map.base_instance_id = builder.CreateExtractValue(draw_arguments, is_indexed_draw ? 4 : 3);
  resource_map.instance_id_with_base = builder.CreateAdd(resource_map.instance_id, resource_map.base_instance_id);

  IOHelper io(air, function, types, resource_map);

  if (auto err = prologue.Run(io)) {
    return err;
  }

  llvm::Expected<llvm::BasicBlock *> real_entry =
      pShaderInternal->dxil ? dxil::convert_basicblocks(pShaderInternal->dxil_entry, ctx, epilogue_bb)
                            : convert_basicblocks(pShaderInternal->entry(), ctx, epilogue_bb);
  if (auto err = real_entry.takeError()) {
    return err;
  }
  builder.CreateBr(real_entry.get());

  builder.SetInsertPoint(epilogue_bb);
  auto epilogue_result = epilogue.Run(function->getReturnType(), io);
  if (auto err = epilogue_result.takeError()) {
    return err;
  }

  builder.CreateBr(will_dispatch);

  builder.SetInsertPoint(will_dispatch);

  auto emitted_count_ptr = builder.CreateConstInBoundsGEP1_32(types._int, payload, warp.payload_count_offset);
  if (warp.precount) {
    /*
     * Counting pass of the GS: the thread of vertex n also takes primitive n of the warp. It only
     * counts what the GS emits, so what the GS computes for its outputs is dead code here.
     */
    auto const zero_const = builder.getInt32(0);
    auto count_bb = llvm::BasicBlock::Create(context, "count", function);
    auto count_active_bb = llvm::BasicBlock::Create(context, "count_active", function);
    auto count_end_bb = llvm::BasicBlock::Create(context, "count_end", function);
    auto count_store_bb = llvm::BasicBlock::Create(context, "count_store", function);
    auto counted_bb = llvm::BasicBlock::Create(context, "counted", function);
    auto barrier_flags =
        llvm::air::MemFlags((uint32_t)llvm::air::MemFlags::Threadgroup | (uint32_t)llvm::air::MemFlags::ObjectData);
    if (getenv("GS_HACK_BARRIER")) // EXPÉRIENCE
      barrier_flags = llvm::air::MemFlags(atoi(getenv("GS_HACK_BARRIER")));

    // every vertex of the warp is in the payload
    if (!getenv("GS_HACK_NOBARRIER1")) // EXPÉRIENCE
    air.CreateBarrier(barrier_flags);
    builder.CreateCondBr(
        builder.CreateICmpULT(warp_vertex_id, builder.getInt32(warp_primitive_count)), count_bb, counted_bb
    );

    // the primitive is valid if all its vertices are (in the draw range, not a strip cut index)
    builder.SetInsertPoint(count_bb);
    auto vertex_valid_base = builder.CreateConstInBoundsGEP1_32(types._int, payload, 4);
    std::vector<pvalue> primitive_vertices(vertex_per_primitive);
    pvalue is_valid = builder.getInt1(true);
    if (auto err = gs_primitive_vertices(
            builder, pGeometryStage->gs_input_primitive, is_strip, vertex_per_primitive, warp_vertex_id,
            [&](pvalue vertex_index, uint32_t vid) {
              primitive_vertices[vid] = vertex_index;
              is_valid = builder.CreateAnd(
                  is_valid,
                  builder.CreateICmpNE(
                      builder.CreateLoad(types._int, builder.CreateGEP(types._int, vertex_valid_base, {vertex_index})),
                      zero_const
                  )
              );
            }
        )) {
      return err;
    }
    if (getenv("GS_HACK_CONSTCOUNT")) // EXPÉRIENCE : pas d'évaluation du GS dans l'étage object, compte constant
      is_valid = builder.getInt1(false);
    auto count_exit_bb = builder.GetInsertBlock();
    builder.CreateCondBr(is_valid, count_active_bb, count_store_bb);

    builder.SetInsertPoint(count_active_bb);
    auto vertices_ptr = builder.CreateBitCast(
        payload_output_ptr, types._int4->getPointerTo((uint32_t)air::AddressSpace::object_data)
    );
    for (uint32_t vid = 0; vid < vertex_per_primitive; vid++) {
      for (unsigned reg = 0; reg < max_output_register; reg++) {
        builder.CreateStore(
            builder.CreateLoad(
                types._int4,
                builder.CreateGEP(
                    types._int4, vertices_ptr,
                    {builder.CreateAdd(
                        builder.CreateMul(primitive_vertices[vid], builder.getInt32(max_output_register)),
                        builder.getInt32(reg)
                    )}
                )
            ),
            builder.CreateGEP(
                gs_input_type, gs_resource_map.input.ptr_int4, {zero_const, builder.getInt32(vid), builder.getInt32(reg)}
            )
        );
      }
    }
    gs_resource_map.patch_id = builder.CreateAdd(
        builder.CreateMul(warp_id, builder.getInt32(warp_primitive_count)),
        builder.CreateURem(warp_vertex_id, primitive_id_modulus)
    );
    struct context gs_ctx{
        .builder = builder,
        .air = air,
        .binding = *gs_binding_map,
        .llvm = context,
        .module = module,
        .function = function,
        .resource = gs_resource_map,
        .types = types,
        .pso_sample_mask = 0xffffffff,
        .shader_type = pGeometryStage->shader_type,
        .metal_version = metal_version,
    };
    auto gs_entry = convert_basicblocks(pGeometryStage->entry(), gs_ctx, count_end_bb);
    if (auto err = gs_entry.takeError()) {
      return err;
    }
    builder.CreateBr(gs_entry.get());

    builder.SetInsertPoint(count_end_bb);
    gs_counters->cut();
    auto emitted = gs_counters->emitted();
    auto count_end_exit_bb = builder.GetInsertBlock();
    builder.CreateBr(count_store_bb);

    builder.SetInsertPoint(count_store_bb);
    auto count = builder.CreatePHI(types._int, 2);
    count->addIncoming(getenv("GS_HACK_CONSTCOUNT") ? builder.getInt32(atoi(getenv("GS_HACK_CONSTCOUNT"))) : zero_const, count_exit_bb);
    count->addIncoming(emitted, count_end_exit_bb);
    builder.CreateStore(count, builder.CreateGEP(types._int, emitted_count_ptr, {warp_vertex_id}));
    builder.CreateBr(counted_bb);

    // the counts of the warp are in the payload
    builder.SetInsertPoint(counted_bb);
    if (!getenv("GS_HACK_NOBARRIER2")) // EXPÉRIENCE
    air.CreateBarrier(barrier_flags);
  }

  builder.CreateCondBr(
      builder.CreateICmp(llvm::CmpInst::ICMP_EQ, warp_vertex_id, builder.getInt32(0)), dispatch, return_
  );
  builder.SetInsertPoint(dispatch);

  // base and modulus of SV_PrimitiveID
  builder.CreateStore(
      builder.CreateMul(warp_id, builder.getInt32(warp_primitive_count)),
      builder.CreateConstInBoundsGEP1_32(types._int, payload, 0)
  );
  builder.CreateStore(primitive_id_modulus, builder.CreateConstInBoundsGEP1_32(types._int, payload, 1));

  // each mesh threadgroup runs `mesh_threads` GS invocations
  auto mesh_grid = air.getInt3(
      (warp_primitive_count + warp.mesh_threads - 1) / warp.mesh_threads, pGeometryStage->gs_instance_count, 1
  );
  if (warp.precount) {
    // none at all for a warp that emits nothing: with a GS that culls, that is most of them
    pvalue any_emitted = builder.getInt32(0);
    for (uint32_t primitive = 0; primitive < warp_primitive_count; primitive++)
      any_emitted = builder.CreateOr(
          any_emitted,
          builder.CreateLoad(types._int, builder.CreateConstInBoundsGEP1_32(types._int, emitted_count_ptr, primitive))
      );
    air.CreateSetMeshProperties(
        builder.CreateSelect(builder.CreateICmpNE(any_emitted, builder.getInt32(0)), mesh_grid, air.getInt3(0, 0, 0))
    );
  } else {
    air.CreateSetMeshProperties(mesh_grid);
  }

  builder.CreateBr(return_);

  builder.SetInsertPoint(return_);

  builder.CreateRetVoid();

  module.getOrInsertNamedMetadata("air.object")->addOperand(function_metadata);
  return llvm::Error::success();
};

} // namespace dxmt::dxbc
