#include "d3d11_pipeline_cache.hpp"
#include "airconv_public.h"
#include "d3d11_device.hpp"
#include "d3d11_shader.hpp"
#include "d3d11_pipeline.hpp"
#include "dxmt_shader_cache.hpp"
#include "dxmt_tasks.hpp"
#include "log/log.hpp"
#include "sha1/sha1_util.hpp"
#include "../d3d10/d3d10_shader.hpp"
#include "../d3d10/d3d10_input_layout.hpp"
#include "config/config.hpp"
#include <cstring>
#include <chrono>
#include <deque>
#include <shared_mutex>

namespace dxmt {

class MTLD3D11InputLayout final
    : public MTLD3D11DeviceChild<IMTLD3D11InputLayout> {
public:
  MTLD3D11InputLayout(MTLD3D11Device *device, ManagedInputLayout input_layout)
      : MTLD3D11DeviceChild<IMTLD3D11InputLayout>(device),
        input_layout(input_layout), d3d10(this) {}

  ~MTLD3D11InputLayout() {}

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid,
                                           void **ppvObject) final {
    if (ppvObject == nullptr)
      return E_POINTER;

    *ppvObject = nullptr;

    if (riid == __uuidof(IUnknown) || riid == __uuidof(ID3D11DeviceChild) ||
        riid == __uuidof(ID3D11InputLayout) ||
        riid == __uuidof(IMTLD3D11InputLayout)) {
      *ppvObject = ref(this);
      return S_OK;
    }

    if (riid == __uuidof(ID3D10DeviceChild) ||
        riid == __uuidof(ID3D10InputLayout)) {
      *ppvObject = ref(&d3d10);
      return S_OK;
    }

    if (logQueryInterfaceError(__uuidof(ID3D11InputLayout), riid)) {
      WARN("D3D311InputLayout: Unknown interface query ", str::format(riid));
    }

    return E_NOINTERFACE;
  };

  ManagedInputLayout GetManagedInputLayout() override { return input_layout; }

private:
  ManagedInputLayout input_layout;
  MTLD3D10InputLayout d3d10;
};

/* neutron: the interpolants a VS writes and a PS reads, from the DXBC signature chunks. airconv links
   them by name "reg<register>_<first component>" and component type, so a VS output signature that
   holds every user input of a PS links with it in Metal (used to find partners for pipeline priming). */
struct SignatureElement {
  uint32_t reg;
  uint32_t first_component;
  uint32_t component_type;
  bool operator==(const SignatureElement &o) const {
    return reg == o.reg && first_component == o.first_component && component_type == o.component_type;
  }
};

static uint32_t
DxbcProgramType(const void *pBytecode, size_t BytecodeLength) {
  auto p = (const uint8_t *)pBytecode;
  if (BytecodeLength < 32 || memcmp(p, "DXBC", 4))
    return ~0u;
  uint32_t count;
  memcpy(&count, p + 28, 4);
  for (uint32_t i = 0; i < count && 32 + i * 4 + 4 <= BytecodeLength; i++) {
    uint32_t off;
    memcpy(&off, p + 32 + i * 4, 4);
    if (off + 12 > BytecodeLength)
      continue;
    if (!memcmp(p + off, "SHDR", 4) || !memcmp(p + off, "SHEX", 4)) {
      uint32_t version;
      memcpy(&version, p + off + 8, 4);
      return version >> 16; /* 0 PS, 1 VS, 2 GS, 3 HS, 4 DS, 5 CS */
    }
  }
  return ~0u;
}

/* output = false: the input signature (ISGN, ISG1), true: the output signature (OSGN, OSG1, OSG5).
   Only elements without a system value. */
static std::vector<SignatureElement>
DxbcSignature(const void *pBytecode, size_t BytecodeLength, bool output) {
  std::vector<SignatureElement> ret;
  auto p = (const uint8_t *)pBytecode;
  if (BytecodeLength < 32 || memcmp(p, "DXBC", 4))
    return ret;
  uint32_t count;
  memcpy(&count, p + 28, 4);
  for (uint32_t i = 0; i < count && 32 + i * 4 + 4 <= BytecodeLength; i++) {
    uint32_t off, size;
    memcpy(&off, p + 32 + i * 4, 4);
    if (off + 16 > BytecodeLength)
      continue;
    memcpy(&size, p + off + 4, 4);
    const uint8_t *c = p + off;
    size_t stride = 0, skip = 0;
    if (!memcmp(c, output ? "OSGN" : "ISGN", 4))
      stride = 24;
    else if (output && !memcmp(c, "OSG5", 4))
      stride = 28, skip = 4;
    else if (!memcmp(c, output ? "OSG1" : "ISG1", 4))
      stride = 32, skip = 4;
    if (!stride)
      continue;
    const uint8_t *data = c + 8;
    uint32_t n;
    memcpy(&n, data, 4);
    if (8 + (size_t)n * stride > size || off + 8 + size > BytecodeLength)
      return {};
    for (uint32_t e = 0; e < n; e++) {
      const uint8_t *el = data + 8 + e * stride + skip;
      uint32_t sysval, type, reg;
      memcpy(&sysval, el + 8, 4);
      memcpy(&type, el + 12, 4);
      memcpy(&reg, el + 16, 4);
      uint8_t mask = el[20];
      if (sysval != 0 || !(mask & 0xf))
        continue;
      ret.push_back({reg, (uint32_t)std::countr_zero((uint32_t)(mask & 0xf)), type});
    }
    return ret;
  }
  return ret;
}

/* neutron: which pixel shader variant bits change the converted code. airconv only applies the unorm
   output fix to float SV_Target outputs and only drops oDepth when the shader writes it, so a PS drawn
   with and without a depth target, or with other render target formats where it writes nothing,
   compiled the same function again (airconv and Metal). Returns false if the signature is unreadable. */
static bool
DxbcPixelOutputs(const void *pBytecode, size_t BytecodeLength, bool *writes_depth, uint32_t *float_targets) {
  auto p = (const uint8_t *)pBytecode;
  *writes_depth = false;
  *float_targets = 0;
  if (BytecodeLength < 32 || memcmp(p, "DXBC", 4))
    return false;
  uint32_t count;
  memcpy(&count, p + 28, 4);
  for (uint32_t i = 0; i < count && 32 + i * 4 + 4 <= BytecodeLength; i++) {
    uint32_t off, size;
    memcpy(&off, p + 32 + i * 4, 4);
    if (off + 16 > BytecodeLength)
      continue;
    memcpy(&size, p + off + 4, 4);
    const uint8_t *c = p + off;
    size_t stride = 0, skip = 0;
    if (!memcmp(c, "OSGN", 4))
      stride = 24;
    else if (!memcmp(c, "OSG1", 4))
      stride = 32, skip = 4;
    if (!stride)
      continue;
    const uint8_t *data = c + 8;
    uint32_t n;
    memcpy(&n, data, 4);
    if (8 + (size_t)n * stride > size || off + 8 + size > BytecodeLength)
      return false;
    for (uint32_t e = 0; e < n; e++) {
      const uint8_t *el = data + 8 + e * stride + skip;
      uint32_t name_off, type, reg;
      memcpy(&name_off, el, 4);
      memcpy(&type, el + 12, 4);
      memcpy(&reg, el + 16, 4);
      if (name_off >= size)
        return false;
      const char *name = (const char *)data + name_off;
      size_t max_len = size - name_off;
      if (!strnlen(name, max_len) || strnlen(name, max_len) == max_len)
        return false;
      if (!strncasecmp(name, "SV_Depth", 8)) {
        *writes_depth = true;
      } else if (!strcasecmp(name, "SV_Target")) {
        if (reg < 8 && type == 3 /* D3D_REGISTER_COMPONENT_FLOAT32 */)
          *float_targets |= 1u << reg;
      } else {
        return false; /* SV_Coverage, SV_StencilRef: keep the variant as it is */
      }
    }
    return true;
  }
  return false;
}

template <> struct task_trait<ThreadpoolWork *> {
  ThreadpoolWork *run_task(ThreadpoolWork *task) {
    return task->RunThreadpoolWork();
  }
  bool get_done(ThreadpoolWork *task) { return task->GetIsDone(); }
  void set_done(ThreadpoolWork *task) { task->SetIsDone(true); }
};

class PipelineCache : public MTLD3D11PipelineCacheBase {

  class CachedSM50Shader final : public Shader {
    PipelineCache *cache;
    sm50_shader_t shader = nullptr;
    Sha1Digest sha1_;
    MTL_SHADER_REFLECTION reflection_;
    MTL_SM50_SHADER_ARGUMENT *arguments_info_buffer;
    std::unordered_map<ShaderVariant, std::unique_ptr<CompiledShader>> variants;
    /* neutron: get_shader is called from pipeline creation (one mutex per pipeline kind) and now also
       from shader creation and pipeline priming on other threads */
    dxmt::mutex variants_mutex_;

  public:
    /* neutron: for pipeline priming */
    uint32_t program_type = ~0u;
    std::vector<SignatureElement> link_signature; /* VS: outputs, PS: inputs */
    std::atomic_bool used = false;                /* part of a pipeline a draw asked for */
    std::atomic_bool prime_queued = false;
    /* neutron: see DxbcPixelOutputs */
    bool ps_outputs_known = false;
    bool ps_writes_depth = true;
    uint32_t ps_float_targets = ~0u;

    CachedSM50Shader(PipelineCache *cache, sm50_shader_t shader_transferred,
                     const Sha1Digest &hash, MTL_SHADER_REFLECTION &reflection)
        : cache(cache), shader(shader_transferred), sha1_(hash),
          reflection_(reflection) {
      if (reflection_.NumConstantBuffers + reflection_.NumArguments) {
        arguments_info_buffer = (MTL_SM50_SHADER_ARGUMENT *)malloc(
            sizeof(MTL_SM50_SHADER_ARGUMENT) *
            (reflection_.NumConstantBuffers + reflection_.NumArguments));
        SM50GetArgumentsInfo(shader, arguments_info_buffer,
                             arguments_info_buffer +
                                 reflection_.NumConstantBuffers);
      } else {
        arguments_info_buffer = nullptr;
      }
    }

    ~CachedSM50Shader() {
      if (shader) {
        SM50Destroy(shader);
        if (arguments_info_buffer)
          free(arguments_info_buffer);
        shader = nullptr;
      }
    };

    CachedSM50Shader(CachedSM50Shader &&moved) = delete;
    CachedSM50Shader(const CachedSM50Shader &copy) = delete;

    virtual sm50_shader_t handle() { return shader; };
    virtual MTL_SHADER_REFLECTION &reflection() { return reflection_; }
    virtual MTL_SM50_SHADER_ARGUMENT *constant_buffers_info() {
      return arguments_info_buffer;
    };
    virtual MTL_SM50_SHADER_ARGUMENT *arguments_info() {
      return arguments_info_buffer + reflection_.NumConstantBuffers;
    };
    virtual CompiledShader *get_shader(ShaderVariant variant, bool background) {
      if (auto pixel = std::get_if<ShaderVariantPixel>(&variant); pixel && ps_outputs_known) {
        if (!ps_writes_depth)
          pixel->disable_depth_output = false;
        pixel->unorm_output_reg_mask &= ps_float_targets;
      }
      std::lock_guard<dxmt::mutex> lock(variants_mutex_);
      auto c = variants.insert({variant, nullptr});
      if (c.second) {
        c.first->second = std::visit(
            [=, this](auto var) {
              return CreateVariantShader(cache->device, this, var);
            },
            variant);
        if (background)
          cache->scheduler_.submit_background(c.first->second.get());
        else
          cache->scheduler_.submit(c.first->second.get());
      }
      return c.first->second.get();
    }
    virtual const Sha1Digest &sha1() { return sha1_; };

#ifdef DXMT_DEBUG
    void *bytecode;
    size_t bytecode_length;

    virtual void dump() {
      std::fstream dump_out;
      dump_out.open("shader_dump_" + sha1_.string() + ".cso",
                    std::ios::out | std::ios::binary);
      if (dump_out) {
        dump_out.write((char *)bytecode, bytecode_length);
      }
      dump_out.close();
      WARN("shader dumped to ./shader_dump_" + sha1_.string() + ".cso");
    }
#else
    virtual void dump() {}
#endif

    virtual WMT::Reference<WMT::DispatchData> find_cached_variant(Sha1Digest &variant_digest) final {
      auto reader = cache->scache_.getReader();
      if (reader)
        return reader->get(std::make_pair(sha1_, variant_digest));
      return {};
    };
    virtual void update_cached_variant(Sha1Digest &variant_digest, WMT::DispatchData data) final {
      auto writer = cache->scache_.getWriter();
      if (writer)
        writer->set(std::make_pair(sha1_, variant_digest), data);
    }
  };

  class CachedInputLayout final : public InputLayout {
  private:
  public:
    CachedInputLayout(
        std::vector<MTL_SHADER_INPUT_LAYOUT_ELEMENT_DESC> &&attributes,
        uint32_t input_slot_mask)
        : attributes_(attributes), input_slot_mask_(input_slot_mask) {
      Sha1HashState h;
      h.update(input_slot_mask);
      h.update(attributes_.size());
      for (auto &el : attributes_) {
        h.update(el);
      }
      sha1_ = h.final();
    }

    virtual uint32_t input_slot_mask() final { return input_slot_mask_; }

    virtual uint32_t input_layout_element(
        MTL_SHADER_INPUT_LAYOUT_ELEMENT_DESC **ppElements) final {
      *ppElements = attributes_.data();
      return attributes_.size();
    }

    virtual Sha1Digest &sha1() final { return sha1_; }

    std::vector<MTL_SHADER_INPUT_LAYOUT_ELEMENT_DESC> attributes_;
    Sha1Digest sha1_;
    uint32_t input_slot_mask_;
  };

  ShaderCache& scache_;

  task_scheduler<ThreadpoolWork *> scheduler_;

  MTLD3D11Device *device;
  StateObjectCache<D3D11_BLEND_DESC1, IMTLD3D11BlendState> blend_states;

  std::unordered_map<MTL_INPUT_LAYOUT_DESC, std::unique_ptr<CachedInputLayout>> input_layouts;
  dxmt::mutex mutex_ia_;

  StateObjectCache<MTL_STREAM_OUTPUT_DESC, IMTLD3D11StreamOutputLayout>
      so_layouts;
  dxmt::mutex mutex_so_;

  std::unordered_map<Sha1Digest, std::unique_ptr<CachedSM50Shader>> shaders_;
  std::shared_mutex mutex_shares;

  std::unordered_map<MTL_GRAPHICS_PIPELINE_DESC, std::unique_ptr<MTLCompiledGraphicsPipeline>> pipelines_;
  dxmt::mutex mutex_;

  std::unordered_map<MTL_GRAPHICS_PIPELINE_DESC, std::unique_ptr<MTLCompiledGeometryPipeline>> pipelines_gs_;
  dxmt::mutex mutex_gs_;

  std::unordered_map<MTL_GRAPHICS_PIPELINE_DESC, std::unique_ptr<MTLCompiledTessellationMeshPipeline>> pipelines_ts_;
  dxmt::mutex mutex_ts_;

  std::unordered_map<ManagedShader, std::unique_ptr<MTLCompiledComputePipeline>> pipelines_cs_;
  dxmt::mutex mutex_cs_;

  /* neutron: pipeline priming. Measured on an M5 Max (dev/bench/shader_stutter.c, docs/neutron.md):
     newRenderPipelineState is about 90% of the cost of a new shader pair (20 ms against 2 ms for
     airconv), and Metal compiles a pipeline in parts it caches on its own: the vertex function (about
     5 ms, independent of the fragment function and the render state), the fragment function (about
     13 ms, per color formats and blend state, independent of the vertex function) and a link (about
     3 ms). A new VS or PS is therefore compiled in the background, right after its creation, in a
     pipeline with a partner that links with it (the VS created next to it, or one a draw used) and
     with the render state draws used most with as many render targets. The draw that needs the real
     pipeline then finds it compiled (same partner and state) or only pays the link and the parts
     that were guessed wrong. DXMT_CONFIG "d3d11.primePipelines = False" turns it off. */
  struct PrimeEvent {
    CachedSM50Shader *shader;
    ManagedInputLayout layout;
    bool is_vs;
    std::chrono::steady_clock::time_point created;
  };
  struct PrimeTemplate {
    MTL_GRAPHICS_PIPELINE_DESC desc; /* shaders and input layout are null */
    uint32_t rt_mask;
    uint64_t uses;
  };
  class PrimeTask final : public ThreadpoolWork {
  public:
    PipelineCache *cache;
    std::atomic_bool queued = false;
    std::atomic_bool done = false;
    ThreadpoolWork *RunThreadpoolWork() override {
      queued.store(false);
      cache->RunPriming();
      return this;
    }
    bool GetIsDone() override { return done; }
    void SetIsDone(bool state) override { done = state; }
  };
  bool prime_enabled_ = true;
  dxmt::mutex prime_mutex_;
  dxmt::mutex prime_run_mutex_;
  std::vector<PrimeEvent> prime_new_;  /* creation order since the last priming run */
  std::vector<PrimeEvent> prime_left_; /* no partner or no render state yet */
  std::vector<PrimeTemplate> prime_templates_;
  std::deque<std::pair<CachedSM50Shader *, ManagedInputLayout>> prime_known_vs_;
  std::deque<std::pair<CachedSM50Shader *, MTL_GRAPHICS_PIPELINE_DESC>> prime_known_ps_;
  std::unordered_set<MTLCompiledGraphicsPipeline *> prime_speculative_;
  PrimeTask prime_task_;
  uint64_t prime_count_ = 0;
  uint64_t prime_hits_ = 0;
  uint64_t compute_prime_count_ = 0;
  static constexpr uint64_t kMaxPrimedPipelines = 8192;
  static constexpr size_t kMaxPrimeLeft = 4096;
  static constexpr size_t kMaxKnown = 64;
  static constexpr int kPairDistance = 8;
  /* a shader a draw uses within this time is compiled for that draw, with foreground priority: a
     background compile started at utility QoS stays slow in MTLCompilerService even when its worker
     is raised later (shaders created in the frame that draws them: 2130 ms over the frame budget
     without the delay, 1890 ms with it) */
  static constexpr auto kPrimeDelay = std::chrono::milliseconds(50);

  static bool
  Links(CachedSM50Shader *vs, CachedSM50Shader *ps) {
    for (auto &in : ps->link_signature) {
      if (std::find(vs->link_signature.begin(), vs->link_signature.end(), in) == vs->link_signature.end())
        return false;
    }
    return true;
  }

  void
  KickPriming() {
    if (!prime_task_.queued.exchange(true))
      scheduler_.submit_background(&prime_task_);
  }

  void
  AddPrimeEvent(CachedSM50Shader *shader, ManagedInputLayout layout, bool is_vs) {
    if (!prime_enabled_ || shader->used.load() || shader->prime_queued.exchange(true))
      return;
    bool kick;
    {
      std::lock_guard<dxmt::mutex> lock(prime_mutex_);
      prime_new_.push_back({shader, layout, is_vs, std::chrono::steady_clock::now()});
      kick = !prime_templates_.empty();
    }
    if (kick)
      KickPriming();
  }

  /* the render state most pipelines used for pixel shaders with the same render targets; none yet:
     the PS waits for one (the fragment part is compiled per color formats and blend state) */
  const MTL_GRAPHICS_PIPELINE_DESC *
  TemplateFor(CachedSM50Shader *ps) {
    const PrimeTemplate *best = nullptr;
    uint32_t mask = ps->reflection().PixelShader.ValidRenderTargets;
    for (auto &t : prime_templates_) {
      if (t.rt_mask == mask && (!best || t.uses > best->uses))
        best = &t;
    }
    return best ? &best->desc : nullptr;
  }

  /* a draw asked for this pipeline: remember its render state and its shaders as partners */
  void
  RecordPipelineUse(const MTL_GRAPHICS_PIPELINE_DESC &desc) {
    if (!prime_enabled_ || desc.HullShader || desc.DomainShader || desc.GeometryShader || desc.SOLayout ||
        !desc.PixelShader || !desc.RasterizationEnabled)
      return;
    auto vs = static_cast<CachedSM50Shader *>(desc.VertexShader);
    auto ps = static_cast<CachedSM50Shader *>(desc.PixelShader);
    vs->used.store(true);
    ps->used.store(true);
    MTL_GRAPHICS_PIPELINE_DESC t = desc;
    t.VertexShader = nullptr;
    t.PixelShader = nullptr;
    t.InputLayout = nullptr;
    bool kick;
    {
      std::lock_guard<dxmt::mutex> lock(prime_mutex_);
      bool added = false;
      auto it = std::find_if(prime_templates_.begin(), prime_templates_.end(), [&](const PrimeTemplate &x) {
        return std::equal_to<MTL_GRAPHICS_PIPELINE_DESC>{}(x.desc, t);
      });
      if (it != prime_templates_.end()) {
        it->uses++;
      } else if (prime_templates_.size() < 64) {
        added = true;
        uint32_t rt_mask = 0;
        for (unsigned i = 0; i < t.NumColorAttachments; i++)
          if (t.ColorAttachmentFormats[i] != WMTPixelFormatInvalid)
            rt_mask |= 1u << i;
        prime_templates_.push_back({t, rt_mask, 1});
      }
      if (prime_known_vs_.size() >= kMaxKnown)
        prime_known_vs_.pop_back();
      prime_known_vs_.push_front({vs, desc.InputLayout});
      if (prime_known_ps_.size() >= kMaxKnown)
        prime_known_ps_.pop_back();
      prime_known_ps_.push_front({ps, t});
      /* events wait for the first render state; leftovers retry with a new one */
      kick = !prime_new_.empty() || (added && !prime_left_.empty());
    }
    if (kick)
      KickPriming();
  }

  void
  RunPriming() {
    std::lock_guard<dxmt::mutex> run_lock(prime_run_mutex_);
    std::vector<MTL_GRAPHICS_PIPELINE_DESC> descs;
    {
      std::lock_guard<dxmt::mutex> lock(prime_mutex_);
      if (prime_templates_.empty())
        return;
      std::vector<PrimeEvent> events;
      events.swap(prime_left_);
      /* events younger than kPrimeDelay stay for a later run (a new pipeline or shader kicks it) */
      auto now = std::chrono::steady_clock::now();
      size_t old = 0;
      while (old < prime_new_.size() && now - prime_new_[old].created >= kPrimeDelay)
        old++;
      events.insert(events.end(), prime_new_.begin(), prime_new_.begin() + old);
      prime_new_.erase(prime_new_.begin(), prime_new_.begin() + old);
      std::vector<bool> taken(events.size(), false);
      auto make = [&](CachedSM50Shader *vs, ManagedInputLayout layout, CachedSM50Shader *ps,
                      const MTL_GRAPHICS_PIPELINE_DESC *t) {
        MTL_GRAPHICS_PIPELINE_DESC d = *t;
        d.VertexShader = vs;
        d.PixelShader = ps;
        d.InputLayout = layout;
        descs.push_back(d);
      };
      /* a PS and the closest VS created next to it that links with it (engines create a
         material's shaders together) */
      for (int i = 0; i < (int)events.size(); i++) {
        if (taken[i] || events[i].is_vs)
          continue;
        auto ps = events[i].shader;
        if (ps->used.load()) {
          taken[i] = true;
          continue;
        }
        for (int d = 1; d <= kPairDistance; d++) {
          int j = -1;
          if (i - d >= 0 && !taken[i - d] && events[i - d].is_vs && Links(events[i - d].shader, ps))
            j = i - d;
          else if (i + d < (int)events.size() && !taken[i + d] && events[i + d].is_vs &&
                   Links(events[i + d].shader, ps))
            j = i + d;
          if (j < 0)
            continue;
          if (auto t = TemplateFor(ps)) {
            make(events[j].shader, events[j].layout, ps, t);
            taken[i] = taken[j] = true;
          }
          break;
        }
      }
      /* the rest with a partner a draw used */
      for (unsigned i = 0; i < events.size(); i++) {
        if (taken[i] || events[i].shader->used.load())
          continue;
        auto &e = events[i];
        bool done = false;
        if (e.is_vs) {
          for (auto &[ps, t] : prime_known_ps_) {
            if (Links(e.shader, ps)) {
              make(e.shader, e.layout, ps, &t);
              done = true;
              break;
            }
          }
        } else if (auto t = TemplateFor(e.shader)) {
          for (auto &[vs, layout] : prime_known_vs_) {
            if (Links(vs, e.shader)) {
              make(vs, layout, e.shader, t);
              done = true;
              break;
            }
          }
        }
        if (!done && prime_left_.size() < kMaxPrimeLeft)
          prime_left_.push_back(e);
      }
    }
    for (auto &d : descs) {
      std::lock_guard<dxmt::mutex> lock(mutex_);
      if (prime_count_ >= kMaxPrimedPipelines)
        break;
      if (pipelines_.contains(d))
        continue;
      auto [iter, inserted] = pipelines_.insert({d, CreateGraphicsPipeline(device, &d, true)});
      prime_speculative_.insert(iter->second.get());
      scheduler_.submit_background(iter->second.get());
      prime_count_++;
    }
  }

  /* a draw needs a pipeline that may still wait in the background queue */
  void
  Promote(ThreadpoolWork *work) {
    if (work->GetIsDone())
      return;
    scheduler_.promote(work);
    for (unsigned i = 0; i < 2; i++) {
      if (auto dep = work->GetDependency(i); dep && !dep->GetIsDone())
        scheduler_.promote(dep);
    }
  }

  CachedSM50Shader *CreateShader(const void *pBytecode,
                                 uint32_t BytecodeLength, bool *created = nullptr) {
    if (created)
      *created = false;
    auto sha1 = Sha1HashState::compute(pBytecode, BytecodeLength);
    {
      std::shared_lock<std::shared_mutex> lock(mutex_shares);
      auto result = shaders_.find(sha1);
      if (result != shaders_.end()) {
        return shaders_.at(sha1).get();
      }
    }
    sm50_error_t err;
    sm50_shader_t sm50;
    MTL_SHADER_REFLECTION reflection;
    if (SM50Initialize(pBytecode, BytecodeLength, &sm50, &reflection, &err)) {
      ERR("Failed to initialize shader: ", SM50GetErrorMessageString(err));
      SM50FreeError(err);
      return nullptr;
    }
    auto shader = std::make_unique<CachedSM50Shader>(this, sm50, sha1, reflection);
    shader->program_type = DxbcProgramType(pBytecode, BytecodeLength);
    if (shader->program_type == 1)
      shader->link_signature = DxbcSignature(pBytecode, BytecodeLength, true);
    else if (shader->program_type == 0) {
      shader->link_signature = DxbcSignature(pBytecode, BytecodeLength, false);
      shader->ps_outputs_known =
          DxbcPixelOutputs(pBytecode, BytecodeLength, &shader->ps_writes_depth, &shader->ps_float_targets);
      if (!shader->ps_outputs_known) {
        shader->ps_writes_depth = true;
        shader->ps_float_targets = ~0u;
      }
    }
    {
      std::unique_lock<std::shared_mutex> lock(mutex_shares);
      auto result = shaders_.find(sha1);
      if (result != shaders_.end()) {
        return shaders_.at(sha1).get();
      }
#ifdef DXMT_DEBUG
      shader->bytecode = malloc(BytecodeLength);
      shader->bytecode_length = BytecodeLength;
      memcpy(shader->bytecode, pBytecode, BytecodeLength);
#endif
      if (created)
        *created = true;
      return shaders_.emplace(sha1, std::move(shader)).first->second.get();
    }
  }

  virtual HRESULT AddVertexShader(const void *pBytecode,
                                  uint32_t BytecodeLength,
                                  ID3D11VertexShader **ppShader) override {
    bool created;
    auto managed_shader = CreateShader(pBytecode, BytecodeLength, &created);
    if (!managed_shader) {
      return E_FAIL;
    }
    /* neutron: a VS without vertex inputs needs no input layout: convert it now, in the background,
       and prime its pipeline (with input it waits for CreateInputLayout) */
    if (created && prime_enabled_ && managed_shader->program_type == 1 &&
        DxbcSignature(pBytecode, BytecodeLength, false).empty()) {
      managed_shader->get_shader(ShaderVariantVertex{nullptr, ~0u, false}, true);
      AddPrimeEvent(managed_shader, nullptr, true);
    }
    *ppShader =
        ref(new TShaderBase<ID3D11VertexShader, MTLD3D10VertexShader>(device, managed_shader));
    return S_OK;
  }

  virtual HRESULT AddPixelShader(const void *pBytecode, uint32_t BytecodeLength,
                                 ID3D11PixelShader **ppShader) override {
    bool created;
    auto managed_shader = CreateShader(pBytecode, BytecodeLength, &created);
    if (!managed_shader) {
      return E_FAIL;
    }
    /* neutron: its variant depends on the render targets, it is converted with its primed pipeline */
    if (created && managed_shader->program_type == 0)
      AddPrimeEvent(managed_shader, nullptr, false);
    *ppShader = ref(new TShaderBase<ID3D11PixelShader, MTLD3D10PixelShader>(device, managed_shader));
    return S_OK;
  }

  virtual HRESULT AddHullShader(const void *pBytecode, uint32_t BytecodeLength,
                                ID3D11HullShader **ppShader) override {
    auto managed_shader = CreateShader(pBytecode, BytecodeLength);
    if (!managed_shader) {
      return E_FAIL;
    }
    *ppShader = ref(new TShaderBase<ID3D11HullShader>(device, managed_shader));
    return S_OK;
  }

  virtual HRESULT AddDomainShader(const void *pBytecode,
                                  uint32_t BytecodeLength,
                                  ID3D11DomainShader **ppShader) override {
    auto managed_shader = CreateShader(pBytecode, BytecodeLength);
    if (!managed_shader) {
      return E_FAIL;
    }
    *ppShader =
        ref(new TShaderBase<ID3D11DomainShader>(device, managed_shader));
    return S_OK;
  }

  virtual HRESULT AddGeometryShader(const void *pBytecode,
                                    uint32_t BytecodeLength,
                                    ID3D11GeometryShader **ppShader) override {
    auto managed_shader = CreateShader(pBytecode, BytecodeLength);
    if (!managed_shader) {
      return E_FAIL;
    }
    *ppShader =
        ref(new TShaderBase<ID3D11GeometryShader, MTLD3D10GeometryShader>(device, managed_shader));
    return S_OK;
  }

  virtual HRESULT AddComputeShader(const void *pBytecode,
                                   uint32_t BytecodeLength,
                                   ID3D11ComputeShader **ppShader) override {
    bool created;
    auto managed_shader = CreateShader(pBytecode, BytecodeLength, &created);
    if (!managed_shader) {
      return E_FAIL;
    }
    /* neutron: a compute pipeline depends on nothing but the shader: compile it now, in the
       background, instead of at the first Dispatch */
    if (created && prime_enabled_ && managed_shader->program_type == 5) {
      std::lock_guard<dxmt::mutex> lock(mutex_cs_);
      auto [iter, inserted] =
          pipelines_cs_.insert({managed_shader, CreateComputePipeline(device, managed_shader, true)});
      if (inserted) {
        scheduler_.submit_background(iter->second.get());
        compute_prime_count_++;
      }
    }
    *ppShader =
        ref(new TShaderBase<ID3D11ComputeShader>(device, managed_shader));
    return S_OK;
  }

  HRESULT AddInputLayout(const void *pShaderBytecodeWithInputSignature,
                         const D3D11_INPUT_ELEMENT_DESC *pInputElementDesc,
                         UINT NumElements,
                         IMTLD3D11InputLayout **ppInputLayout) override {
    std::lock_guard<dxmt::mutex> lock(mutex_ia_);
    std::vector<MTL_SHADER_INPUT_LAYOUT_ELEMENT_DESC> buffer(NumElements);
    uint32_t num_metal_ia_elements;
    HRESULT hr;
    if (FAILED(hr = ExtractMTLInputLayoutElements(
            device, pShaderBytecodeWithInputSignature, pInputElementDesc,
            NumElements, buffer.data(), &num_metal_ia_elements))) {
      return hr;
    }
    buffer.resize(num_metal_ia_elements);
    if (!input_layouts.contains(buffer)) {
      uint32_t input_slot_mask = 0;
      for (auto &element : buffer) {
        input_slot_mask |= (1 << element.Slot);
      }
      input_layouts.emplace(buffer, std::make_unique<CachedInputLayout>(
                                        std::move(buffer), input_slot_mask));
    }
    auto layout = input_layouts.at(buffer).get();
    *ppInputLayout = ref(new MTLD3D11InputLayout(device, layout));
    /* neutron: the layout and the VS it was created with give the exact VS variant a draw will use */
    if (prime_enabled_) {
      auto p = (const uint8_t *)pShaderBytecodeWithInputSignature;
      uint32_t size = 0;
      if (!memcmp(p, "DXBC", 4))
        memcpy(&size, p + 24, 4);
      if (size >= 32) {
        auto sha1 = Sha1HashState::compute(pShaderBytecodeWithInputSignature, size);
        CachedSM50Shader *vs = nullptr;
        {
          std::shared_lock<std::shared_mutex> lock(mutex_shares);
          if (auto it = shaders_.find(sha1); it != shaders_.end())
            vs = it->second.get();
        }
        if (vs && vs->program_type == 1 && !vs->used.load() && !vs->prime_queued.load()) {
          vs->get_shader(ShaderVariantVertex{layout, ~0u, false}, true);
          AddPrimeEvent(vs, layout, true);
        }
      }
    }
    return hr;
  }

  HRESULT
  AddStreamOutputLayout(const void *pShaderBytecode, UINT NumEntries,
                        const D3D11_SO_DECLARATION_ENTRY *pEntries,
                        UINT NumStrides, const UINT *pStrides,
                        UINT RasterizedStream,
                        IMTLD3D11StreamOutputLayout **ppSOLayout) override {
    std::lock_guard<dxmt::mutex> lock(mutex_so_);
    std::vector<MTL_SHADER_STREAM_OUTPUT_ELEMENT_DESC> buffer(NumEntries * 4);
    std::array<uint32_t, 4> strides = {{}};
    uint32_t num_metal_so_elements;
    if (FAILED(ExtractMTLStreamOutputElements(
            device, pShaderBytecode, NumEntries, pEntries, buffer.data(),
            &num_metal_so_elements))) {
      return E_FAIL;
    }
    buffer.resize(num_metal_so_elements);
    for (unsigned i = 0; i < NumStrides; i++) {
      strides[i] = pStrides[i];
    }
    MTL_STREAM_OUTPUT_DESC desc;
    memcpy(desc.Strides, strides.data(), sizeof(strides));
    desc.Elements = std::move(buffer);
    desc.RasterizedStream = RasterizedStream;
    return so_layouts.CreateStateObject(&desc, ppSOLayout);
  };

  HRESULT AddBlendState(const D3D11_BLEND_DESC1 *pBlendDesc,
                        IMTLD3D11BlendState **ppBlendState) override {
    return blend_states.CreateStateObject(pBlendDesc, ppBlendState);
  }

  void GetGraphicsPipeline(MTL_GRAPHICS_PIPELINE_DESC *pDesc,
                           MTLCompiledGraphicsPipeline **ppPipeline) override {
    {
      std::lock_guard<dxmt::mutex> lock(mutex_);

      if (auto found = pipelines_.find(*pDesc); found != pipelines_.end()) {
        *ppPipeline = found->second.get();
        /* neutron: primed in the background: move it to the front */
        if (prime_speculative_.empty() || !prime_speculative_.erase(*ppPipeline))
          return;
        prime_hits_++;
        Promote(*ppPipeline);
      } else {
        /* neutron: shaders converted in the background (CreateInputLayout) move to the front with
           the pipeline when it waits for them, see task_scheduler::worker_func */
        auto [iter, inserted] = pipelines_.insert({*pDesc, CreateGraphicsPipeline(device, pDesc)});
        if (!inserted) {
          D3D11_ASSERT(0 && "duplicated graphics pipeline");
        } else {
          scheduler_.submit(iter->second.get());
        }
        *ppPipeline = iter->second.get();
      }
    }
    RecordPipelineUse(*pDesc);
  }

  void GetGeometryPipeline(
      MTL_GRAPHICS_PIPELINE_DESC *pDesc,
      MTLCompiledGeometryPipeline **ppPipeline) override {
    std::lock_guard<dxmt::mutex> lock(mutex_gs_);

    if (auto iter = pipelines_gs_.find(*pDesc); iter != pipelines_gs_.end()) {
      *ppPipeline = iter->second.get();
      return;
    }
    auto [iter, inserted] = pipelines_gs_.insert({*pDesc, CreateGeometryPipeline(device, pDesc)});
    if (!inserted) {
      D3D11_ASSERT(0 && "duplicated geometry pipeline");
    } else {
      scheduler_.submit(iter->second.get());
    }
    *ppPipeline = iter->second.get();
  }

  void GetTessellationPipeline(MTL_GRAPHICS_PIPELINE_DESC * pDesc,
                                   MTLCompiledTessellationMeshPipeline *
                                       *ppPipeline) override {
    std::lock_guard<dxmt::mutex> lock(mutex_ts_);

    if (auto iter = pipelines_ts_.find(*pDesc); iter != pipelines_ts_.end()) {
      *ppPipeline = iter->second.get();
      return;
    }
    auto [iter, inserted] = pipelines_ts_.insert({*pDesc, CreateTessellationMeshPipeline(device, pDesc)});
    if (!inserted) {
      D3D11_ASSERT(0 && "duplicated tessellation pipeline");
    } else {
      scheduler_.submit(iter->second.get());
    }
    *ppPipeline = iter->second.get();
  }

  void GetComputePipeline(MTL_COMPUTE_PIPELINE_DESC *pDesc,
                                  MTLCompiledComputePipeline **ppPipeline) override {
   std::lock_guard<dxmt::mutex> lock(mutex_cs_);

    if (auto iter = pipelines_cs_.find(pDesc->ComputeShader); iter != pipelines_cs_.end()) {
      *ppPipeline = iter->second.get();
      /* neutron: compiled in the background since CreateComputeShader */
      Promote(*ppPipeline);
      return;
    }
    auto [iter, inserted] = pipelines_cs_.insert({pDesc->ComputeShader, CreateComputePipeline(device, pDesc->ComputeShader)});
    if (!inserted) {
      D3D11_ASSERT(0 && "duplicated compute pipeline");
    } else {
      scheduler_.submit(iter->second.get());
    }
    *ppPipeline = iter->second.get();
  }

public:
  PipelineCache(MTLD3D11Device *pDevice) :
      scache_(ShaderCache::getInstance(pDevice->GetDXMTDevice().metalVersion())),
      device(pDevice),
      blend_states(pDevice),
      so_layouts(pDevice) {
    prime_enabled_ = Config::getInstance().getOption<bool>("d3d11.primePipelines", true);
    prime_task_.cache = this;
  };

  ~PipelineCache() {
    /* neutron: workers run tasks that point into the maps below (destroyed before scheduler_) */
    scheduler_.shutdown();
    if (prime_count_ || compute_prime_count_)
      Logger::info(str::format("Pipeline priming: ", prime_count_, " primed, ", prime_hits_, " used by draws, ",
                               compute_prime_count_, " compute pipelines at shader creation"));
  }
};

std::unique_ptr<MTLD3D11PipelineCacheBase>
InitializePipelineCache(MTLD3D11Device *device) {
  return std::make_unique<PipelineCache>(device);
}

}; // namespace dxmt