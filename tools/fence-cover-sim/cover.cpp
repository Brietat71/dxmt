// Checks, with DXMT's real dependency tracker, that every true dependency of an encoder is covered (directly or
// transitively) by the fence waits it encodes. A wait on bit b designates, as it does once encoded with the fence
// pool, the last previous encoder whose id is b modulo kParityLane.
//
// build: clang++ -std=c++20 -O2 -I../../src/dxmt -I../../src/util cover.cpp ../../src/dxmt/dxmt_deptrack.cpp -o cover
// usage: cover [seed=1] [encoders=3000] [resources=40]   (exit code 1 if a dependency is not covered)
//        for s in $(seq 1 60); do ./cover $s 4000 40 | tail -1; done
#include "dxmt_deptrack.hpp"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
using namespace dxmt;
struct Node { EncoderId id; FenceSet wait; int enc; std::vector<int> deps; }; // deps = indices of the nodes it truly depends on
int main(int argc, char **argv) {
  unsigned seed = argc > 1 ? atoi(argv[1]) : 1; int N = argc > 2 ? atoi(argv[2]) : 3000; int R = argc > 3 ? atoi(argv[3]) : 40;
  std::mt19937 rng(seed);
  std::vector<GenericAccessTracker> res(R);
  struct RS { int last_writer = -1; std::vector<int> readers; }; std::vector<RS> rs(R);
  std::vector<Node> nodes; FenceLocalityCheck loc; EncoderId next = kParityLane;
  auto pick = [&](int n) { return (int)(rng() % n); };
  // "cold" resources are rarely touched, so that their last writer is often very old (a target written once per frame)
  auto res_index = [&]() { return pick(10) == 0 ? R / 2 + pick(R / 2) : pick(R / 2); };
  for (int e = 0; e < N; e++) {
    bool render = pick(3) != 0;
    EncoderBarrierState bs{};
    if (render) {
      EncoderId v = next++, f = next++;
      Node nv{v, {}, e, {}}, nf{f, {}, e, {}};
      int iv = nodes.size(), ifr = iv + 1;
      int na = 1 + pick(4);
      for (int a = 0; a < na; a++) {
        int r = res_index(); int kind = pick(8);
        auto &t = res[r]; auto &s = rs[r];
        if (kind < 4) { // fragment read
          t.accessSharedFragment(f, nf.wait, bs);
          if (s.last_writer >= 0 && nodes.size() > (size_t)s.last_writer && s.last_writer < iv) nf.deps.push_back(s.last_writer);
          s.readers.push_back(ifr);
        } else if (kind < 7) { // fragment write (attachment)
          t.accessExclusiveFragment(f, nf.wait, bs, false);
          if (s.last_writer >= 0 && s.last_writer < iv) nf.deps.push_back(s.last_writer);
          for (int rd : s.readers) if (rd < iv) nf.deps.push_back(rd);
          s.last_writer = ifr; s.readers.clear();
        } else { // pre-raster read
          t.accessSharedPreRaster(v, nv.wait, bs);
          if (s.last_writer >= 0 && s.last_writer < iv) nv.deps.push_back(s.last_writer);
          s.readers.push_back(iv);
        }
      }
      nv.wait = loc.collectAndSimplifyWaits(nv.wait, v);
      nf.wait = loc.collectAndSimplifyWaits(nf.wait, f, true);
      nodes.push_back(nv); nodes.push_back(nf);
    } else {
      EncoderId id = next++; Node n{id, {}, e, {}}; int in = nodes.size();
      int na = 1 + pick(3);
      for (int a = 0; a < na; a++) {
        int r = res_index(); auto &t = res[r]; auto &s = rs[r];
        if (pick(2)) {
          t.accessShared(id, n.wait, bs);
          if (s.last_writer >= 0 && s.last_writer < in) n.deps.push_back(s.last_writer);
          s.readers.push_back(in);
        } else {
          t.accessExclusive(id, n.wait, bs, false);
          if (s.last_writer >= 0 && s.last_writer < in) n.deps.push_back(s.last_writer);
          for (int rd : s.readers) if (rd < in) n.deps.push_back(rd);
          s.last_writer = in; s.readers.clear();
        }
      }
      n.wait = loc.collectAndSimplifyWaits(n.wait, id);
      nodes.push_back(n);
    }
  }
  // transitive closure of the waits
  int M = nodes.size(); std::vector<std::vector<bool>> done(M, std::vector<bool>(M, false));
  EncoderId first = kParityLane;
  long violations = 0, checked = 0, far = 0;
  for (int i = 0; i < M; i++) {
    auto &n = nodes[i];
    auto wait_on = [&](int w) { if (w < 0 || w >= i) return; if (!done[i][w]) { done[i][w] = true; for (int k = 0; k < w; k++) if (done[w][k]) done[i][k] = true; } };
    n.wait.forEach([&](EncoderId bit) {
      EncoderId cand = (n.id & ~(EncoderId)(kParityLane - 1)) + bit; // same generation
      while (cand >= n.id) cand -= kParityLane;                     // last previous id congruent to the bit
      if (cand >= first) wait_on((int)(cand - first));              // ids are consecutive: index = id - first
    });
    if (i > 0 && nodes[i - 1].enc == n.enc) wait_on(i - 1);         // the fragment stage follows the pre-raster stage of its encoder
    for (int d : n.deps) {
      if (nodes[d].enc == n.enc) continue;
      checked++; if (n.id - nodes[d].id >= kLane) far++;
      if (!done[i][d]) { violations++; if (violations <= 5) printf("  not covered: node id=%llu depends on id=%llu (distance %llu)\n", (unsigned long long)n.id, (unsigned long long)nodes[d].id, (unsigned long long)(n.id - nodes[d].id)); }
    }
  }
  printf("seed %u: %d nodes, %ld dependencies checked (%ld of them 64 ids away or more), %ld NOT COVERED\n", seed, M, checked, far, violations);
  return violations != 0;
}
