# Perf geometry shaders WARNO — notes de travail (branche `perf-gs`)

Objectif : Shaders Élevé ≥ 60 fps sur la scène Hesse (déploiement, 2560×1600, 3D 1920), rendu identique.
Les optimisations restent dans le code commun (airconv + d3d11), sans rien de spécifique x86_64 (ARM64EC prévu).

## 1. Comment DXMT émule un GS (lu dans le code, base `eb1a631f`)

Fichiers : `src/airconv/dxbc_converter_gs.cpp`, `src/d3d11/d3d11_pipeline_gs.cpp`,
`src/d3d11/d3d11_context_impl.cpp` (`GeometryDraw*`), `src/winemetal/unix/winemetal_unix.c` (`WMTRenderCommandDXMTGeometryDraw*`).

- Pipeline mesh Metal : **object shader = VS D3D**, **mesh shader = GS D3D**.
- Object : un threadgroup par « warp » = paquet de sommets (30 pour une liste de triangles, 32 pour des points).
  Un thread par sommet, il exécute le VS et écrit ses sorties dans le payload.
  Grille object = `(warp_count, InstanceCount, 1)`, `warp_count = ceil(IndexCount / 30)`.
- Payload : **16 256 octets fixes** (`payload_memory_length`, `InputPayload{.size = 16256}`), quel que soit le besoin réel
  (16 o d'en-tête + sommets × registres de sortie VS × 16 o).
- Mesh : chaque object threadgroup lance une grille `(warp_primitive_count, gs_instance_count, 1)`,
  soit 10 mesh threadgroups pour une liste de triangles, **1 thread par threadgroup** (`threadsPerMeshThreadgroup = (1,1,1)`).
  Un mesh threadgroup = une invocation du GS pour une primitive ; il sort tout de suite si la primitive est invalide
  (masque `valid_primitive_mask` du payload).
- Mesh déclaré `InputMesh{max_vertex_out, max_vertex_out, topology}` : max primitives = `maxvertexcount`.

Conséquences :
- Le GS tourne en scalaire : 1 thread utile par threadgroup (pas de remplissage SIMD).
- Un draw instancié de 6 indices lance, par instance, 1 object threadgroup de 30 threads (6 utiles)
  et 10 mesh threadgroups (2 utiles).

## 2. Ce que WARNO envoie (probe, scène Hesse, Shaders Moyen, 21,5 fps)

Données brutes : `~/code/dxmt-gs-data/gsprobe/` (`pairs.txt`, `stats.txt`, `<sha1>.cso` des VS et GS). Hors repo : bytecode du jeu.

- 16 pipelines GS, **4 GS distincts** : `499930125e9b`, `ce4706da3100`, `e9bce06f16b5`, `73747483a028`.
- **Tous sans render target couleur, profondeur seule** (`rtv=0`, `ds=260`), avec un PS : ce sont des passes d'ombre / profondeur.
- Tous en `TRIANGLELIST` (topo 4), indexés (index 16 bits), sauf `73747483a028` (non indexé, pas vu dans les stats de draws).
- Total : ~2 112 draws GS par frame (recoupe le HUD D3DMetal : 2 109).

Par frame (stats sur fenêtres de 10 s, divisées par 21,5 fps) :

| VS + GS | draws/frame | indices/frame | instances/draw | object TG/frame | mesh TG/frame |
|---|---|---|---|---|---|
| `ee4035d16fea` + `e9bce06f16b5` | 203 | 7,0 M | **5 759** (6 indices = 1 quad) | **1,17 M** | **11,7 M** (2,3 M utiles) |
| `75b4cf05cb4c` + `499930125e9b` | 1 450 | 2,16 M | 1 | ~73 k | ~730 k |
| `e1ad7892ae4b` + `499930125e9b` | 133 | 0,66 M | 41 | ~27 k | ~270 k |
| `ea3b4bf99bdd` + `ce4706da3100` | 207 | 0,30 M | 1 | ~10 k | ~100 k |
| `5e6a21a321fb` + `ce4706da3100` | 55 | 0,25 M | 1 | ~8 k | ~80 k |
| 8 autres paires | ~65 | ~0,5 M | 1 à 10 | ~17 k | ~170 k |

Lecture : le draw instancié de quads (`e9bce06f16b5`, 203 draws × 5 759 instances) pèse à lui seul
~90 % des threadgroups. Il gaspille 24 threads object sur 30 et 8 mesh threadgroups sur 10.

## 3. Mesures (HUD Metal, OCR, 3 lectures)

| Config | FPS | GPU ms | Remarque |
|---|---|---|---|
| Shaders Moyen, build probe (`eb1a631f` + probe) | 21,58 | 46,4 | ioreg Device 99 %, Renderer 65 % |
| Shaders Élevé, basculé en jeu (Options → Appliquer) | 21,56 | 46,5 | mêmes draws GS, aucun nouveau shader GS |

- Élevé basculé en jeu = Moyen sur cette scène, pour le coût GS. À refaire en chargeant directement avec `ShaderQuality = 0`.
- Références du coordinateur : Bas 134-150 fps (GPU ~6 ms) ; Moyen 21,7 fps ; canal Vertex 40 ms/frame en Moyen.

## 4. Le probe patch (branche `gs-probe`, commit `e9b56486`, poussée, à ne pas fusionner)

- `dump()` rendu inconditionnel : écrit `<sha1>.cso` pour le VS et le GS de chaque pipeline GS créé.
- `pairs.txt` : une ligne par pipeline GS (sha VS, sha GS, strip, format d'index, sha PS, nb RTV, format depth).
- `stats.txt` : toutes les 10 s, par paire VS+GS : draws/s, indices/s, instances/draw, max indices, topologie, histogramme.
- Chemin de sortie codé en dur (`GS_PROBE_DIR` dans `d3d11_shader.hpp`) : pas d'env possible sans toucher à l'Info.plist.
- Les draws GS indirects ne sont pas comptés (WARNO n'en fait pas d'après le HUD).

## 5. Pistes, par gain attendu

1. **Draws instanciés à petit nombre d'indices** : regrouper plusieurs instances dans un object threadgroup
   (aujourd'hui 1 warp de 6 sommets par instance). Vise le poste à 1,17 M object TG.
2. **Ne lancer que les mesh threadgroups utiles** : `set_threadgroups_per_grid` = nombre de primitives valides du warp, pas 10 fixes.
3. **Payload dimensionné au besoin réel** au lieu de 16 256 o.
4. Plusieurs primitives par mesh threadgroup (threads en parallèle) si `maxvertexcount` le permet (limite Metal : 256 sommets, 512 primitives).
5. Chemin rapide si le GS est un passthrough (à confirmer par le DXBC).

## 6. Reste à faire

- [ ] Désassembler les 4 GS : `maxvertexcount`, topologie de sortie, instancing GS, sorties système (`SV_RenderTargetArrayIndex`, `SV_ViewportArrayIndex`).
- [ ] Coder les pistes 1 à 3, mesurer dos à dos contre le build de base (même session, ≤ 10 min de verrou).
- [ ] Mesure Élevé en chargement direct, captures avant/après.

## 7. Banc

- Build : `scratchpad/mkpayload.sh <dest>` (compile + install build64/build32, assemble `wine/…`). Payload de base : build de `eb1a631f`.
- Session jeu : `game.sh lock gs` → install → `load_save.sh` → `bench.sh` → `restore` + `stop` + Option.ini validé + `METAL_HUD=0` → `unlock gs`. 10 min max.
- Le menu pause s'ouvre bien avec Échap depuis l'écran de déploiement (le briefing reste affiché derrière).
