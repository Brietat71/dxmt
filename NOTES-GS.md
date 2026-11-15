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

## 5. Ce que font les GS de WARNO (IR produit par `airconv -S` sur les `.cso`)

Les 4 GS sont le même motif : **répartition des triangles dans les cascades d'ombre** (rendu en couches).

- Entrée : triangle. Sortie : triangle strip, **`maxvertexcount = 9`** (3 triangles), pas d'instancing GS, pas de stream.
- Par triangle : 3 produits scalaires par sommet (matrice lumière dans `cb0`), boîte englobante 2D,
  puis boucle sur 4 cascades. Les bornes de chaque cascade sont lues dans un buffer d'entiers `t0` (8 entiers par cascade).
  Si la boîte recoupe la cascade : 3 `emit` + `cut`, position recalée dans la cascade, `SV_RenderTargetArrayIndex = i`.
- `499930125e9b` : position seule (VS à 1 registre de sortie). Terrain, objets.
- `e9bce06f16b5`, `ce4706da3100`, `73747483a028` : position + un TEXCOORD recopié (alpha test dans le PS). VS à 2 registres.
- Le VS `ee4035d16fea` du gros draw instancié est un **billboard** : coin du quad par sommet, position/taille/UV par instance,
  un échantillonnage de texture. Ce sont les arbres (imposteurs) rendus dans les ombres : 203 draws × 5 759 instances = 1,17 M de quads par frame.
- Un triangle au centre peut recouper les 4 cascades : 12 sommets émis pour 9 déclarés. D3D ignore l'excédent ; l'ancien mapping
  le perdait par débordement du mesh (couche 3 vide dans le banc), le nouveau le tronque explicitement. Même image.

## 6. Où part le temps (banc synthétique MSL, `tools/gs-bench/bench.swift`, GPU au repos)

Charge B = 203 draws × 5 759 instances de 6 indices. Charge A = 1 450 draws × 498 triangles.

| Variante | B (ms GPU) | A (ms GPU) |
|---|---|---|
| D'origine (payload 16 256 o, 10 mesh TG de 1 thread par warp) | **33,7** | 2,5 |
| Payload au plus juste seul | 33,3 | 2,8 |
| Grille mesh = primitives valides, payload 16 256 o | 22,0 | |
| Grille mesh = primitives valides, payload juste | 8,2 | 2,9 |
| 1 mesh TG de 10 threads + préfixe 2 passes, payload 16 256 o | pas de gain | |
| 1 mesh TG de 10 threads + préfixe 2 passes, payload juste | 8,2 | 2,3 |
| Paquets d'instances seuls (5 instances par object TG) | 7,3 | |
| **Paquets + 1 mesh TG de 10 threads + payload juste** | **2,9** | 2,3 |

- Le modèle reproduit l'ordre de grandeur du jeu : 36 ms synthétiques pour 40 ms de canal Vertex mesurés.
- **Presque tout le coût est le draw instancié de billboards**, pas le terrain.
- Les trois leviers se multiplient. Le payload de 16 Ko annule le gain du regroupement des threads s'il reste.
- Variante atomiques (ordre non déterministe) : 2,6 ms au lieu de 2,9. Écartée : l'ordre des primitives doit rester celui de D3D.

## 7. Changements faits (branche `perf-gs`)

1. **Payload au plus juste** : `SM50GSPayloadSize()` = 16 + 32 × 16 × registres de sortie du VS (528 o ou 1 040 o pour WARNO, au lieu de 16 256).
   Convertisseur (object et mesh) et `payload_memory_length` du pipeline.
2. **GS regroupés par mesh threadgroup** : `SM50GSMeshThreads()` = min(primitives du warp, 256 / (maxvertexcount + 1)) threads,
   un thread par invocation du GS (10 pour WARNO au lieu de 1).
   - Le corps du GS est converti deux fois. Passe 1 : compter sommets et primitives émis. Somme préfixe sur le threadgroup
     (mémoire threadgroup + barrière). Passe 2 : écrire dans la tranche du thread.
   - Le mesh reste compact et **dans l'ordre des primitives d'entrée** : même ordre de rendu qu'avant.
   - Les écritures spéculatives (sommet émis avant de savoir si son strip est valide) qui sortent de la tranche vont dans
     un sommet/une primitive « poubelle » par thread, hors du compte de primitives.
   - `maxvertexcount` est appliqué : les `emit` au-delà sont ignorés.
   - Pas de mesh threadgroup du tout si le warp n'a aucune primitive complète.
   - GS DXIL : laissés à 1 thread (chemin non exercé), marqué `ponytail:` dans `gs_mesh_threads()`.
3. **Paquets d'instances** : `SM50GSInstancesPerWarp()`. Pour un draw en liste de `C` sommets avec `C ≤ warp/2` et `C` multiple
   de la taille d'une primitive, un object threadgroup traite `warp / C` instances (5 quads au lieu de 1).
   - Object shader : thread → (instance, sommet). `SV_InstanceID`, le tirage des vertex buffers par instance et `SV_PrimitiveID` suivent.
   - Grille de dispatch : `(1, ceil(instances / K))`. Draws directs dans `GeometryDraw*`, draws indirects dans le marshal GPU
     (`gs_draw_arguments_marshal`), même formule aux trois endroits.
4. `AIRCONV_VERSION` 26 → 27 : invalide le cache de shaders (l'ancien AIR est incompatible avec le nouveau dispatch).
5. `airconv` (CLI de dev) : `--ia-layout`, `--ia-index-format`, `--gs-strip`, pour produire hors-jeu les shaders tels que le runtime les compile.

Rien de spécifique à une architecture : tout est dans le convertisseur, l'encodeur de draws et le shader de marshal.

## 8. Vérification hors-jeu (`tools/gs-bench/real.swift`, vrais shaders de WARNO)

Tampon de profondeur 4 couches comparé **au bit près** entre l'ancien convertisseur + ancien dispatch et le nouveau :

| Scénario | Ancien (ms GPU) | Nouveau (ms GPU) | Profondeur |
|---|---|---|---|
| A : 1 450 draws × 498 triangles | 4,1 | 1,45 | identique |
| B : 203 draws × 5 759 billboards | 40,5 | 5,1 | identique |
| C : indices 1 à 60, instances 1 à 300 (paquets, warps partiels) | 8,3 | 2,6 | identique |
| S : triangle strip (25 threads, 2 mesh TG par warp) | 8,4 | 2,7 | identique |

- A et B mesurés GPU au repos ; C et S pendant qu'un autre agent faisait tourner le jeu (ratios valables, valeurs absolues gonflées).
- Sous charge concurrente : B 52-67 → 6,5-8,7 ms, A 5,0-8,8 → 2,0-3,3 ms.
- Contrôle négatif : nouveaux shaders avec l'ancien dispatch → empreinte différente.
- Non couvert hors-jeu : sorties point et ligne, GS instancié, draws GS indirects, chemin d3d11 réel (pipeline, cache). À voir en jeu.

## 9. Reste à faire

- [ ] Session jeu ≤ 10 min : A/B base contre `perf-gs`, Élevé (chargement direct) puis Moyen (bascule en jeu), captures.
- [ ] Si le canal Vertex reste haut : Metal System Trace sur le build optimisé pour voir ce qui reste.

## 10. Banc

- Build : `scratchpad/mkpayload.sh <dest>` (compile + install build64/build32, assemble `wine/…`). Payload de base : build de `eb1a631f`.
- Session jeu : `game.sh lock gs` → install → `load_save.sh` → `bench.sh` → `restore` + `stop` + Option.ini validé + `METAL_HUD=0` → `unlock gs`. 10 min max.
- Le menu pause s'ouvre bien avec Échap depuis l'écran de déploiement (le briefing reste affiché derrière).
- Hors-jeu : `tools/gs-bench/README.md`.
