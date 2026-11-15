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

Trois commits : `58b45832` (points 1 à 3), `4d0b7d2a` (point 4), `cc3d8a81` (suites de la relecture).

1. **Payload au plus juste** : `SM50GSPayloadSize()` = en-tête + sommets du warp × (mot de validité + 16 o × registres de sortie du VS),
   soit 1 296 o ou 2 320 o pour WARNO au lieu de 16 256. Convertisseur (object et mesh) et `payload_memory_length` du pipeline.
2. **GS regroupés par mesh threadgroup** : min(primitives du warp, `SM50GSMeshThreadsLimit()` = 256 / (maxvertexcount + 1)) threads,
   un thread par invocation du GS (20 pour WARNO au lieu de 1).
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
4. **Warps de 64 sommets** pour les topologies en liste (`SM50GSWarpScale()`), quand le payload a la place (jusqu'à 15 registres de sortie du VS).
   Liste de triangles : 60 sommets, 20 invocations de GS par mesh threadgroup, 10 quads par paquet. Deux fois moins de threadgroups.
   - Rendu possible par les **mots de validité** : chaque thread object écrit dans le payload si son sommet est valide
     (dans le draw, pas un index de coupure). Le mesh shader en déduit la validité de sa primitive.
   - L'object shader n'a plus ni masque atomique 32 bits, ni barrière, ni boucle de masques par primitive.
   - Un thread dont le GS n'émet rien saute la passe d'écriture.
5. `AIRCONV_VERSION` 26 → 27 : invalide le cache de shaders (l'ancien AIR est incompatible avec le nouveau dispatch).
6. `airconv` (CLI de dev) : `--ia-layout`, `--ia-index-format`, `--gs-strip`, pour produire hors-jeu les shaders tels que le runtime les compile.

Rien de spécifique à une architecture : tout est dans le convertisseur, l'encodeur de draws et le shader de marshal.

## 8. Vérification hors-jeu (`tools/gs-bench/real.swift`, vrais shaders de WARNO)

Tampon de profondeur 4 couches comparé **au bit près** entre l'ancien convertisseur + ancien dispatch et le nouveau.
GPU au repos, médiane de 6 à 10 frames, `4d0b7d2a` :

| Scénario | Ancien (ms GPU) | Nouveau (ms GPU) | Profondeur |
|---|---|---|---|
| A : 1 450 draws × 498 triangles | 3,2 | 1,1 | identique |
| B : 203 draws × 5 759 billboards | 38,4 | 5,6 | identique |
| C : indices 1 à 60, instances 1 à 300 (paquets, warps partiels) | 4,2 | 1,3 | identique |
| S : triangle strip avec index de coupure (25 threads, 2 mesh TG par warp) | 3,3 | 1,3 | identique |

Géométrie seule (rastérisation coupée, `NORASTER=1`) :

| Scénario | Ancien | `58b45832` | `4d0b7d2a` |
|---|---|---|---|
| B : billboards | 38,4 | 2,6 | **2,0** |
| A : triangles indexés | 3,3 | 1,2 | **0,9** |

- Dans B, la rastérisation des 1,5 M de petits triangles coûte 3,6 ms. Avant, elle était cachée derrière la géométrie
  (38,4 ms avec ou sans). **Le coût de l'émulation GS passe de ~42 ms à ~3 ms ; ce qui reste dans les passes d'ombre est le remplissage.**
- Contrôle négatif : nouveaux shaders avec l'ancien dispatch → empreinte différente.
- Sous charge (un autre agent fait tourner le jeu), les valeurs absolues montent de 30 à 70 % ; les ratios tiennent.
- Non couvert hors-jeu : sorties point et ligne, GS instancié, draws GS indirects, chemin d3d11 réel (pipeline, cache). À voir en jeu.
- Piste écartée après mesure : warps de 84 sommets, aucun gain de plus que 60 (banc synthétique).

### Relecture indépendante du diff (agent de revue, chemins non exercés par le banc)

- Rien trouvé sur : sorties ligne et point en mode regroupé, équivalence des mots de validité avec les anciens masques
  (listes, strips, adjacences), chemin non regroupé, accord convertisseur / encodeur / marshal, disposition des structs, dominance SSA.
- Corrigé dans `cc3d8a81` : un GS avec UAV reste à 1 invocation par mesh threadgroup (le corps tourne deux fois en mode regroupé) ;
  arrondi du marshal indirect aligné sur l'encodeur.
- Signalé, antérieur à ce travail, laissé tel quel : en triangle strip avec adjacence, le sommet adjacent de la primitive 0
  est lu à l'index 2 au lieu de 31 (`URem` sur une valeur non signée qui a bouclé, `dxbc_converter_gs.cpp`, cas `TRIANGLE_ADJ`).
  Le corriger changerait le rendu par rapport à la base : hors périmètre.
- Les shaders de WARNO convertis par `cc3d8a81` sont identiques à l'octet près à ceux de `4d0b7d2a`.

## 9. Mesures en jeu (session `s1`, 2026-11-15 20:02-20:08, verrou tenu 6 min 05 s)

Même session, dos à dos, scène Hesse au déploiement, 2560×1600, 3D 1920, tout le reste au maximum. HUD Metal, 2 lectures à 10 s d'écart.

| Shaders | Base `eb1a631f` | `perf-gs` `cc3d8a81` | Gain |
|---|---|---|---|
| **Élevé** (chargé directement, `ShaderQuality = 0`) | 21,24 fps, GPU 47,17 ms | **89,03 fps, GPU 11,32 ms** | × 4,19 |
| **Moyen** (basculé en jeu) | 21,56 fps, GPU 46,46 ms | **91,24 fps, GPU 11,05 ms** | × 4,23 |

- Lectures brutes : optimisé Élevé 88,72 / 89,33 ; Moyen 91,21 / 91,27. Base Élevé 21,21 / 21,27 ; Moyen 21,59 / 21,54.
- ioreg : Device 99 % dans les quatre cas. Le GPU reste saturé, à 89 fps au lieu de 21.
- Rappel : Bas = 134-150 fps, GPU ~6 ms. Élevé coûte maintenant ~5 ms de plus que Bas, contre ~41 ms avant.

Rendu, captures plein écran `~/code/dxmt-gs-data/captures/` (`s1-A-*` = optimisé, `s1-B-*` = base), comparées par `imgdiff.py`
(part des pixels dont un canal diffère de plus de 8/255, de plus de 32/255) :

| Paire | > 8 | > 32 |
|---|---|---|
| Bruit : même build, 10 s d'écart (base ; optimisé) | 7,7 % ; 14,1 % | 0,17 % ; 0,28 % |
| Base contre optimisé, Élevé (t = 5 s ; t = 15 s) | 3,6 % ; 4,4 % | 0,11 % ; 0,18 % |
| Base contre optimisé, Moyen (t = 5 s ; t = 15 s) | 3,3 % ; 2,2 % | 0,14 % ; 0,16 % |

- L'écart base/optimisé est **sous le bruit d'animation** d'un même build. Le pavé le plus différent (`captures/eleve-base-vs-opt-worst.png`) :
  forêt et ruines, ombres des arbres et des bâtiments présentes des deux côtés, seule la fumée animée change.
- La preuve au bit près est celle du banc hors-jeu (section 8) ; en jeu, la scène est animée.

Non fait :
- **Trace Metal du build optimisé** : `xctrace` coupé à 40 s avant la fin de sa finalisation, trace inexploitable. Pas de ventilation Vertex/Fragment de ce qui reste.
  `session.sh` laisse maintenant 120 s.
- **Combat** (`run_combat.sh`) : non mesuré en Élevé. Référence en Bas : ~70 fps en 4v4.

## Mission 2 : minimum fps en vue dense, réglages au maximum (en cours)

Cible : minimum ≥ 60 fps en Élevé sur le parcours caméra (`campath.sh`), profil `set_video.sh max`
(3456×2234, Résolution 3D Off, Anisotropie 8X, Post-traitement Extrême). Rendu identique.
La bouteille Rosetta est bloquée par une boîte Steam « Conflit avec le cloud » (choix réservé à l'utilisateur) :
les mesures en jeu passent par la pile arm64.

### Demandes de mesure (agent arm64)

**Priorité 1 (aucune construction spéciale)** : `perf-gs` étiquette maintenant les encodeurs de rendu (commit `feat(dxmt): label render encoders`,
intégré automatiquement dans le runtime `gs`). Refaire la **trace Metal en vue dense fixe, profil max**, et donner la sortie complète de
`parse_gpu.py <export.xml> ""` (les 15 « top libellés » par canal), ou le chemin du `.xml`. Les libellés sont de la forme
`GS 0c1d 2048x2048x4 c/ dC/S` (passes d'ombre par GS), `R 4c1d 3456x2234x1 cLLLL/SSSS dL/S` (passe principale ; `L` = attachements rechargés).
C'est ce qui dira si les 8 ms de Vertex et les 16 ms de Fragment viennent des ombres, de la passe principale ou du post-traitement.

**Priorité 2 (GPU hang, aucune construction spéciale non plus)** : lancer le runtime `gs` avec **`DXMT_DEBUG_ENCODER_STATUS=1`** dans l'environnement.
Les deux `Caused GPU Hang Error` reviennent à chaque lancement (première frame du menu, frames 360-380 ; première frame de la carte, frames 1040-1180).
Avec la variable, le journal reçoit sous l'erreur l'état de chaque encodeur du tampon (`dxmt:   FAULTED   GS 0c1d 2048x2048x4 …`) :
il suffit de recopier ces lignes. Contre-épreuve utile : le même chargement en Élevé avec le runtime `fork` (ancien chemin GS) donne-t-il aussi les deux erreurs ?

**Priorité 3 (charge GS en vue dense)** : ajouter **`DXMT_DEBUG_GS_STATS=1`** à l'environnement. Toutes les 10 s, le journal reçoit par paire VS+GS
`gs stats: vs … gs … draws/s … vertices/s … instances/draw … object_tg/s …`. Recopier les lignes d'une fenêtre entièrement prise en vue dense fixe
(et le fps de cette fenêtre) : c'est ce qui dit combien de triangles passent par les GS de près (vrais modèles d'arbres ou billboards).
Les trois variables peuvent être posées ensemble pour un seul lancement.

Le reste (sonde complète sur `gs-probe2`, seulement si les variables ne suffisent pas) :

Runtime à construire depuis **`origin/gs-probe2`** (= `perf-gs` + 2 commits d'instrumentation, à ne pas fusionner).
Il écrit dans `~/code/dxmt-gs-data/probe2/` (chemins codés en dur, rien à configurer) :
`stats.txt` (toutes les 10 s : `T=<epoch>`, fps, draws/frame, et par paire VS+GS draws, indices, instances, threadgroups object),
`pairs.txt` + `<sha1>.cso` (pipelines GS créés), `errors.txt` (erreurs de command buffer et numéro de frame),
`errors-encoders.txt` (encodeurs du tampon fautif : `etat=4` = fautif, avec son étiquette).
Les encodeurs de rendu sont étiquetés pour la trace Metal : `GS`/`TS`/`R`, nombre de couleurs, profondeur, taille, actions de chargement/stockage
(ex. `GS 0c1d 2048x2048x4 c/ dC/S` ; `L` = attachement rechargé, donc passe coupée). Branche à `54fd47a3`.

1. **Vue dense fixe, profil max, Shaders Élevé.** Hesse → Lancer partie → x4 60 s → x1 → début du parcours
   (zoom avant 6 crans, défilement droite 7 s, haut 7 s), puis **caméra arrêtée ≥ 40 s**. Dans cette vue :
   fps, GPU ms, état GPU, CPU du thread principal ; **Metal System Trace 3 s** et l'export de la table
   `metal-gpu-intervals` (chemin du `.xml`) ; noter l'heure (`date +%s`) d'entrée et de sortie de la vue.
2. **Même vue en Shaders Bas** (basculé en jeu, sans bouger la caméra) : fps, GPU ms, CPU du thread principal, trace.
3. **Parcours caméra** (2 boucles) dans le profil max : minimum, p5, médiane, moyenne.
4. **GPU hang** : après un chargement en Élevé, le contenu de `errors.txt` et `errors-encoders.txt`.
   Contre-épreuve : le même chargement en Élevé avec le runtime `fork` (ancien chemin GS) produit-il aussi
   `Device error ... Caused GPU Hang Error` ? (attribution : mon chemin regroupé, ou les passes GS/TS en général).

### Relevé arm64 en profil max (runtime `gs` = `cc3d8a81`, PROGRESS.md 21:36)

| Scène | FPS | GPU ms |
|---|---|---|
| Déploiement | 63,5 (min 60) | 15,9, limité GPU |
| Parcours caméra, 62 s | moy 62, **min 38, p5 41**, médiane 57 | 22-27 dans les vues denses |
| Vue dense fixe | 51-52 | 17,5 pour un intervalle de 19 ; thread principal 98 % |

Trace Metal en vue dense : **Fragment 15,8 ms + Vertex 8,2 ms + Compute 0,6 ms** de travail GPU par image.
Cible : intervalle ≤ 16,7 ms. Il manque ~35 % de temps GPU dans les vues denses, et ~20 % sur le thread principal (processeur émulé).

### Lecture de la trace étiquetée en vue dense (`~/code/arm64-wine/traces/max-dense-gs-labeled.xml`, 149 images, 22,2 ms d'intervalle)

Analyse hors jeu, script jetable sur l'export (`start`, `duration`, `encoder-id`). Trois constats.

**1. Les passes 512×512 « à 2,2 ms » et 32×64 « à 0,7 ms » sont un artefact d'attribution, pas un coût.**
La durée d'un intervalle court de sa mise en file à sa fin. Le canal Fragment est une file : la passe 512×512 est soumise pendant le fragment
de la passe principale et se termine juste après lui (début 49,47 ms, fin 53,88 ms = fin de la passe principale). En temps **exclusif**
(fin moins la plus tardive de : son début, la fin de l'intervalle précédent du canal), par image :

| Passe | Vertex | Fragment |
|---|---|---|
| Principale `R 3c1d 3456x2234 cLLL/SSS dL/S` | 2,95 | 5,42 |
| **Ombres par GS** `GS 0c1d 2048x2048x4` | **3,80** | **2,55** |
| Post-traitement plein écran (10,8 passes `R 1c0d 3456x2234 cL/S`) | 0,12 | 1,77 |
| `R 1c0d 32x64` + `R 1c0d 1024x1024 cC/S` (soumises ensemble, 1,1 ms à deux) | | 1,33 |
| `R 1c1d 3456x2234 cL/S dL/S` (4,9 passes) | | 0,84 |
| Terrain tessellé `TS 3c1d 3456x2234 cCCC/SSS dC/S` | 0,42 | 0,63 |
| Atlas 4096×2048 (3 passes) | 0,58 | 0,64 |
| `R 1c0d 512x512` | | ~0 |
| Total des canaux | 8,61 | 15,2 (+ Compute 0,81) |

GPU occupé (union de tous les intervalles) : **21,9 ms sur 22,2 par image, 98 %**. Il n'y a pas de temps mort à récupérer entre encodeurs.

**2. L'image est une chaîne presque entièrement sérielle, imposée par le jeu.** Ordre observé (image 75, ms depuis le début de son travail GPU) :

| Début | Durée | Étape | Pendant ce temps |
|---|---|---|---|
| 0,0 | 3,0 | atlas, chaînes de mips, terrain tessellé | |
| 3,0 | 3,0 | **Vertex** passe principale | cœurs fragment presque inoccupés |
| 6,0 | 5,35 | **Fragment** passe principale | canal vertex inoccupé (0,3 ms de petites passes) |
| 11,3 | 0,4 | chaîne de réduction 1728×1117 → 1×1, puis `Compute Command 88` | |
| 11,7 | 4,0 | **Vertex** ombres par GS | **cœurs fragment inoccupés** |
| 15,7 | 2,6 | **Fragment** ombres par GS | canal vertex inoccupé |
| 18,3 | 4,0 | post-traitement (≈ 45 passes) | |

- Le vertex des ombres ne démarre qu'à la fin de `Compute Command 88`, qui suit la réduction de la profondeur de la passe principale :
  les cascades sont ajustées à la profondeur de l'image courante (le GS lit leurs bornes dans un buffer d'entiers). L'ordre
  « fragment principal → réduction → compute → ombres » vient du jeu ; DXMT ne peut pas faire chevaucher les ombres et la passe principale.
- **Pour l'anomalie « état Medium »** (agent arm64) : le GPU est « occupé » 98 % du temps, mais pendant 7 des 22 ms seule la géométrie travaille
  (vertex principal 3,0 + vertex des ombres 4,0), et le vertex des ombres est borné par un coût fixe par mesh threadgroup, pas par le calcul.
  Occupation réelle des cœurs faible sur un tiers de l'image : c'est cohérent avec un régulateur qui ne monte pas. Je n'ai rien vu dans DXMT qui
  attende entre deux encodeurs (146 encodeurs dans un seul command buffer par image, exécutés dos à dos).

**3. La coupure de la passe principale ne vient pas de la tessellation.** Ordre des encodeurs : `TS 3c1d` (terrain, n° 70), puis
**`R 1c0d 3456x2234 cL/S` (n° 71, 0,08 ms)**, puis `R 3c1d … cLLL/SSS dL/S` (n° 72, tout le reste). Le jeu change de cible entre le terrain et les objets
pour une petite passe plein écran sur une autre texture, puis revient au G-buffer : D3D11 impose de stocker puis de recharger.
DXMT ne peut fusionner 70 et 72 que si 71 peut passer avant ou après les deux, donc s'il ne lit pas le G-buffer et n'est pas lu par la suite.
À vérifier en jeu (quelles textures lit le n° 71) avant d'espérer quoi que ce soit ; le gain serait un stockage et un rechargement de 4 attachements.

### Où part la géométrie des GS maintenant (banc aux vrais shaders, expériences jetées, GPU au repos)

| Mesure (rastérisation coupée) | A : triangles indexés | B : billboards |
|---|---|---|
| Chemin complet | 0,70 ms | 1,88 ms |
| Étage object seul (aucun mesh threadgroup lancé) | 0,23 ms | 0,29 ms |
| Mesh threadgroups lancés mais corps du GS sauté | 0,68 ms | 1,87 ms |

- **Le coût restant est un coût fixe par mesh threadgroup** (~13 ns, soit ~0,65 ns par triangle à 20 invocations par threadgroup),
  pas le travail du GS : sauter tout le corps du GS ne change rien. L'étage object (les VS) pèse 15 à 30 %.
- Sans effet mesuré : les barrières (banc synthétique, avec ou sans), le nombre de primitives déclaré (160 → 80), un fragment shader vide ou absent.
- Le nombre d'invocations par mesh threadgroup est borné par la limite Metal de 256 sommets par mesh : 25 au plus pour `maxvertexcount = 9`, 20 aujourd'hui.
- Conclusion : la géométrie des GS est à ~20 % d'un pipeline vertex natif. Il n'y a plus de facteur 2 à y prendre.

### Piste « routage par cascade en pipeline vertex » : évaluée, écartée

Idée : le GS de WARNO ne fait que router le triangle vers la couche de chaque cascade touchée ; le rendre sans object ni mesh shader,
par un draw instancié × cascades dont l'étage vertex sort la couche (`render_target_array_index`).

- **Le test de boîte se remplace bien par le découpage** : un triangle dont la boîte ne recoupe pas la cascade est hors du volume, éliminé.
- **Pas la troncature `maxvertexcount`.** Le GS déclare 9 sommets pour 4 cascades : un triangle qui touche les cascades 0, 1 et 2
  n'est pas émis dans la 3. Reproduire cela exige la boîte du triangle, donc ses trois sommets, dans l'étage vertex :
  la cascade 3 doit passer en draw non indexé où chaque sommet recalcule les trois VS de son triangle.
- **Mesure** (`tools/gs-bench/bench.swift`, modes 4 et 5, cible 64×64 pour ne garder que la géométrie, minimum de 20 frames) :

| Charge | Mesh regroupé actuel (mode 3, warp 60) | Vertex, toutes cascades, inexact (mode 4) | Vertex, exact avec troncature (mode 5) |
|---|---|---|---|
| A : 1 450 draws × 498 triangles indexés | 0,70 ms | 0,55 ms | 0,57-0,59 ms |
| B : 203 draws × 5 759 billboards | 1,19 ms | 2,30-2,33 ms | 2,65 ms |

- Au mieux 20 % de mieux sur les triangles indexés, **deux fois pire sur les billboards** (chaque quad part dans les 4 cascades).
- Lecture : le chemin mesh regroupé est déjà au niveau d'un pipeline vertex natif. Ce qui reste dans les passes d'ombre est le travail
  lui-même (VS et remplissage), pas le surcoût d'émulation. **Je n'implémente pas cette piste** : gain faible et de signe variable,
  pour un second chemin de rendu à prouver et à maintenir.
- Correction d'une mesure antérieure : dans le banc synthétique, la charge A lançait 30 threads par object threadgroup quel que soit le warp.
  Le « × 2 à 60 sommets » annoncé pour A venait de là. Les gains des warps de 64 retenus sont ceux du banc aux vrais shaders (section 8), au bit près.

### GPU hang (2 erreurs sur arm64, Élevé, aux frames 375 et 1179) : pas reproduit hors jeu

- Les deux erreurs tombent à la **première frame d'une scène** (menu 3D, puis carte), d'après `~/Library/Logs/neutron/neutron-1611600.log`.
- Banc aux vrais shaders, données adverses (`ADV=`), ancien et nouveau chemin : cascades nulles, cascades à INT_MAX, instances NaN,
  instances nulles → aucune erreur GPU, sorties identiques (vides). Quads géants (`insthuge`) → **9,3 s de GPU par frame sans erreur**, sorties identiques.
- Charge géométrie extrême : 8 000 draws × 5 759 instances → 71 ms (nouveau), **1,34 s (ancien)**, sans erreur.
- Sommets NaN ou infinis épars (un seul sommet invalide par triangle, cas où les `min`/`max` « rapides » pourraient diverger
  entre la passe de comptage et la passe d'écriture) : sorties identiques, aucune erreur.
- Désaccord forcé entre les deux passes (expérience : les threads impairs comptent mais n'écrivent pas) : aucune erreur, et la sortie est
  celle d'un saut propre. Les primitives comptées mais jamais écrites ne rendent rien sur ce GPU. Pas de garde ajoutée.
- Nouveau scénario `N` du banc (draws non indexés, troisième paire de vrais shaders, 3 registres de sortie du VS) : identique au bit près.
- L'erreur est **reproductible à la première frame du menu** (frames 375 puis 380 sur deux lancements arm64 en Élevé).
- Donc ni la durée d'un tampon ni des données dégénérées ne suffisent à produire un hang ici. Rien dans le chemin regroupé n'écrit hors de ses bornes
  (relecture + sorties au bit près), sauf si le pipeline lié n'est pas celui du draw. Attribution impossible sans le jeu : voir la demande 4.

## 10. Reste à faire

- [ ] Mesurer Élevé en combat 4v4 (`run_combat.sh`), où le CPU compte davantage.
- [ ] Trace Metal du build optimisé pour ventiler les ~11 ms restants.
- [ ] Chemins relus mais jamais exécutés : sorties GS point et ligne, GS instancié, draws GS indirects, GS DXIL (laissés à 1 thread).
- [ ] Session jeu : `~/code/dxmt-gs-data/session.sh <tag> <payload A> <payload B>` (verrou, mesures, restauration, chien de garde 540 s).
      Payloads figés : `~/code/dxmt-gs-data/p-base` (`eb1a631f`), `p-opt1` (`58b45832`), `p-opt2` (`cc3d8a81`, version finale mesurée).

## 11. Banc

- Build : `scratchpad/mkpayload.sh <dest>` (compile + install build64/build32, assemble `wine/…`). Payload de base : build de `eb1a631f`.
- Session jeu : `game.sh lock gs` → install → `load_save.sh` → `bench.sh` → `restore` + `stop` + Option.ini validé + `METAL_HUD=0` → `unlock gs`. 10 min max.
- Le menu pause s'ouvre bien avec Échap depuis l'écran de déploiement (le briefing reste affiché derrière).
- Hors-jeu : `tools/gs-bench/README.md`.
