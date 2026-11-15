# Banc hors-jeu de l'émulation GS

Deux bancs Metal en ligne de commande, sans Wine ni jeu. Ils servent à mesurer et à vérifier un changement
du mapping GS → mesh shader avant de prendre le verrou du jeu.

## `real.swift` : les vrais shaders, au bit près

Exécute les shaders object (VS) et mesh (GS) convertis par `airconv`, avec les conventions de liaison de DXMT
(tables de cbuffers et d'arguments, table des vertex buffers, arguments de draw), dans une passe profondeur à
4 couches. Sort le temps GPU médian et une empreinte du tampon de profondeur.

Le bytecode vient des shaders dumpés de WARNO (`~/code/dxmt-gs-data/gsprobe/*.cso`, hors repo : contenu du jeu).

```sh
# metallibs avec l'ancien et le nouveau convertisseur (airconv natif : build64/src/airconv/darwin/airconv)
./mklibs.sh <ancien>/airconv libs/base
./mklibs.sh <nouveau>/airconv libs/new
swiftc -O real.swift -o real
for s in A B C S; do ./real libs/base $s legacy; ./real libs/new $s new; done
```

L'empreinte `profondeur` doit être identique entre `legacy` et `new` pour chaque scénario.

| Scénario | Contenu |
|---|---|
| A | 1 450 draws × 498 triangles indexés (VS `75b4cf05` + GS `49993012`) |
| B | 203 draws × 5 759 instances d'un quad, les billboards d'arbres (VS `ee4035d1` + GS `e9bce06f`) |
| C | shaders de B, nombres d'indices (1 à 60) et d'instances (1 à 300) variés : paquets d'instances, warps partiels |
| S | shaders de A en triangle strip, avec index de coupure : 25 threads par mesh threadgroup, 2 threadgroups par warp |

`legacy` = dispatch d'origine (payload 16 256 o, 1 thread par mesh threadgroup, un warp par instance).
`new` = dispatch de `perf-gs`. `new1` = dispatch du commit `58b45832` (warps de 30), pour comparer les deux étapes.
`NORASTER=1` coupe la rastérisation : coût de la géométrie seule. Le banc recopie ce que font `GeometryDraw*` et `d3d11_pipeline_gs.cpp` : à tenir à jour.

Contrôle négatif : `./real libs/new B legacy` donne une autre empreinte.
Données adverses : `ADV=casc0|cascmax|instnan|inst0|insthuge` (cascades nulles ou à INT_MAX, instances NaN, nulles, quads géants).

## `bench.swift` + `gs.metal` : modèle synthétique en MSL

Même structure que l'émulation (object = VS, mesh = GS « cascades d'ombres »), écrite à la main, pour comparer
des variantes de mapping sans toucher au convertisseur.

```sh
swiftc -O bench.swift -o bench
./bench <A|B|AB> <mode 0..3> <mots de payload, 0 = au plus juste> <pack 0|1> [frames] [sommets par warp]
```

Modes : 0 = d'origine, 1 = grille mesh = primitives valides, 2 = un threadgroup + atomiques, 3 = un threadgroup + préfixe en 2 passes,
4 = routage par cascade en pipeline vertex (draw instancié × 4 cascades, sans troncature `maxvertexcount`, inexact),
5 = idem exact (cascades 0 à 2 instanciées, cascade 3 non indexée avec la troncature).
`RT=64` réduit la cible : remplissage négligeable, reste la géométrie.
