# Build local DXMT (fork Brietat71/dxmt, branche `warno-fixes`)

Build "builtin" Wine (x86_64 + i386) sur MacBook Pro M5 Max, macOS 27.2, Xcode 27.2.
Recette = jobs CI `build-clang-release-x86_64-windows-cross` + `build-clang-release-x86-windows-cross`
(`.github/workflows/ci.yml`), parce que le payload en service (Sikarugir WARNO) est un build llvm-mingw (libc++) avec `d3d12`.

## Base

- Le payload en service annonce `v0.80-250-g427228c` (fichier `version` : `v0.80-244-g7c8dee1`).
  `427228c` est introuvable : ni upstream (branches, tags, `refs/pull/*/head|merge`), ni API GitHub, ni CI runs.
  Le `winemetal.so` de référence a `sdk 27.0` alors que la CI tourne sur Xcode 16.1 : build local du mainteneur Sikarugir.
- Base retenue : upstream `main` = `0df86108348bab2aaa5c4610a5eb63a9de9047ac` (`v0.80-408-g0df86108`, 2026-11-10).
- Ancêtre public le plus proche du payload : `7c8dee1c2d73415301ceb7d1fa810861cef4cd67` (`v0.80-244`).
  Pour coller au payload : `git rebase --onto 7c8dee1 upstream/main warno-fixes`.

## Toolchain (`~/code/dxmt-toolchain`, hors repo)

| Outil | Version | Source |
|---|---|---|
| LLVM (airconv) | 15.0.7, x86_64, static | build source, flags CI |
| llvm-mingw | 20260908 ucrt macos-universal (clang 23.1.1) | release mstorsjo |
| Wine (headers/winebuild) | v8.16-3shain (winebuild 8.17) | release 3Shain/wine |
| meson | 1.10.0 (venv, comme CI) | pip |
| ninja / cmake | 1.13.2 / 4.4.4 | Homebrew (déjà présents) |
| Apple clang / metal | 21.0.0 / 32023.922 | Xcode 27.2 |

Rosetta requis (LLVM x86_64 : llvm-tblgen tourne sous Rosetta pendant le build).

```sh
TC=~/code/dxmt-toolchain
mkdir -p $TC && cd $TC
git clone --depth 1 --branch llvmorg-15.0.7 https://github.com/llvm/llvm-project.git llvm-project
curl -fsSL https://github.com/3Shain/wine/releases/download/v8.16-3shain/wine.tar.gz -o wine.tar.gz
mkdir -p wine && tar -zxf wine.tar.gz -C wine
curl -fsSL https://github.com/mstorsjo/llvm-mingw/releases/download/20260908/llvm-mingw-20260908-ucrt-macos-universal.tar.xz -o llvm-mingw.tar.xz
tar -xf llvm-mingw.tar.xz
python3 -m venv venv && venv/bin/pip install meson==1.10.0

cmake -B ./llvm-darwin-build -S ./llvm-project/llvm \
  -DCMAKE_INSTALL_PREFIX="$(pwd)/llvm-darwin" \
  -DCMAKE_OSX_ARCHITECTURES=x86_64 \
  -DLLVM_HOST_TRIPLE=x86_64-apple-darwin \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DLLVM_ENABLE_ZSTD=Off \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-D_LIBCPP_KEEP_TRANSITIVE_INCLUDES_LLVM23" \
  -DLLVM_TARGETS_TO_BUILD="" \
  -DLLVM_BUILD_TOOLS=Off \
  -DLLVM_INCLUDE_BENCHMARKS=Off \
  -DBUG_REPORT_URL="https://github.com/3Shain/dxmt" \
  -DPACKAGE_VENDOR="DXMT" \
  -DLLVM_VERSION_PRINTER_SHOW_HOST_TARGET_INFO=Off \
  -G Ninja
cmake --build ./llvm-darwin-build
cmake --install ./llvm-darwin-build
```

LLVM 15.0.7 passe tel quel avec Apple clang 21 (warnings seulement, 0 patch).

## Build DXMT

```sh
cd ~/code/dxmt
git submodule update --init --recursive
TC=~/code/dxmt-toolchain
export PATH="$TC/venv/bin:$PATH:$TC/llvm-mingw-20260908-ucrt-macos-universal/bin"

# 64-bit : PE32+ + winemetal.so (Mach-O x86_64)
meson setup --cross-file build-win64.txt -Denable_nvapi=true -Denable_nvngx=true -Denable_d3d12=true \
  -Dnative_llvm_path=$TC/llvm-darwin -Dwine_install_path=$TC/wine \
  build64 --buildtype release --prefix $(pwd)/install64 --strip
meson compile -C build64 && meson install -C build64

# 32-bit (WoW64) : PE32
meson setup --cross-file build-win32.txt -Dwine_install_path=$TC/wine \
  build32 --buildtype release --prefix $(pwd)/install32 --strip
meson compile -C build32 && meson install -C build32

# Payload
rm -rf out && mkdir -p out/wine
cp -R install64/x86_64-windows install64/x86_64-unix install32/i386-windows out/wine/
find out -name '*.a' -delete
mv out/wine/x86_64-windows/d3d12.dll out/wine/x86_64-windows/_d3d12.dll   # comme le payload Sikarugir (d3d12 inerte)
git describe --always > out/version
```

Rebuild incrémental après modif : `meson compile -C build64 && meson compile -C build32` puis refaire le bloc Payload (ou `meson install`).

`build32/ build64/ install32/ install64/ out/` sont dans `.git/info/exclude` (pas dans `.gitignore`, pour ne pas toucher l'arbre upstream).

## Durées (M5 Max, 18 cœurs)

| Étape | Durée |
|---|---|
| Téléchargements + clone LLVM | ~1-2 min (non chronométré) |
| LLVM configure / build / install | 17 s / 143 s / 2 s |
| DXMT 32-bit (setup+compile+install) | 36 s |
| DXMT 64-bit (setup+compile+install) | 17 s |
| Total, de zéro (mur, inclut les pauses) | 385 s ≈ 6,5 min |

## Vérification

```
out/wine/i386-windows/{d3d10core,d3d11,dxgi,winemetal}.dll        PE32 Intel 80386
out/wine/x86_64-unix/winemetal.so                                 Mach-O 64-bit dylib x86_64
out/wine/x86_64-windows/{_d3d12,d3d10core,d3d11,dxgi,nvapi64,nvngx,winemetal}.dll  PE32+ x86-64
```

Jeu de fichiers identique à `WARNO.app/Contents/SharedSupport/renderer/dxmt/wine/` (tailles à ±1 %).
Écarts connus vs référence :

- Subsystem PE : `Windows GUI` ici, `Windows CUI` en référence (aucun flag dans le repo ; défaut lld). Sans effet pour une DLL.
- `winemetal.so` : `minos 27.2` ici (défaut toolchain), `14.0` en référence.
  Si besoin d'un payload portable : `export MACOSX_DEPLOYMENT_TARGET=14.0` puis `meson compile -C build64 --clean && meson compile -C build64`.
