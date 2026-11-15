#!/bin/zsh
# mklibs.sh <airconv> <outdir> : metallibs object (VS) + mesh (GS) des deux scénarios, avec le layout IA du banc
A=$1; O=$2; D=~/code/dxmt-gs-data/gsprobe; mkdir -p $O
VA=75b4cf05cb4c8de00f2fe73bdb82271058b28eaa; GA=499930125e9b4e17d3b82c45df694a7bf061cb80
VB=ee4035d16fea99b95798f6108b6fd974d1feda7c; GB=e9bce06f16b570dcfceb3b0c3c7173b05feb9732
$A -A --geometry-after-vertex=$D/$GA.cso --ia-layout=0:0:0:30 --ia-index-format=1 $D/$VA.cso -o $O/A_obj.metallib
$A -A --vertex-before-geometry=$D/$VA.cso $D/$GA.cso -o $O/A_mesh.metallib
$A -A --geometry-after-vertex=$D/$GB.cso --ia-layout=0:0:0:29,1:1:0:31:1,2:1:16:31:1 --ia-index-format=1 $D/$VB.cso -o $O/B_obj.metallib
$A -A --vertex-before-geometry=$D/$VB.cso $D/$GB.cso -o $O/B_mesh.metallib
$A -S --geometry-after-vertex=$D/$GB.cso --ia-layout=0:0:0:29,1:1:0:31:1,2:1:16:31:1 --ia-index-format=1 $D/$VB.cso -o $O/B_obj.ll
$A -S --geometry-after-vertex=$D/$GA.cso --ia-layout=0:0:0:30 --ia-index-format=1 $D/$VA.cso -o $O/A_obj.ll
$A -A --gs-strip --geometry-after-vertex=$D/$GA.cso --ia-layout=0:0:0:30 --ia-index-format=1 $D/$VA.cso -o $O/S_obj.metallib
$A -A --gs-strip --vertex-before-geometry=$D/$VA.cso $D/$GA.cso -o $O/S_mesh.metallib
ls -la $O | awk '{print $5, $NF}'
