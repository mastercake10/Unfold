#!/bin/bash
# usage: batch_real.sh <samples dir with N_x.jpg + scanN.jpg> <out root> <tag> <views e.g. "l r u d"> [docscan3d args...]
dir=$1; root=$2; tag=$3; views=$4; shift 4
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$root"
for i in 1 2 3 4 5; do
  dpi=300; [ $i = 4 ] && dpi=308
  o=$root/${i}_$tag
  imgs=""; for v in $views; do imgs="$imgs $dir/${i}_$v.jpg"; done
  if python3 "$here/../docscan3d.py" $imgs -o "$o" "$@" > "$o.log" 2>&1; then
    r=$(python3 "$here/eval_real.py" "$o/flat.png" "$dir/scan$i.jpg" --scan-dpi $dpi --json)
    f=$(python3 -c "import json;r=json.load(open('$o/report.json'));print(f\"{r['focal_px']:.0f} {r['reproj_rms_px']:.3f} {r['outline_fit_rms_mm']:.2f} {r['outline_points']}\")")
    echo "$i $tag $f $r"
  else
    echo "$i $tag FAIL $(tail -1 $o.log)"
  fi
done
