# Reproduce the Aukerman demo

Requirements: COLMAP, Nerfstudio (splatfacto) with CUDA, and `pip install -e ".[colmap]"`. It took about 40 minutes on an RTX 4080 Super.

```bash
git clone --depth 1 https://github.com/OpenDroneMap/odm_data_aukerman.git
cd odm_data_aukerman && mkdir -p colmap/sparse

colmap feature_extractor --database_path colmap/db.db --image_path images \
  --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV --SiftExtraction.max_image_size 3200
colmap exhaustive_matcher --database_path colmap/db.db
colmap mapper --database_path colmap/db.db --image_path images --output_path colmap/sparse

georef-splat align --model colmap/sparse/0 --images images --output colmap/sparse_enu/0 \
  --manifest georeference_manifest.json

# half-resolution copies in images_2/, then:
ns-train splatfacto --data . --max-num-iterations 30000 colmap \
  --colmap-path colmap/sparse_enu/0 --downscale-factor 2 \
  --orientation-method none --center-method none --auto-scale-poses False \
  --assume-colmap-world-coordinate-convention False --load-3D-points True
ns-export gaussian-splat --load-config outputs/*/splatfacto/*/config.yml --output-dir export

georef-splat tiles --ply export/splat.ply --manifest georeference_manifest.json --out tiles \
  --min-opacity 0.08 --max-scale 3 --crop-to-cameras 25 --clip-above-ground 32 --sh-degree 1
```

Dense stereo (the measurement surface), then the audit and the defensible-space screen:

```bash
colmap image_undistorter --image_path images --input_path colmap/sparse_enu/0 --output_path dense --max_image_size 2000
colmap patch_match_stereo --workspace_path dense --PatchMatchStereo.geom_consistency true
colmap stereo_fusion --workspace_path dense --input_type geometric --output_path dense/fused.ply

georef-splat audit --ply export/splat.ply --points dense/fused.ply --manifest georeference_manifest.json \
  --crop-to-cameras 25 --clip-above-ground 32 --ground 3dep --ground-offset -1.27 --out audit
georef-splat zones --points dense/fused.ply --manifest georeference_manifest.json \
  --crop-to-cameras 25 --ground 3dep --ground-offset -1.27 --out zones
python evidence_chips.py colmap/sparse_enu/0 images zones/zones_summary.json zones/zones.geojson zones/chips
```

`--ground-offset` is the vertical bias measured by `vertical_check.py` (reconstruction minus 3DEP).

Independent vertical check (queries USGS EPQS and NOAA NGS):

```bash
python vertical_check.py check_candidates.json georeference_manifest.json vertical_check.json
python build_demo_data.py georeference_manifest.json vertical_check.json ../site/splat-to-earth/data
```

`check_candidates.json` holds ground picks from the sparse cloud: the lowest 10% of points in each 30 m cell, keeping only cells where those points agree within 1.5 m.
