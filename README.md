# georef-splat

[![tests](https://github.com/Emerlads/georef-splat-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/Emerlads/georef-splat-tools/actions/workflows/ci.yml)

Put a drone Gaussian splat at its true latitude, longitude and height, then **measure** how right that is.

**Live demo: [emerlads.github.io/splat-to-earth](https://emerlads.github.io/splat-to-earth/).** It uses public CC0 drone data, a CesiumJS viewer, and a vertical-datum toggle.

```
GPS-tagged photos
  → COLMAP SfM
  → georef-splat align      RANSAC Sim3 onto EXIF GPS, local East-North-Up metres
  → Nerfstudio splatfacto   train in the aligned frame
  → georef-splat tiles      ENU → glTF → SPZ v2 → 3D Tiles 1.1 (KHR_gaussian_splatting)
  → CesiumJS                root transform = ENU→ECEF at the origin
```

## Results on public data

[OpenDroneMap `odm_data_aukerman`](https://github.com/OpenDroneMap/odm_data_aukerman) is 77 photos from a 2016 Sony DSC-WX220, flown about 100 m above ground in Ohio.

| Measure | Value | What it means |
|---|---|---|
| Cameras registered | 76 / 77 | COLMAP, 1.24 px mean reprojection error |
| Camera fit residual | 0.93 m H, 0.84 m V | Solved cameras vs the GPS they were fitted to. A QA number, **not** accuracy |
| Vertical vs USGS 3DEP lidar | −1.27 m bias, 1.46 m RMSE (44 pts) | Independent: SfM ground vs bare-earth lidar + NOAA GEOID18 |
| Vertical, 3DEP-anchored | 0.74 m RMSE | Bias fitted on half the checkpoints, scored on the other half |
| Same model, geoid misapplied | −35.2 m | What happens if ellipsoidal GPS height is treated as sea level |

Horizontal accuracy against independent control is not claimed. It needs surveyed GCPs.

### Photoreal splats from commercial trainers (LA County Arboretum)

Most production splats come from Postshot, RealityScan and similar tools. They export in their own local frame, often with swapped axes. `examples/botanical/georef_trainer_splat.py` georeferences such a splat using only what those tools export:
1. Recover the splat-to-grid frame by testing all 24 axis rotations against the trainer's dense cloud.
2. Refine with trimmed ICP.
3. Chain a RANSAC Sim3 from the grid-frame cameras to GPS in ENU.

On a 2,046-photo DJI Mavic 3M RTK survey (2.99M-splat UHD model, used with permission):

| Measure | Value |
|---|---|
| Splat → grid frame | axes (x, z, −y), scale 0.998 |
| Camera fit vs raw EXIF GPS | 0.09 m H, 0.50 m V (2,046 cameras) |
| Vertical vs USGS 3DEP lidar | −0.42 m bias, 0.72 m RMSE (59 checkpoints) |
| 3DEP-anchored, held out | 0.59 m RMSE |
| Altitude datum | ellipsoidal (treating it as MSL: −34.25 m) |

## Splat geometry audit: can you measure off a splat?

Gaussian splats are optimised for appearance. On low-texture surfaces they often get the colour right at the wrong depth. `georef-splat audit` renders the splat straight down by alpha compositing (as a viewer would) and compares that surface, cell by cell, with a multi-view-stereo cloud from the same photos. Where available, it also compares against lidar.

On Aukerman, with both surfaces checked against 3DEP lidar at the same independent ground checkpoints:

| Surface | vs 3DEP lidar | Notes |
|---|---|---|
| COLMAP multi-view stereo | +0.48 m median, 0.72 m RMSE (n=36) | |
| Gaussian splat, standard sparse-point init | +8.5 m median, 19.1 m RMSE (n=26) | 70% of its surface floats > 2.5 m above the stereo surface |
| Gaussian splat, initialised from stereo | +3.2 m median, 9.8 m RMSE (n=22) | 54% floating, from the same photos and settings |

The splat looks like a clean orthophoto from above. Its geometry is not fit for measurement. Seeding it from the stereo cloud (`examples/aukerman/mvs_init_points.py`) halves the error, but that's still metres. So measure from stereo and show the splat.

```bash
georef-splat audit --ply splat.ply --points dense/fused.ply --manifest georeference_manifest.json \
    --crop-to-cameras 25 --clip-above-ground 32 --ground 3dep --ground-offset -1.27 --out audit/
```

## Defensible-space screening

`georef-splat zones` grids a georeferenced surface (stereo points or a splat), takes bare earth from USGS 3DEP if asked (with a smooth local offset correction), and classifies each cell as ground, low vegetation, shrub / tall grass, tree canopy, structure or raised object. It finds structures with no footprint data and measures fuel in California's Zone 0 / 1 / 2 rings (5 / 30 / 100 ft). Outputs are GeoJSON for ArcGIS or QGIS, a classification PNG and a summary. `examples/aukerman/evidence_chips.py` adds the source photo that best shows each structure, with the footprint projected onto it.

Structures must stand above their *own surroundings*, not just above the lidar ground. That rule removed a raised footpath that otherwise passed every other test. Small "woody" patches with weak average greenness become objects (on Aukerman: picnic tables), which in Zone 0 is its own flag.

## The vertical datum trap

GPS altitude can be height above the WGS84 **ellipsoid** or above **mean sea level** (the geoid). EXIF doesn't say which, and vendors differ. At Aukerman the two are 33.9 m apart. Getting it wrong moves the model by the full geoid height, while the horizontal fit still looks perfect.

`examples/aukerman/vertical_check.py` settles it empirically. It takes SfM ground points, converts them to geodetic coordinates, and compares them with USGS 3DEP bare-earth elevation plus the GEOID18 undulation. For this camera, the heights are ellipsoidal.

## Install

```bash
pip install -e ".[colmap,test]"
pytest -q
```

## Use

```bash
georef-splat align --model colmap/sparse/0 --images images --output colmap/sparse_enu/0 \
    --manifest georeference_manifest.json
# if your drone writes MSL altitude: --altitude-datum orthometric --geoid-undulation <N>

ns-train splatfacto colmap --data . --colmap-path colmap/sparse_enu/0 \
    --orientation-method none --center-method none --auto-scale-poses False \
    --assume-colmap-world-coordinate-convention False
ns-export gaussian-splat --load-config <run>/config.yml --output-dir export

georef-splat tiles --ply export/splat.ply --manifest georeference_manifest.json \
    --out tiles --max-splats 1200000
```

Nerfstudio must not re-centre, re-orient or re-scale the poses. Otherwise the trained splat leaves the ENU frame.

The full Aukerman reproduction is in [`examples/aukerman/`](examples/aukerman/).

## What's inside

| Module | |
|---|---|
| `geodesy.py` | WGS84 geodetic ↔ ECEF ↔ ENU, ENU→ECEF 4×4 (same as Cesium `eastNorthUpToFixedFrame`) |
| `sim3.py` | Umeyama similarity + RANSAC, residual statistics |
| `exif.py` | EXIF GPS reader (altitude datum deliberately left unspecified) |
| `ply.py` | 3DGS PLY read/write |
| `tiles.py` | SPZ v2 encoder (with SH up to degree 3), glTF `KHR_gaussian_splatting` + `KHR_gaussian_splatting_compression_spz_2` writer, 3D Tiles 1.1 tileset |
| `audit.py` | Splat-vs-reference surface audit (alpha-composited nadir render vs MVS / lidar) |
| `zones.py` | Surface gridding, local lidar ground correction, classification, Zone 0/1/2 screening |
| `usgs.py` | USGS 3DEP bare-earth export and NOAA GEOID18 lookups |
| `tools/` | Earlier single-file scripts (image subset selection, decimation, axis-fix) kept for reference |

Tests cover:
- geodesy round-trips and axis directions
- Sim3 recovery under outliers
- SPZ round-trips, spherical harmonics included, through an independent decoder
- an end-to-end check that a tile's content lands at the right ECEF position
- audit detection of a floating layer
- zone screening on a synthetic house, tree and shrubs

Design note: data stays in local ENU, and the glTF node carries the ENU→glTF axis change. The renderer then rotates positions, orientations and spherical harmonics together, so no Wigner-D rotation is needed in the converter.

Lessons from Aukerman are built into the defaults and documented in the code:
- Down-sampling by splat size keeps blurry splats.
- Down-sampling by opacity deletes lawns.
- Opacity-weighted random sampling keeps coverage even.
- CesiumJS also needs `3DTILES_content_gltf` declared on the tileset, or it draws splats as points.

Limitations: a single tile with no LOD; zone thresholds are parameters, not standards.

---

**Roberto Bloom**, Geospatial Software Engineer and FAA Part 107 remote pilot · [emerlads.github.io](https://emerlads.github.io) · [LinkedIn](https://linkedin.com/in/roberto-bloom)
