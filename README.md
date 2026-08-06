# georef-splat-tools

Command-line tools for taking GPS-tagged drone imagery through Structure-from-Motion into a **georeferenced** Gaussian splat that renders in the browser at true world coordinates.

Commercial photogrammetry suites do this behind a closed pipeline. These tools do it with open tooling, so every stage is inspectable and the resulting accuracy is *measured* rather than asserted.

---

## The actual problem

Structure-from-Motion gives you camera poses in an arbitrary local frame at arbitrary scale. A splat trained on that model is geometrically self-consistent and geospatially meaningless — it has no idea where on Earth it is or how large it is.

Bridging that gap is the interesting part, and it's where most of this code lives:

```
GPS-tagged imagery
   ↓  select_gps_image_subset.py          spatially diverse subset selection
   ↓  prepare_colmap_images.py            deterministic image set
   ↓  COLMAP / pycolmap                   solve camera poses (arbitrary frame)
   ↓  align_colmap_to_exif_gps_enu.py     ← Sim3 fit: poses → EXIF GPS, local ENU metres
   ↓  compute_georef_residuals.py         ← measure what that alignment actually achieved
   ↓  Nerfstudio splatfacto               train on the ENU-aligned model
   ↓  decimate_binary_ply.py              reduce for delivery
   ↓  create_axis_fixed_tileset.py        3D Tiles with corrected axis convention
   ↓  CesiumJS                            render at true lat/lon/elevation
```

Alignment happens **before** training, not after. Training on an already-metric model means the output is measurable by construction rather than being scaled into place afterward.

---

## Coordinate handling

`align_colmap_to_exif_gps_enu.py` does the geodesy explicitly rather than delegating it:

- EXIF GPS rationals → decimal degrees (`dms_to_degrees`)
- Geodetic → ECEF on the WGS84 ellipsoid (`geodetic_to_ecef`)
- ECEF → local ENU via the rotation matrix at the scene origin (`ecef_to_enu_matrix`)
- Similarity (Sim3) fit of solved camera centres onto those ENU positions, recovering scale, rotation, and translation

The output is a COLMAP model whose units are metres in a local ENU frame — directly consumable by Nerfstudio.

**A note on vertical datums.** GPS altitude is height above the WGS84 *ellipsoid*, not above mean sea level. Conflating the two puts models tens of metres off vertically, and it does so silently because the horizontal fit still looks perfect. Anything comparing against MSL-referenced terrain needs an explicit geoid correction.

---

## On accuracy claims

`compute_georef_residuals.py` emits horizontal and vertical residuals as RMSE, median, and p90, and stamps every output with:

> `"note": "Camera-prior fit residual, not independent surveyed GCP/checkpoint error."`

That distinction matters and is routinely blurred. A camera-prior fit residual measures how well the solved cameras agree with the GPS priors they were fitted to — it is not an independent accuracy assessment. Genuine accuracy validation requires surveyed checkpoints that were withheld from the fit.

Consumer GPS priors are also weak targets — metre-level at best. RTK positions or surveyed ground control should replace them when available, and the tooling is structured to accept that substitution.

Reported honestly, these numbers are useful for QA and regression tracking. Reported as "accuracy," they'd be overselling.

---

## Tools

| Tool | Purpose |
|---|---|
| `align_colmap_to_exif_gps_enu.py` | Sim3 alignment of a COLMAP model to EXIF GPS in local ENU metres |
| `compute_georef_residuals.py` | Horizontal/vertical residual statistics with an explicit methodology note |
| `select_gps_image_subset.py` | Select images by GPS proximity to a target area |
| `select_diverse_gps_image_subset.py` | Spatially diverse subset selection to avoid clustered viewpoints |
| `prepare_colmap_images.py` | Build a deterministic, reproducible COLMAP image set |
| `decimate_binary_ply.py` | Reduce Graphdeco-style binary PLY splats for delivery |
| `create_axis_fixed_tileset.py` | Emit 3D Tiles with corrected axis convention |
| `build_source_image_index.py` | Index source imagery for nearest-source-image lookup in the viewer |

Every tool is argparse-driven with no hardcoded paths.

---

## Requirements

```
numpy
pillow
pycolmap
```

Plus COLMAP and Nerfstudio for the reconstruction and training stages.

---

## Results from production use

Applied to real drone captures, this pipeline produced:

- **0.107 m horizontal GPS residual** across 119 of 120 solved cameras on a target subset
- **0.24 m 3D georeferencing RMSE**, surfaced in-app as a per-reconstruction QA readout
- A 109k-Gaussian scene compressed to roughly 7.4 MB for practical browser delivery

Both figures are camera-prior fit residuals under the caveat above.

---

## Scope

These are the georeferencing and preparation tools only. The viewer application, classification system, and client data are not part of this repository.

Published with permission. No client data, site identifiers, or proprietary application code is included.

---

**Roberto Bloom** — Geospatial Software Engineer, FAA Part 107
[LinkedIn](https://linkedin.com/in/roberto-bloom)
