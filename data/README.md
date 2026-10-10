# User-provided data

Raw observations and external plate models are not distributed with NORN. See [docs/data.md](../docs/data.md) for schemas and provenance requirements.

The generic `configs/reconstruction.json` expects the following user-supplied files:

```text
data/observations.csv
data/modern.dat
data/plate_gmt/topology_0.00Ma.gmt ... topology_60.00Ma.gmt
data/deformation_gmt/topology_0.00Ma.gmt ... topology_60.00Ma.gmt
data/external/zahirovic2022/Rotations/Zahirovic_etal_2022_OptimisedMantleRef_and_NNRMantleRef.rot
data/external/zahirovic2022/StaticPolygons/Global_EarthByte_GPlates_PresentDay_StaticPlatePolygons.shp
data/external/zahirovic2022/ContinentalPolygons/Global_EarthByte_GPlates_PresentDay_ContinentalPolygons.shp
data/external/zahirovic2022/Topologies/*.gpml
```

`scripts/fetch_gmt_model.py` obtains hash-pinned model support, not observations or GMTs. Dynamic geometry binds reconstructed continents and actual same-age plate/network GMT inputs to every 0–60 Ma anchor; original GPML verifies numerical frame and export completeness. Both `no_physics` and `physics` use the same frozen inputs; see [docs/gmt_tags.md](../docs/gmt_tags.md). All reconstruction files must use a consistent model and reference frame; see [docs/dynamic.md](../docs/dynamic.md). Continental shapefiles are also supported, including their attribute and GPlates sidecar files.

Paths and filenames can be changed in the configuration. The original workspace-specific configuration `configs/a40.json` references existing files outside this source distribution and is documented as a local reproduction recipe. `norn demo` generates fully synthetic inputs and needs no external dataset.

Do not commit raw observations, model archives or generated NPZ files. Each preparation run records the actual source hashes and declared observation semantics.
