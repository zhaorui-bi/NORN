# User-provided data

Raw observations and external plate models are not distributed with NORN. See [docs/data.md](../docs/data.md) for schemas and provenance requirements.

The generic `configs/reconstruction.json` expects the following user-supplied files:

```text
data/observations.csv
data/modern.dat
data/rotations.rot
data/static_polygons.gpmlz
```

Paths and filenames can be changed in the configuration. The original workspace-specific configuration `configs/a40.json` references existing files outside this source distribution and is documented as a local reproduction recipe. `norn demo` generates fully synthetic inputs and needs no external dataset.

Do not commit raw observations, model archives or generated NPZ files. Each preparation run records the actual source hashes and declared observation semantics.
