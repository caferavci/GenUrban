# Private input guide

The repository contains workflow code, not research datasets. Prepare your own
inputs in the directory selected by `GENURBAN_WORKSPACE`; its default is
`private_workspace` relative to the project. Keep inputs and generated outputs out
of version control.

## Directory layout

The following is a layout description, not a set of files shipped with the code:

```text
private_workspace/
├── ACS/
│   ├── 36_061.csv
│   ├── 32_003.csv
│   ├── 06_075.csv
│   └── 27_053.csv
├── census_mapping.csv         # Needed only when fetching missing ACS tables
├── New-york-city-boroughs/    # New York City OD CSVs
├── LasVegas/                 # Las Vegas OD CSVs
├── San_Francisco/            # San Francisco OD CSVs
├── Minneapolis/              # Minneapolis OD CSVs
├── regions/                  # Zone GeoJSON files
└── Overture/                 # Prepared building GeoJSON files
```

Only inputs for your selected cities are needed. Use the exact city names in
`GENURBAN_CITIES`; geographic labels and file names are not interchangeable.

| Configured city | State/county FIPS | Zone file | Prepared building file |
|:--|:--|:--|:--|
| `NYC` | `36` / `061` | `regions/new-york-city-boroughs.geojson` | `Overture/NYC_buildings.geojson` |
| `Las Vegas` | `32` / `003` | `regions/lasvagas.geojson` | `Overture/Las_Vegas_buildings.geojson` |
| `San Francisco` | `06` / `075` | `regions/sanfrancisco.geojson` | `Overture/San_Francisco_buildings.geojson` |
| `Minneapolis` | `27` / `053` | `regions/minneapolis.geojson` | `Overture/Minneapolis_buildings.geojson` |

The county inputs reflect the original workflow configuration; a city label does
not imply that its administrative boundary and ACS county boundary coincide.

## ACS county tables

Place one county table at `ACS/<state>_<county>.csv`. Preserve leading zeros in
state/county identifiers. The workflow reads the original descriptive column
labels, including:

```text
Estimate!!Total!!Total population
```

This field must contain the county population used for the population-total
check. Include the demographic marginals needed by the original population
prompt; a table containing only the total is not an equivalent demographic input.

If the table is already present, the helper can read it locally. To fetch a missing
table instead, provide your own Census configuration and a `census_mapping.csv`
with `Variable` and `Label` columns. The mapping identifies requested Census
variables and their descriptive labels; no mapping or Census response is bundled.
Ensure that the configured endpoint, variable definitions, and intended data
vintage agree. Never put credentials in the mapping file.

## OD flow tables

The original workflow has four training batches and one held-out batch per city.
Use these file stems in the corresponding city directory:

| City directory | File stem |
|:--|:--|
| `New-york-city-boroughs` | `new-york-city-boroughs.geojson Flow` |
| `LasVegas` | `lasvagas.geojson Flow` |
| `San_Francisco` | `sanfrancisco.geojson Flow` |
| `Minneapolis` | `minneapolis.geojson Flow` |

Append ` 6-9.csv`, ` 10-13.csv`, ` 14-17.csv`, and ` 18-21.csv` for training,
and ` 22-25.csv` for the held-out input. For example, the New York held-out file is
`New-york-city-boroughs/new-york-city-boroughs.geojson Flow 22-25.csv`.

Each CSV must contain:

| Field | Meaning |
|:--|:--|
| `Origin` | Origin zone identifier |
| `Destination` | Destination zone identifier |
| `flow_count` **or** numeric columns containing `Trips` | Flow value, or the values summed to construct it |

Zone identifiers should match the zone/building convention, such as `Region 0`.
The original model omits self-links and cannot use zones absent from the learned
training topology. Supply nonnegative, finite flow values and consistent zone
identifiers across all batches.

The filename suffixes are historical batch identifiers. Do not interpret them as
verified observation dates without consulting the underlying data headers. The
original notebook derives its temporal features from those file names rather than
expanding every dated hourly column into a separate observation. See
[Reproducibility](REPRODUCIBILITY.md#observation-construction).

## Zone boundaries

Provide a valid GeoJSON FeatureCollection readable by GeoPandas. Geometries must
have a defined coordinate reference system; the workflow converts them to
`EPSG:4326` for spatial joins.

The original implementation assigns identifiers by feature order: the first
feature becomes `Region 0`, the next `Region 1`, and so on. **Feature order must
agree with the OD identifiers.** Reordering the GeoJSON without updating the OD
labels changes the spatial meaning of the inputs.

## Building inputs

Prepared building GeoJSON is the most direct way to supply reproducible built
environment inputs. Each feature must have valid geometry and the attributes:

| Attribute | Purpose |
|:--|:--|
| `subtype` | Building category used by the original aggregation |
| `capacity` | Finite numeric capacity proxy |

The original helper constructs capacity as projected building area multiplied by
height. The prediction engine aggregates capacity by zone. It groups
`residential` and `apartments` as residential capacity and other subtypes as
commercial capacity; this is an implementation convention, not a validated
classification of every building's use.

The helper also supports acquiring Overture buildings when a prepared file is
absent. Live source coverage and schemas can change. Inspect the geometry,
coverage, height values, and computed capacity before interpreting a run. A
building-processing failure may trigger the notebook's documented density
fallback; fallback values are not observed building measurements.

## Before execution

Run `python run_workflow.py --check` from the repository root. This performs local
preflight checks only; it does not authenticate to a provider or prove that the
selected geographic inputs are substantively correct.

Confirm separately that you have permission to use the OD records and that all
geographic inputs describe the intended study area. The code does not redistribute
or grant access to provider data.
