# Scout

Scout is an aerial knowledge acquisition system for drones.

The first Scout capability is a VPS, or visual positioning system: given an
image from a drone, Scout searches a georeferenced aerial image database and
returns the most likely location. The larger goal is broader than positioning.
Scout should acquire, extract, and reason over knowledge from aerial imagery:
terrain features, infrastructure, hazards, scene state, objects, temporal
change, and operational context.

This repository is a fork of the FastAPI full-stack template, adapted into a
full-stack Scout MVP with a FastAPI backend, React dashboard, Postgres metadata,
Redis vector search, and Weave tracing.

## Current MVP

The current implementation builds and searches an image embedding database for
San Francisco aerial imagery.

- Ingest georeferenced aerial imagery with known bounds.
- Tile imagery into overlapping patches.
- Store patch metadata and geospatial bounds in Postgres.
- Embed patches with a deterministic local image descriptor.
- Store vectors in Redis Stack for nearest-neighbor search.
- Upload a query image and return likely coordinates plus visual match previews.
- Generate holdout query examples that are shifted/zoomed crops, not exact
  duplicates of indexed images.
- Emit traces with Weave when `WEAVE_ENABLED=True` and `WANDB_API_KEY` is set.

The current embedding model is:

```text
scout-pil-equalized-grayscale-32x32-v1
```

It is a lightweight 1024-dimensional PIL/NumPy descriptor intended for MVP
plumbing and smoke tests. It is not the final VPS model. Future model candidates
include CLIP/SigLIP-style vision embeddings, DINOv2, geospatial foundation
models, YOLO-family detectors, SAM-family segmentation models, and task-specific
fine-tuned aerial imagery models.

## Scout As Knowledge Acquisition

Scout should maintain an evolving world model of an area, not just a vector
index. Each image, detection, feature, and agent conclusion becomes structured
knowledge that can be searched, compared, audited, and acted on.

Core knowledge objects:

- **Imagery**: raw drone frames, satellite images, orthophotos, map tiles, and
  derived crops.
- **Position evidence**: visual matches, candidate coordinates, confidence,
  geospatial bounds, and traceable source patches.
- **Scene features**: roads, intersections, rooftops, water, vegetation,
  construction, terrain, landing zones, smoke, debris, vehicles, people, and
  other operationally relevant features.
- **Object detections**: model outputs from detectors such as YOLOv8n or later
  aerial-specific detectors.
- **Temporal observations**: change over time, anomaly events, repeated
  sightings, and confidence deltas.
- **Operational actions**: alerts, human review queues, authority notifications,
  mission recommendations, and audit logs.

The system should answer questions such as:

- Where is this drone image most likely located?
- What important features are visible in this aerial image?
- What changed since the last known image of this area?
- Is there smoke, fire, flooding, blocked road access, or other urgent evidence?
- What should be escalated to a human operator or external authority?

## Multi-Agent Orchestration

Scout will use multiple specialized agents coordinated around a shared aerial
knowledge graph, image store, vector index, and event bus. Agents should be
small, observable, and task-specific. Each agent receives evidence, produces
structured outputs, and records traces through Weave.

Planned orchestration pattern:

- **Coordinator agent**: receives imagery, mission context, and operator goals;
  decides which specialist agents to run.
- **Shared memory**: Postgres for structured metadata, Redis for fast vector and
  cache lookups, object storage/local storage for imagery, and later a graph
  layer for relationships.
- **Model adapters**: isolated services for embedding, detection, segmentation,
  OCR, geocoding, and change detection.
- **Policy layer**: controls escalation, confidence thresholds, rate limits,
  privacy boundaries, and authority notification rules.
- **Human review loop**: routes uncertain or high-impact events to operators
  before irreversible actions.
- **Traceability**: every agent run should record inputs, model version, output,
  confidence, and downstream action in Weave.

## Agent Use Cases

1. **Visual localization agent**: estimates drone position by matching a query
   frame against georeferenced aerial patches.
2. **VPS confidence agent**: evaluates whether localization evidence is strong
   enough to trust or whether GPS/manual review is needed.
3. **Smoke detection agent**: detects smoke plumes, estimates spread direction,
   and escalates likely fire events.
4. **Fire hotspot agent**: combines visual smoke/fire evidence with thermal or
   external signals when available.
5. **Authority notification agent**: prepares structured alerts for fire
   departments, police, emergency operations, or internal dispatch teams.
6. **Object detection agent**: runs YOLOv8n or a later aerial detector for cars,
   trucks, boats, people, aircraft, heavy equipment, and other classes.
7. **Road blockage agent**: identifies blocked roads, stalled traffic, debris,
   emergency vehicles, or inaccessible routes.
8. **Landing zone agent**: finds candidate safe landing or emergency landing
   zones based on flatness, obstacles, roads, people, and restricted areas.
9. **Infrastructure inspection agent**: detects damage or anomalies around
   bridges, towers, substations, rooftops, solar farms, and rail corridors.
10. **Construction progress agent**: compares imagery over time to measure
    construction state, material staging, excavation, and equipment movement.
11. **Flood detection agent**: identifies water accumulation, coastline change,
    flooded streets, and water encroachment near infrastructure.
12. **Vegetation and fuel-load agent**: maps vegetation density, dry brush, tree
    canopy, and fire-risk features near assets.
13. **Change detection agent**: compares new imagery against prior indexed
    imagery and flags meaningful scene changes.
14. **Anomaly detection agent**: finds unusual patterns without a fixed class
    list, such as unexpected gatherings, unknown objects, or sudden surface
    changes.
15. **Perimeter monitoring agent**: tracks boundaries around restricted zones,
    events, fires, construction sites, or security-sensitive areas.
16. **Search-and-rescue agent**: scans imagery for people, vehicles, signals,
    shelters, trails, and plausible movement corridors.
17. **Maritime awareness agent**: detects boats, wakes, docks, shoreline
    activity, and waterway obstructions.
18. **Urban feature extraction agent**: extracts intersections, crosswalks,
    roof shapes, parking lots, parks, sports fields, and landmarks useful for
    localization and planning.
19. **Dataset curation agent**: selects high-value imagery for indexing,
    removes duplicates, balances geography, and creates holdout evaluation
    sets.
20. **Model evaluation agent**: runs benchmark queries, measures localization
    error, detection quality, drift, and regression risk across model versions.

## Data Sources

The current SF seed uses USGS NAIP Plus imagery. Additional candidate sources
for future ingestion adapters include:

- USGS NAIP Plus / The National Map imagery services.
- NOAA aerial and coastal imagery archives.
- OpenAerialMap, where suitable openly licensed imagery exists.
- DataSF and San Francisco GIS imagery resources.
- Sentinel-2 or Landsat for low-resolution context and high-altitude tests.
- Future drone-collected Scout imagery with known pose or reviewed ground truth.

## Local Development

Start Docker Desktop, then run:

```bash
docker compose up --build
```

The dashboard is available at:

```text
http://localhost:5173
```

The API is available at:

```text
http://localhost:8000
```

Set local secrets in `.env`. The `.env` file is intentionally ignored by git.
Use `.env.example` as the safe template.

## Seed San Francisco

Seed or rebuild the SF index from the backend container:

```bash
docker compose exec backend python -m app.scout.seed_sf --clear
```

For a faster first pass:

```bash
docker compose exec backend python -m app.scout.seed_sf --clear --profile quick --chips 1 --image-size 512
```

The expanded profile builds multiple SF chips across different apparent
altitude scales and creates holdout examples:

```bash
docker compose exec backend python -m app.scout.seed_sf --clear --profile expanded --examples 10
```

## Weave Tracing

Weave is enabled by default:

```text
WEAVE_ENABLED=True
WEAVE_PROJECT=scout
```

Set `WANDB_API_KEY` in local `.env`, then restart the backend:

```bash
docker compose up -d --build backend
```

Scout traces cover embedding, indexing, import, and image search operations.

## Redis

Redis Stack is used for vector search. The local Docker override maps Redis to
host port `6380` to avoid conflicts with any Redis already running on `6379`.

```text
redis://localhost:6380/0
```

Inside Docker services, the app uses:

```text
redis://redis:6379/0
```

## Verification

Useful local checks:

```bash
cd backend && uv run ruff check app && uv run ty check app
npm --prefix frontend run build
docker compose up -d --build backend frontend
```

## Roadmap

- Replace the MVP descriptor with a stronger aerial/geospatial embedding model.
- Add dedicated model services for object detection and segmentation.
- Add an evaluation harness with known-location query sets and localization
  error metrics.
- Add agent run tables, agent outputs, confidence policies, and review queues.
- Add imagery source adapters beyond USGS NAIP Plus.
- Add map visualization, geofences, mission timeline views, and operator review
  workflows.
- Add authority notification integrations behind explicit policy and human
  approval controls.

## License

This project inherits the upstream template license unless replaced by a Scout
project license.
