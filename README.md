# Assetto Corsa Map Generator

Turn real roads into Assetto Corsa tracks: pick a start and an end point, and get a driveable
track built from OpenStreetMap road data and real elevation.

Status: **M0 (format spike)**. See [`docs/PLAN.md`](docs/PLAN.md) for the full plan and
[`docs/checklists/M0.md`](docs/checklists/M0.md) for the in-game test checklist.

## Development

```bash
uv sync            # Python 3.12 environment + dev tools
uv run pytest      # tests
uv run drift-maps --help
```

Road data © OpenStreetMap contributors (ODbL). Not affiliated with or endorsed by Kunos
Simulazioni; Assetto Corsa is their trademark, named only to state compatibility.
