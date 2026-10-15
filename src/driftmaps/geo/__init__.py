"""Road data and routing: Overpass corridor fetch, OSM edge graph, waypoint router, attributes.

PLAN §5.1 / docs/research/geo.md §1. Typical use:

    route = route_from_pins([(lat, lon), ...], cache_dir)   # fetch (cached) + snap + route
    route.save(track_dir / "route.json")                    # the build reads only this
    pl = Route.load(path).polyline(1.0)                     # lon/lat + per-point attributes

Road data is (c) OpenStreetMap contributors, ODbL 1.0 (`overpass.OSM_ATTRIBUTION`).
"""

from driftmaps.geo.project import LocalProjection
from driftmaps.geo.router import Route, RouteError, RouterOptions, route_from_pins

__all__ = ["LocalProjection", "Route", "RouteError", "RouterOptions", "route_from_pins"]
