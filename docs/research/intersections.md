# Ontario intersections and urban streets: the numbers the builder uses

What a real Ontario street and intersection measures, and the value the generator builds with.
Values marked **ASSUMED** were not confirmed in a primary source. Each row cites one; the
sources list is at the end. Code: `geometry/lanes.py`, `geometry/streetscape.py`,
`geometry/divided.py`, `geometry/network.py`, `geometry/furniture.py`,
`geometry/street_items.py`.

## 1. Lanes

| Item | Ontario value | Built | Source |
|---|---|---|---|
| Lane width, any road | 3.0 m to 4.0 m between lane lines | as below | [OTM11] §3.5 "Lane Lines" |
| Through lane, urban (Toronto) | 3.0 m target; 3.0-3.5 m range | 3.0 m local, 3.3 m collector | [TOR-LW] §3.1 |
| Curb lane | 3.3 m target; 3.5 m at 60 km/h | – | [TOR-LW] §3.1 |
| Suburban arterial lane (Halton, MTO) | 3.5-3.75 m | 3.6 m primary, 3.4 m secondary | **ASSUMED** (TAC GDG ch. 4 range [TAC]) |
| Left- or right-turn lane | 3.0 m, up to 3.3 m | the lanes share the carriageway equally | [TOR-LW] §3.2 |
| Lanes when OSM has no `width` | `lanes` x the class lane width | `geo/attributes.py` `LANE_WIDTH_M["na"]` | – |

## 2. Lines (OTM Book 11, 2000)

| Line | Ontario value | Built | Source |
|---|---|---|---|
| Lane line, urban | broken white, ~10 cm, 3 m line / 6 m gap | yes | [OTM11] §3.5 |
| Lane line, rural | broken white, ~10 cm, 3 m line / 9 m gap | yes | [OTM11] §3.5 |
| Lane line where lane changes are discouraged (beside a turn lane) | solid white ~10 cm | solid where one lane is turn-only (`turn:lanes`) | [OTM11] §3.5 |
| Directional dividing (centre) line | yellow: single solid, single broken (3 m / 6 m), double solid, or solid + broken | – | [OTM11] §3.4 |
| Double line | where passing must be shown, at 70 km/h or more, or for emphasis; lines one line-width apart | double solid on multi-lane roads and town arterials, and on rural roads (unchanged from before) | [OTM11] §3.4 |
| Centre line on urban local streets | not necessary on low-volume local roads | none on town residential streets | [OTM11] §3.4 |
| Centre line through a major intersection | discontinued | stops at the stop lines | [OTM11] §3.4 |
| Lane lines and centre line at urban intersections | end at the stop line | yes | [OTM11] §3.8 |
| Edge line | solid ~10 cm; white on the drivers' right, yellow on their left | white / yellow (a one-way carriageway's median side); none where a curb is the edge | [OTM11] §3.5; curbed omission **ASSUMED** |
| Guide lines through a junction | broken white 20 cm, 1 m / 1 m | not built | [OTM11] §3.8 |

## 3. Stop lines, crosswalks, arrows

| Item | Ontario value | Built | Source |
|---|---|---|---|
| Stop line | solid white, 30-60 cm wide, from the right pavement edge to the centre line or median (one-way: to the left edge) | 45 cm, across the approach lanes | [OTM11] §3.8 |
| Stop line, no crosswalk | 1.25-3 m before the projected near edge of the crossing road | 1.5 m (route), 2.0 m (side roads) | [OTM11] §3.8 |
| Stop line behind a crosswalk | 1 m from the crosswalk line | 1.0 m | [OTM11] §3.8 |
| Crosswalk lines | two solid white parallel lines 10-20 cm (60 cm above 60 km/h) | 20 cm lines | [OTM11] §3.8 |
| Crosswalk width | at least 2.5 m; 3-4 m typical urban | 3.0 m | [OTM11] §3.8 |
| Crosswalk position | its downstream edge at least 60 cm in from the projected curb line of the cross street | 1.0 m beyond the crossing road's edge | [OTM11] §3.8 |
| Crosswalk at signals (rule-made) | marked where pedestrians use the signal | ladder: zebra bars 0.5 m / 0.6 m gaps between the two lines | bars **ASSUMED** (municipal practice) |
| Mapped crossings | OSM `crossing=*`, `crossing:markings=*` | zebra or two lines per the tags | furniture.py |
| Turn arrows | lane-use arrows (Fig. 53), in turn bays and where a lane becomes turn-only; Fig. 29 shows 3 m arrows | per lane from `turn:lanes`, tip 6 m behind the stop line, 4.2 m long | [OTM11] §3.12, Fig. 29; setback **ASSUMED** |
| Arrow spacing when repeated | at least 3 x the symbol height on low-speed roads | one arrow per lane | [OTM11] §3.12 |

## 4. Signals (for the signal builder)

| Item | Ontario value | Source |
|---|---|---|
| Heads per approach | at least two (Highway Traffic Act s.124(20)) | [TOR-SIG] §3.6 |
| Primary head | highway type with backboard, far right of the approach, 12-55 m beyond the near stop line (15 m desirable) | [TOR-SIG] §3.6.1 |
| Primary head height | at least 4.5 m clear of the pavement; standard 4.9 m | [TOR-SIG] §3.6.1 |
| Secondary head | far left of the approach | [TOR-SIG] §3.6.2 |
| Secondary / auxiliary head clear of the roadway | bottom at least 2.75 m above the sidewalk; standard 4.9 m | [TOR-SIG] §3.6.2-3 |
| Auxiliary head | near right, or where needed | [TOR-SIG] §3.6.3 |
| Pedestrian head | at least 2.75 m high; within the crosswalk lines' extension | [TOR-SIG] §3.6.4 |
| Arrow lens | 300 mm | [TOR-SIG] §3.4 |
| Pole set-back from the curb | at least 1.0 m where possible | [TOR-SIG] §3.9 |
| Mast arm length | 1.2-7.6 m (standard lengths); 7.6 m single arm on a 9.1 m pole | [TOR-SIG] Tables 1-2 |
| Push buttons | within 3.0 m of the crosswalk | [TOR-SIG] §3.9 |
| When the generator signalizes | OSM `highway=traffic_signals`; else a town junction where the approach and the road it crosses are both at least collectors and one is an arterial | **ASSUMED** rule (`street_items.signalized`) |

The furniture layer places one `signal:traffic` item per signalized route approach at the near
right corner, 1 m beyond the road edge; the network adds one per signalized side-road approach.

## 5. Corners, curbs, boulevards, sidewalks

| Item | Ontario value | Built | Source |
|---|---|---|---|
| Corner (curb return) radius, local-local | 7.5 m (Richmond Hill); 5.0 m minimum (Pickering) | 9 m residential | [RH-C] Table C-3; [PICK]; built value **ASSUMED** |
| Corner radius, collectors | 10-12 m | 12 m tertiary | **ASSUMED** (TAC ch. 9 range [TAC]) |
| Corner radius, arterials and truck routes | 12-15 m | 13-15 m | **ASSUMED** (TAC ch. 9 range [TAC]) |
| Barrier curb | 150 mm reveal, 150 mm top, 450 mm deep | 150 mm reveal, 150 mm top | [HAM-RD] RD-103 |
| Curb and gutter, per side | 0.5 m allowed | 0.30 m gutter + curb | [HAM-RD] RD-113.01 |
| Curb face in the game | vertical (40 mm batter) | rises over 180 mm: AC tests tyre contact at one point per wheel | [OVERTAKE] |
| Boulevard (grass) | 1.75 m, 2 % up to the property line | 1.75 m, 2 % | [HAM-RD] RD-113.01 |
| Sidewalk | 1.5 m typical; 1.8 m against the curb; 2.0 m (Mississauga) | 1.5 m local, 1.8 m collectors and arterials | [STREET] §4; [MISS] |
| Sidewalk crossfall | 2 % towards the road | 2 % | [OPSD-310] |
| Behind the sidewalk | at least 0.3 m | 0.3 m grass | [OPSD-310] |
| Driveway curb cut | curb drops to a 25 mm lip; 300 mm transitions | 25 mm; 0.3 m transitions on side roads | [HAM-RD] RD-106 |
| Oakville driveways | constant width, curb and sidewalk continuous | sidewalk carries on; the boulevard becomes a concrete apron | [OAK-DW] |
| Where sidewalks are | OSM `sidewalk=*`; Oakville's open Sidewalks layer; Burlington's Pedestrian Network (switch); else urban collectors and arterials both sides, local streets none, rural none | as stated | [OAK-SW]; [BUR-PN] |

## 6. Medians, left-turn lanes, cul-de-sacs

| Item | Ontario value | Built | Source |
|---|---|---|---|
| Raised median, urban | 1.2-6 m, curbed; 4.8 m+ where a left-turn lane is cut in | raised curbed island up to 8 m wide: concrete top up to 2 m, grass above | **ASSUMED** (TAC ch. 8 range [TAC]) |
| Wide median | grass, depressed on rural divided highways | grass at grade (the terrain) | **ASSUMED** |
| Left-turn lane | 3.0-3.3 m | from OSM lanes / `turn:lanes` | [TOR-LW] §3.2 |
| Cul-de-sac bulb, curb radius | 13.0 m (Hamilton RD-116.01), 15.0 m (Richmond Hill C-15); measured 9-14 m | 12 m (9-15 m from OSM `diameter`) | [HAM-RD]; [RH-C]; [STREET] §4 |
| Turning loop island | – | 5 m radius | **ASSUMED** |

## 7. Urban or rural

A road gets the urban section (curb, boulevard, sidewalk) when OSM says so (a sidewalk tag,
`lit=yes` at 70 km/h or less, or a multi-lane arterial at 70 km/h or less), when a municipal
sidewalk line runs beside it, or when it lies in a subdivision (500 m of residential streets
within 150 m and 60 km/h or less). `sidewalk=no` or `lit=no` keeps it rural. **ASSUMED** rule;
see `geometry/streetscape.py`.

## Sources

- [OTM11] Ontario Traffic Manual Book 11, Pavement, Hazard and Delineation Markings (MTO, March 2000): https://www.library.mto.gov.on.ca/SydneyPLUS/Sydney/ViewRecord.aspx?template=Books&record=7fede97e-4361-4435-85dc-4c94d04c6131&lang=en-US (copy read: https://www.atstraffic.ca/wp-content/uploads/2022/03/Book-11-Pavement-Hazard-and-Delineation-Markings.pdf)
- [TOR-SIG] City of Toronto TTR 801.320, Location of Traffic Signal Equipment (Jan 2012, archived; cites HTA s.124): https://www.toronto.ca/wp-content/uploads/2023/08/943b-ecs-specs-tcas-TTR-801.320-Jan2012.pdf
- [TOR-LW] City of Toronto Lane Widths Guideline v3.0: https://www.toronto.ca/wp-content/uploads/2025/11/96f7-Lane-Widths-Guideline-Version-3.0.pdf
- [TAC] TAC Geometric Design Guide for Canadian Roads (2017): https://www.tac-atc.ca/en/publications/ptm-gdg-e/ (ranges taken from the repo's earlier research, not re-read)
- [HAM-RD] City of Hamilton road standard drawings (RD-103, RD-106, RD-113.01, RD-116.01): https://www.hamilton.ca/sites/default/files/2022-04/pedpolicies-construction-material-specs-roaddrawings-feb2021.pdf
- [RH-C] Richmond Hill Design Standards, Division C: https://www.richmondhill.ca/en/learn-more/resources/Design-Standards/Division-C-Transportation-and-Roadworks.pdf
- [PICK] City of Pickering road standards: https://pickering.ca/media/0evdypkt/roads.pdf
- [MISS] Mississauga Standard Drawing 2240.040: https://www.mississauga.ca/wp-content/uploads/2020/08/12144659/Standard-Drawing-2240040.pdf
- [OPSD-310] OPSD-M310.010 concrete sidewalk: https://whitchurch.civicweb.net/document/74782/
- [OAK-DW] Town of Oakville driveway permits procedure: https://www.oakville.ca/town-hall/policies-procedures/use-of-municipal-right-of-ways-policy/driveway-permits-procedure/
- [OAK-SW] Town of Oakville Sidewalks layer (Open Government Licence - Town of Oakville): https://maps.oakville.ca/oakgis/rest/services/ArcGISOnline/AGO_OpenData/MapServer/23
- [BUR-PN] City of Burlington Pedestrian Network (Open Data Terms of Use, revocable): https://mapping.burlington.ca/arcgisweb/rest/services/COB/RoadStructures/MapServer/40
- [STREET] measurements made for this project (not in the repo): Burlington Pedestrian Network widths (90 % are 1.5 m), Hamilton and Brampton curb lines round cul-de-sac bulbs (median 9-12 m).
- [OVERTAKE] AC's single tyre contact point (second-hand): https://www.overtake.gg/threads/assetto-corsa-competizione-the-5-point-tyre-model-blog.171148/page-8
