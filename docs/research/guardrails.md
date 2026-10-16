# Guardrail warrants (Ontario): where design rules say a barrier is required

Status: v1, 2026-10-15. Scope: **automatic** guardrail placement for the Ontario test roads (Snake Rd and
Sydenham Rd in Hamilton, Forks of the Credit Rd in Caledon), and any other road we build. It follows on
from `scenery.md` §5. That section found that OSM almost never maps guardrails, so the builder now places
none. The user wants rails wherever real road-design rules would call for them.
**VERIFIED** means read in the primary document, or in a state DOT's reproduction of it, on the date
above. **ASSUMED** means it is our own choice or inference and has not been checked against a source.
Downloaded PDFs were kept out of the repo. The numbers below were transcribed from them, and the tables
that pdftotext garbled were checked against rendered page images.

**TL;DR**
- **Use the Ontario rules.** The MTO *Roadside Design Manual* (RDM) is free and Ontario-specific. It
  gives single-value tables for clear zone, curve factor, runout length and barrier hardware. Use it as
  the rulebook. Use AASHTO 2011, through state DOT reproductions, only for the one thing MTO leaves to a
  benefit/cost spreadsheet: the embankment height-vs-slope warrant. The TAC GDG Ch. 7 is paywalled. Its
  one table we could read (clear zone, 90-110 km/h) matches the MTO/AASHTO values.
- **The core rule.** A barrier is warranted where a **hazard lies inside the desirable clear zone (DCZ)**.
  The DCZ depends on design speed, traffic and side slope. On the outside of curves with R ≤ 900 m it is
  multiplied by 1.1-1.5. For our 50-60 km/h roads the DCZ is **3-5.5 m** from the edge of the travelled
  lane, and up to about **8.5 m** on the outside of hairpins.
- **The hazards we can detect automatically:**
  1. **Critical fill slopes** (steeper than 3H:1V) taller than a slope-dependent height: 0.6 m for a
     drop-off, 0.9 m at 1.5H:1V, 1.8 m at 2H:1V, 2.7 m at 2.5H:1V, 6 m at 3H:1V. Measured with lidar.
  2. **Water** at least 1.0 m deep inside the DCZ. From OSM water polygons and rivers.
  3. **Bridge ends.** Rails on all four corners of an undivided road.
  4. Optionally, **rough rock cuts**.

  Trees, poles and ditches are **not** triggers. By the letter of the rules they would put a rail along
  every wooded roadside.
- **Rail length.** Length of need = hazard length + **La = E·(1 − A/B)** upstream for each traffic
  direction. E is the runout length: 21-49 m at ≤ 60 km/h, 38-70 m at 70-80 km/h. Join gaps of 50 m or
  less. Minimum run is **25 m** including terminals. Cut every rail at side roads and driveways.
- **Hardware.** Ontario's standard is the **Type M steel-beam guide rail**: a W-beam with its top at
  785 mm, posts every 1.905 m, 200 mm blocks. Its face sits at the **edge of the shoulder**, at least
  4.25 m from the centreline and at most 4.0 m from the edge of the travelled lane. Cable rail is for long
  straight runs on high-speed highways only.

---

## 1. Sources and how much to trust them

| Source | What we used | Status |
|---|---|---|
| **MTO Roadside Design Manual, draft chapters 1-5, "TBD, 2023"**, [Ch. 2](https://tcp.mto.gov.on.ca/sites/default/files/2022-12/RDM%20Chapter%202%20Draft%202023%20Clean.pdf), [Ch. 3](https://tcp.mto.gov.on.ca/sites/default/files/2022-12/RDM%20Chapter%203%20Draft%202023%20Clean.pdf), [Ch. 4](https://tcp.mto.gov.on.ca/sites/default/files/2022-12/RDM%20Chapter%204%20Draft%202023%20Clean.pdf) | Clear zone (Table 2-2), curve factors (2-3), fill/cut slope tables (2-4 to 2-7), water (2.3.10), culverts (2.3.11), runout and length of need (2-16, Fig. 2-14/2-15), shy line and flare (3-1, 3-2), barrier hardware (Ch. 4) | VERIFIED (read, tables checked against rendered pages) |
| **MTO RDM proposed update (Edition 3)**, [TCP notice 000-0315](https://tcp.mto.gov.on.ca/notice/000-0315), posted 2026-09-16, comments closed 2026-10-14. [Ch. 2](https://tcp.mto.gov.on.ca/sites/default/files/2026-09/RDM%20Part%202%20-%20Ch%202.pdf), [Ch. 3](https://tcp.mto.gov.on.ca/sites/default/files/2026-09/RDM%20Part%203%20-%20Ch%203.pdf), [App. A](https://tcp.mto.gov.on.ca/sites/default/files/2026-09/RDM%20Part%207%20-%20Appendix%20A.pdf) | Same deterministic DCZ table and curve factors **below 90 km/h**. A new relative-risk DCZ (NCHRP 1097) for ≥ 90 km/h. Severity-index tables (App. A) used to cross-check the embankment warrant. Max 4.0 m offset. | VERIFIED (read). Still a proposal, not final. |
| **TAC Geometric Design Guide for Canadian Roads, Ch. 7 Roadside Design (2017, revised 2019)**, [product page](https://www.tac-atc.ca/en/publications/ptm-geodes7-e) ($82, 116 pp.) | Only Table 7.3.1 for 90/100/≥110 km/h, from the [2019 revision summary](https://www.tac-atc.ca/wp-content/uploads/GDG_2019_revisions_e.pdf). Ranges, same as the AASHTO metric values. The 2025 revision did not touch Ch. 7. | VERIFIED (excerpt only). The rest is not read. |
| **AASHTO Roadside Design Guide, 4th ed. 2011 (+2015 errata)** | Clear zone table and K_cz in feet, reproduced by the [Kentucky Transportation Center](https://kp.uky.edu/knowledge-portal/articles/roadside-safety-knowledge-book-clear-zone-clear-zone-2/). The critical-slope height table is reproduced by [NJDOT BDC18MR-01](https://www.nj.gov/transportation/eng/documents/BDC/pdf/BDC18MR-01Attachment.pdf) ("based on the Roadside Design Guide, AASHTO, 2011"). | VERIFIED via reproductions. The book itself was not read. No 5th edition was found. |
| **AASHTO "comparative risk warrants for embankments" chart** (1977/1996 lineage) | [Ohio L&D Vol. 1 Fig. 601-1](https://dam.assets.ohio.gov/image/upload/transportation.ohio.gov/roadway/ld1/figures/601-1) (Apr 1999). [NCDOT Roadway Design Manual Pt 1 Ch 3](https://connect.ncdot.gov/resources/safety/Teppl/TEPPL%20All%20Documents%20Library/Roadside%20Barriers.pdf), Fig. 1 and Tables 2-3 (non-traversable hazard warrants). | VERIFIED (curve read from the rendered chart, ±0.2 m) |
| Other DOTs: [TxDOT RDW 17.3.2](https://www.txdot.gov/manuals/des/rdw/chapter-17-roadside-safety-design-and-roadside-saf/17-3-longitudinal-barriers/17-3-2-applications.html), [MaineDOT roadside practices](https://www.maine.gov/dot/sites/maine.gov.dot/files/inline-files/Roadside%20Design%20Practices%20and%20Procedures.pdf) | Water > 2 ft deep, drop-off steeper than 1:1 and deeper than 2 ft, lateral placement, gap rules | VERIFIED |
| FHWA/Calspan 1986, *Rollover potential of vehicles on embankments* ([TRID 269061](https://trid.trb.org/View/269061)) | Fills should not be steeper than 3:1 at any height (small-car rollover) | VERIFIED (abstract) |
| **Ontario HTA s.128(1)**, [text via J.J. Keller](https://jjkellercompliancenetwork.com/regsense/section-128-rate-of-speed) (official: ontario.ca/laws), [OSM `maxspeed:type=CA-ON:urban`](https://wiki.openstreetmap.org/wiki/Tag:maxspeed:type%3DCA-ON:urban) | Default 50 km/h inside a local municipality or built-up area; 80 km/h in the listed exceptions | VERIFIED (unofficial copy) |
| **MTO memo SCB-HDO-2026-01**, [Design and posted speed on freeways](https://tcp.mto.gov.on.ca/sites/default/files/2026-06/Policy%20Memo%20-%20Design%20and%20Posted%20Speed%20on%20Freeways%20061526%20%28002%29%20-ED%20AB_1.pdf) (2026-06-15) | "Historically, Ontario applied a Factor of Safety of 20 km/h" (design speed − posted). Design speed = posted is "normal practice on Secondary highways". | VERIFIED |

**Caveat that shapes every choice below (VERIFIED, RDM 2.3.1 policy).** MTO says obstacles inside the DCZ
"**should be shielded**" only on *Major Capital Expansion* projects. On reconstruction and rehabilitation
projects they are *evaluated* on benefit/cost. Old municipal escarpment roads were not built to these
tables. So the rules describe what a designer would put there today, which is not necessarily what
stands there. The rules can over-predict on old roads, and they can miss rails added later after
collisions. §6 plans for calibration against the user's own knowledge of the three roads.

---

## 2. Inputs we have, and how each maps to a rule input

| Rule input | Our data | Mapping |
|---|---|---|
| **Design speed V_d** | OSM `maxspeed` (Sydenham 50/60, Forks 50, Snake: missing) | V_d = posted + 10 km/h, rounded up to a table row. **ASSUMED** middle ground: MTO historically used +20 on provincial highways and +0 on Secondary highways (memo above, VERIFIED). If `maxspeed` is missing, use the class default in §7 (ASSUMED). The legal default is 50 km/h in any local municipality (HTA 128(1)(a), VERIFIED), but rural roads are usually signed 60-80. |
| **Traffic (AADT)** | OSM `highway` class | One representative AADT per class (§7, **ASSUMED**). It picks the row in both the DCZ table (< 750 / ≥ 750 / ≥ 1,500 / ≥ 6,000) and the runout table (< 1k / 1-5k / 5-10k / > 10k). |
| **Side slope and drop height** | 1 m HRDEM DTM (`elevation.sources`), or the built terrain (`build._ground_fn`) | A cross profile every 1 m of station, from the edge of the travelled lane out to DCZ + 10 m (§7 step 4) |
| **Curve radius, outside of curve** | `RoadModel.curvature` (signed; + = turning left) | R = 1/\|κ\|. The outside is the side opposite the turn. |
| **Water** | OSM `natural=water`, `waterway=riverbank`, `water=*`, `landuse=reservoir`, `waterway=river/canal`; Land Cover 2020 class 18 (30 m, confirmation only) | Distance from the lane edge to the water edge |
| **Bridges** | OSM `bridge=*` → `geometry/bridges.py` deck station ranges | Bridge-end hazard (§4.3) |
| **Culverts** | OSM `tunnel=culvert` on waterways | Caught by the slope rule through the lidar fill drop (§4.4) |
| **Rock cuts** | Lidar upslope; OSM `natural=cliff/bare_rock/scree` | Optional rule (§4.5) |
| **Gaps** | `route.json` `stubs` (side roads), OSM `highway=service` (+`service=driveway`) touching the route | Cut the rail (§5.4) |
| **Shoulder** | `RoadModel.half_width`, `RoadModel.shoulder` | Barrier face offset (§5.5) |

---

## 3. Desirable clear zone (DCZ)

### 3.1 Base DCZ: MTO Table 2-2 (VERIFIED, 2023 draft p. 2-6; unchanged below 90 km/h in the 2026 proposal)

All values are in metres from the **edge of the travelled way**. Shoulders, rounding and recoverable
slopes all count inside the DCZ (RDM 3.1.2). "Fill" is a falling foreslope (negative); "cut" is a rising
foreslope (positive).

| V_d (km/h) | AADT | Fill 3H:1V | Fill 4H:1V-5H:1V | Fill 6H:1V or flatter | **10H:1V or flatter** | Cut 6H:1V or flatter | Cut 5H:1V-4H:1V | Cut 3H:1V |
|---|---|---|---|---|---|---|---|---|
| ≥ 110 | ≥ 6,000 | Note 1 | 14 | 10.5 | 9.5 | 9 | 9 | 7.5 |
| | ≥ 1,500 | Note 1 | 13 | 10 | 9 | 8.5 | 7.5 | 6 |
| | ≥ 750 | Note 1 | 11 | 8 | 7 | 6.5 | 6 | 5 |
| | < 750 | Note 1 | 8 | 6 | 5.5 | 5 | 5 | 3.5 |
| 100 | ≥ 6,000 | Note 1 | 13.5 | 10 | 9 | 8.5 | 8 | 6.5 |
| | ≥ 1,500 | Note 1 | 12 | 9 | 8.5 | 8 | 6.5 | 5.5 |
| | ≥ 750 | Note 1 | 10 | 7.5 | 7 | 6.5 | 5.5 | 4.5 |
| | < 750 | Note 1 | 7.5 | 5.5 | 5.5 | 5 | 4.5 | 3.5 |
| 90 | ≥ 6,000 | Note 1 | 10 | 7.5 | 7.5 | 7.5 | 6.5 | 5.5 |
| | ≥ 1,500 | Note 1 | 9 | 6.5 | 6.5 | 6.5 | 5.5 | 5 |
| | ≥ 750 | Note 1 | 7.5 | 5.5 | 5.5 | 5.5 | 5 | 3.5 |
| | < 750 | Note 1 | 5.5 | 4.5 | 4 | 3.5 | 3.5 | 3 |
| 70-80 | ≥ 6,000 | Note 1 | 8.5 | 6.5 | 6.5 | 6.5 | 6 | 5 |
| | ≥ 1,500 | Note 1 | 8 | 5.5 | 5.5 | 5.5 | 5 | 4.5 |
| | ≥ 750 | Note 1 | 6 | 5 | 5 | 5 | 4.5 | 3.5 |
| | < 750 | Note 1 | 4.5 | 3.5 | 3.5 | 3.5 | 3 | 3 |
| ≤ 60 | ≥ 6,000 | Note 1 | 5.5 | 5 | 5 | 5 | 5 | 5 |
| | ≥ 1,500 | Note 1 | 5 | 4.5 | 4.5 | 4.5 | 4.5 | 4.5 |
| | ≥ 750 | Note 1 | 4.5 | 3.5 | 3.5 | 3.5 | 3.5 | 3.5 |
| | < 750 | Note 1 | 3 | 3 | 3 | 3 | 3 | 3 |

- **Note 1, non-recoverable slopes (VERIFIED).** On fill slopes between 3H:1V and 4H:1V a vehicle "will
  not likely stop … and should be expected to travel to the bottom of the slope or beyond". The DCZ
  therefore extends past the toe by a **recovery area** = max(DCZ(10H:1V) − (shoulder + ½ rounding),
  3 m). Worked example from RDM Fig. 3-2: at 110 km/h and AADT 4,000, DCZ(10H:1V) = 9.0, shoulder 3.0,
  rounding 1.5, so the recovery area is 5.25 m.
- **Note 2 (VERIFIED).** Positive slopes up to 2H:1V at bridge abutments need no protection.
- **The 90 km/h row** is in the 2023 draft. The 2026 proposal moves design speeds of 90 km/h and up to a
  relative-risk DCZ (NCHRP 1097, Tables 2-2 to 2-4 there). For continuous hazards (obstacle spacing
  < 45 m) it gives **12-21 m**, much wider than the deterministic 7.5-10.5 m. **Decision (ASSUMED): use
  the deterministic table above for all speeds.** Our roads are below 90 km/h. If we ever build 90+ km/h
  roads, consider the wider relative-risk values.
- **TAC cross-check (VERIFIED, excerpt).** At 90 km/h, AADT 1,500-6,000, 6:1 foreslope, TAC 7.3.1 gives
  6.0-6.5 and MTO gives 6.5. At 100 km/h, AADT > 6,000, 5:1-4:1 foreslope, TAC gives 11.0-13.5 and MTO
  gives 13.5. MTO picks the top of the TAC/AASHTO range.
- **Column choice (ASSUMED).** Classify the ground from the lane edge out to DCZ(10H:1V), excluding any
  segment already flagged as a critical slope (§4.1). Use its mean slope: fill 4-5:1 → "Fill 4H:1V-5H:1V";
  fill 6-10:1 → "Fill 6H:1V"; within ±10:1 → flat; rising → the matching cut column. A **mean** fill
  steeper than 4H:1V is non-recoverable: the DCZ then reaches the toe plus the recovery area (Note 1).

### 3.2 Outside of curves: MTO Table 2-3, K_cz (VERIFIED from the rendered page; identical to AASHTO 2011 Table 3-2 in feet)

Apply K_cz **only on the outside** of curves with R ≤ 900 m: DCZ_c = round_to_0.5(DCZ_t × K_cz) (RDM
3.1.2; we round halves up, ASSUMED). "—" means the table has no value there, because the curve is too sharp for that design speed.

| R (m) | 60 | 70 | 80 | 90 | 100 | ≥ 110 |
|---|---|---|---|---|---|---|
| 900 | 1.1 | 1.1 | 1.1 | 1.2 | 1.2 | 1.2 |
| 700 | 1.1 | 1.1 | 1.2 | 1.2 | 1.2 | 1.3 |
| 600 | 1.1 | 1.2 | 1.2 | 1.2 | 1.3 | 1.4 |
| 500 | 1.1 | 1.2 | 1.2 | 1.3 | 1.3 | 1.4 |
| 450 | 1.2 | 1.2 | 1.3 | 1.3 | 1.4 | 1.5 |
| 400 | 1.2 | 1.2 | 1.3 | 1.3 | 1.4 | — |
| 350 | 1.2 | 1.2 | 1.3 | 1.4 | 1.5 | — |
| 300 | 1.2 | 1.3 | 1.4 | 1.5 | 1.5 | — |
| 250 | 1.3 | 1.3 | 1.4 | 1.5 | — | — |
| 200 | 1.3 | 1.4 | 1.5 | — | — | — |
| 150 | 1.4 | 1.5 | — | — | — | — |
| 100 | 1.5 | — | — | — | — | — |

- **Use the nearest table radius at or below R, and the nearest speed column at or above V_d.** For
  "—" cells and R < 100 m (our hairpins), use **K = 1.5** (ASSUMED: the table's ceiling).
- **Taper (VERIFIED, RDM 3.1.2, Fig. 3-4).** With no spiral, transition over **20:1**, i.e. a length of
  20 × (DCZ_c − DCZ_t), placed ⅓ on the curve and ⅔ on the tangent. With a spiral, transition over the
  spiral. Our centrelines are smoothed splines, so use the 20:1 rule on the smoothed per-station K.

### 3.3 Worked DCZ for the test roads (classes and speeds from `out/ontario_v2/*/route.json`; AADT ASSUMED)

| Road | OSM | V_d | AADT row | DCZ tangent (flat) | DCZ outside of R = 100-150 m |
|---|---|---|---|---|---|
| Forks of the Credit | secondary, maxspeed 50 | 60 | ≥ 1,500 | 4.5 m | 4.5 × 1.4-1.5 → **6.5-7.0 m** |
| Sydenham Rd | secondary, maxspeed 50/60 | 60/70 | ≥ 1,500 | 4.5 / 5.5 m | 6.5-7.0 / **8.5 m** (K = 1.5) |
| Snake Rd | tertiary, no maxspeed (→ 60) | 70 | ≥ 750 | 5.0 m | **7.5 m** |

---

## 4. Hazards that warrant a barrier

MTO's list of areas of concern (RDM 3.1.3, VERIFIED) is: boulders; rock faces and cuts; steep backslopes;
transverse slopes; structures (abutments, piers); signs; non-breakaway utility poles and sign supports;
**foreslopes steeper than 3H:1V**; **water ≥ 1 m deep**; deep V ditches; **high fill embankments**;
non-traversable culvert ends; trees ≥ 100 mm diameter; non-yielding mailboxes. MTO's order of preference
is: remove, relocate, redesign, make breakaway, **shield**, delineate, reduce speed (RDM 3.1.4).

**Decision for the builder (ASSUMED).** Shield only the hazards that real roads overwhelmingly treat with
a rail, and that we can detect: **critical fill slopes and drop-offs, deep water, bridge ends**, plus
optional rough rock cuts. **Trees, poles, signs, mailboxes and ditches are not triggers.** By the letter
of the rules, MTO severity indices make trees ≥ 200 mm worse than a guide rail (App. A: a 200 mm tree has
SI 3.5 at 60 km/h vs 2.3 for SBGR, VERIFIED). That would rail every wooded roadside, which is not what
these roads look like. Real practice removes the tree or accepts it.

### 4.1 Critical fill slopes and drop-offs (the embankment warrant)

MTO itself gives no height-vs-slope chart. It says foreslopes steeper than 3H:1V "are considered
critical … and may justify shielding … dependent on height, length, and uniformity" and refers the
designer to its Roadside.xlsx benefit/cost tool (RDM 2.3.2, VERIFIED). We therefore combine three sources:

| Fill slope (H:V) | Max height **without** rail | Source |
|---|---|---|
| 1H:1V or steeper (incl. vertical drop, wall, river bank) | **0.6 m** | AASHTO 1977 via NCDOT Table 2, TxDOT 17.3.2, NJDOT 8.2.4: drop-off steeper than 1:1 and > 2 ft → barrier (VERIFIED) |
| 1.5H:1V | **0.9 m** | NJDOT Table 8-1 "Critical Slope Warrants" (AASHTO 2011 based): 3 ft (VERIFIED). Ohio chart ≈ 0.7 m. |
| 2H:1V | **1.8 m** | NJDOT: 6 ft (VERIFIED). Ohio chart ≈ 1.5 m. |
| 2.5H:1V | **2.7 m** | NJDOT: 9 ft (VERIFIED). Ohio chart ≈ 2.4 m. |
| 3H:1V | **6.0 m** | MTO App. A: SI(3H:1V, smooth) ≤ SI(SBGR) up to 4 m and > SI(SBGR) at 6 m at every speed (VERIFIED, table below). MTO Table 2-4: fills > 6 m high get 2H:1V **plus barrier** (VERIFIED). Ohio chart asymptote ≈ 6 m (20 ft). |
| flatter than 3H:1V | never, for the slope alone | All sources. A non-recoverable 3-4:1 slope only extends the DCZ (§3.1 Note 1). |

Between rows, interpolate h_max linearly in the slope ratio H/V (ASSUMED).

- **The MTO severity-index check (VERIFIED, 2026 proposal App. A, surface A "smooth and firm").** MTO
  decides where a slope's area of concern starts by finding "where the severity index for the critical
  2H:1V [slope] is higher than the severity index for the barrier system" (RDM 3.1.6 worked example).
  - The SBGR SI by speed, 50/60/70/80/90/100 km/h: **2.0 / 2.3 / 2.5 / 2.9 / 3.1 / 3.4**.
  - **3H:1V.** At h = 2-4 m the SI is 2.0 / 2.3 / 2.5 / 2.8 / 3.0 / 3.3, which is not higher. At
    h = 6 m it is 2.1 / 2.6 / 2.8 / 3.2 / 3.4 / 3.7, which is higher.
  - **2H:1V.** Already at h = 0.3 m the SI is 2.1 / 2.6 / 2.8 / 3.2. **Decision (ASSUMED):** the bare SI
    comparison ignores how often vehicles leave the road and what the rail costs. It would rail every
    0.3 m 2:1 bank. Use the AASHTO/NJDOT heights in the table instead, and keep the SI only to justify
    the 3H:1V / 6 m row.
- **Vertical drops and water (VERIFIED, App. A).** A 0.3 m vertical drop has SI 2.6-3.4 (50-80 km/h), and
  water 1 m deep at the road edge has SI 2.6-3.0. Both exceed SBGR. This is consistent with the 0.6 m and
  1.0 m rules (we keep 0.6 m: a 0.3 m lip is a curb).
- **Length (ASSUMED).** NCDOT's cost-effective warrant curves need a minimum fill "warrant length" that
  grows as traffic falls (NCDOT Figs. 4-6, VERIFIED qualitatively). Require the condition over **≥ 10 m**
  of station (after a 5-station median filter) before it becomes a hazard. Water, drops ≥ 2 m and bridge
  ends are exempt.
- **Where it must start.** The top of the critical slope (its hinge) must lie inside the DCZ for that
  station and side. Our roads have the slope starting at or just past the shoulder edge, so in practice
  this always holds.
- **B for the length of need (VERIFIED, RDM 3.1.6).** "B is equal to the horizontal distance from the edge
  of travelled lane to the toe of slope", capped at the DCZ.

### 4.2 Water

- **MTO (VERIFIED, RDM 2.3.10).** "Water bodies … with a normal-water-level depth of 1.0 m or more located
  within the desirable clear zone … should be shielded with a barrier system." AASHTO-derived DOT rules
  use 2 ft (0.6 m) instead (NCDOT Table 2, TxDOT, NJDOT, VERIFIED).
- **We cannot measure depth.** Lidar returns the water surface. Rule (ASSUMED):
  - `natural=water` (lake, pond, reservoir, river area), `waterway=riverbank`, `landuse=reservoir`,
    `waterway=river` (buffered by `width`/2, default 8 m) and `waterway=canal` (default 5 m) count as
    **deep**. `intermittent=yes` and `seasonal=yes` do not.
  - `waterway=stream/brook/ditch/drain` count as **not deep**. Their banks are still caught by §4.1
    through the lidar.
- **Distance.** The water edge is measured from the lane edge. It is a hazard if that distance ≤ DCZ.
  It is also a hazard if it is ≤ toe + recovery area when a non-recoverable 3-4:1 slope leads down to it.
  **B = DCZ** (water extends past it).

### 4.3 Bridge ends (and the approach embankments)

- **Rules.**
  - TxDOT 17.3.2 lists parapet walls, wingwalls and bridge rail ends as barrier candidates for
    approaching traffic, and for opposing traffic within its clear zone (VERIFIED). AASHTO's own
    obstacle table is recalled as saying shielding is "generally required" for these (ASSUMED, not
    read).
  - MTO requires a standard SBGR-to-rigid-barrier transition at structure approaches (OPSD 912.430, RDM
    2.4.4, VERIFIED).
  - On **undivided** roads, both ends of every installation get a crashworthy terminal (RDM 2.4.6,
    VERIFIED).
  - The leaving end of one direction is the approach end of the other. That is why rural Ontario
    bridges have rail at all four corners (ASSUMED as a general observation; ask the user for the test
    roads).
- **Rule (ASSUMED).**
  - For every railed deck (from `geometry/bridges.py`), add a hazard of zero length (Lh = 0) at each
    deck end, on both sides, with B = DCZ.
  - The run length comes from the length-of-need rule (§5.1). On undivided roads, the far-side corners
    get the short La′.
  - Every corner is then raised to the 25 m minimum installation (§5.3).
  - The approach fill usually triggers §4.1 as well, and the spans merge.
- **Exception (ASSUMED).** Urban bridges (V_d ≤ 60 with `sidewalk=*` on the bridge) whose railing
  continues past the deck: no approach rail.

### 4.4 Culverts

MTO wants culvert ends inside the DCZ made traversable (safety-slope ends, bars for openings > 750 mm),
moved out of the DCZ, or shielded (RDM 2.3.11, VERIFIED). We cannot see culvert ends. **Rule (ASSUMED):
no separate rule.** A culvert under a fill shows in the lidar as a local drop on both sides, and §4.1
catches it whenever the fill is high enough. Optionally log OSM `tunnel=culvert` crossings in the report,
for checking.

### 4.5 Rock cuts (optional, default OFF)

- **What the rules say.**
  - MTO App. A gives a *smooth, uniform* vertical rock face SI 1.0-1.9 (50-80 km/h, surface A), which is
    **below** SBGR. A *jagged* face (surface E) has SI 3.2-4.5, **above** SBGR (VERIFIED).
  - AASHTO 1977 / NCDOT Table 2 lists "rough rock cuts" as needing a barrier (VERIFIED).
  - NJDOT: cut slopes "should not ordinarily be shielded". Rail only if the obstruction is on a slope
    flatter than 0.7H:1V, or within 6 ft of the toe of a steeper slope, and inside the 3H:1V clear zone
    (VERIFIED).
- **Rule (ASSUMED).** Warrant only if all of these hold on the rising side:
  - a rising segment steeper than 1H:1V, at least 2 m tall;
  - its toe inside the DCZ for the cut columns;
  - *and* an OSM `natural=cliff|bare_rock|scree` within 15 m.
  
  We cannot tell rough from smooth with a 1 m DTM. B = toe distance. Keep it behind a flag until the user
  confirms the escarpment cuts on Snake and Sydenham Rd have rails.

### 4.6 Not triggers (ASSUMED, see the decision above)

Trees, utility poles, signs, mailboxes, buildings, fences, and V ditches.

---

## 5. Turning hazards into rails

### 5.1 Length of need (VERIFIED, RDM 2.4.3, Figs. 2-14/2-15, Table 2-16)

Runout length E in metres, from **Table 2-16** (modified from AASHTO 2011):

| V_d (km/h) | AADT > 10,000 | 5,000-10,000 | 1,000-5,000 | < 1,000 |
|---|---|---|---|---|
| 130 | 143 | 131 | 116 | 101 |
| 120 | 127 | 116 | 102 | 89 |
| 110 | 110 | 101 | 88 | 76 |
| 100 | 91 | 76 | 64 | 61 |
| 90 | 81 | 67 | 57 | 54 |
| 80 | 70 | 58 | 49 | 46 |
| 70 | 60 | 49 | 42 | 38 |
| 60 | 49 | 40 | 34 | 30 |
| ≤ 50 | 34 | 27 | 24 | 21 |

- **Approach length for the near traffic: La = E·(1 − A/B).**
  - A is the lateral distance from the edge of the travelled way to the **face of the barrier**.
  - B is the distance from the edge of the travelled way to the **back of the hazard**, never more than
    the DCZ.
  - Undivided roads add an approach for **opposing** traffic: **La′ = E·(1 − A′/B′)**. Here A′ and B′ are
    measured **from the centreline**, and B′ is also capped at the DCZ.
  - Total length of need: La + Lh + La′ on undivided roads, La + Lh on divided roads and one-way ramps.
    One-way roads get a leaving-end anchor and no extension.
- **Direction bookkeeping (ASSUMED, our station convention).**
  - The *right* side (as driven, + direction) is hit by + traffic, which arrives at the hazard's low
    station s0. So La extends **below s0** and La′ extends **above s1**.
  - The *left* side is the mirror: La extends above s1 (near traffic is the − direction) and La′ below s0.
  - On curves, measure the extension along the station. AASHTO measures the runout along the tangent
    path, so this slightly overstates it, which errs toward a longer rail.
- **Example (ASSUMED inputs).** Forks of the Credit at V_d = 60, AADT 1-5k, so E = 34 m. Lane 3.25 m,
  rail face A = 0.5 m, B = DCZ = 4.5 m:
  - La = 34 × (1 − 0.5/4.5) = **30 m** upstream.
  - A′ = 3.75 m, B′ = 4.5 m, so La′ = 34 × (1 − 3.75/4.5) = **6 m** downstream.
- **Terminals** are counted inside these lengths. MTO lets part of a terminal count toward the length of
  need (RDM 3.1.6, VERIFIED), and our visuals do not need the exact split.

### 5.2 Joining and gaps (VERIFIED: MTO RDM 3.1.6; others for comparison)

- **MTO.** "Consider combining … when the LON for adjacent obstacles overlap; or gap between barrier
  systems is less than or equal to **50 m**." The exception is "when a gap is required … to accommodate an
  entrance to a property".
- **Other states:** MaineDOT joins gaps < 200 ft (61 m) unless the gap is needed for access. NJDOT avoids
  gaps ≤ 200 ft. NCDOT extends through gaps < 300 ft (91 m).
- **Rule.** Same side: join spans whose gap is ≤ 50 m, unless a connection (§5.4) lies in the gap.

### 5.3 Minimum lengths (VERIFIED, RDM 4.2.1, 4.2.2.1)

- **Type M SBGR.** "The minimum installation length is **25 m** including terminals and leaving end
  treatments, and there is no maximum installation length."
- **High-tension cable.** "Recommended minimum length … is **100 m** including terminals" (maximum 6,000 m).
- **Rule (ASSUMED).**
  - Grow a run shorter than 25 m symmetrically to 25 m, unless that would cross a connection or a
    railed deck.
  - A piece that a connection has cut shorter than **12 m** is dropped. NJDOT's driveway details use
    short-radius/CRT treatments here, which we do not model.

### 5.4 Side roads and driveways (VERIFIED principle; numbers ASSUMED)

- **Principle.** Rails never cross an access. Gaps for entrances are the stated exception to joining
  (MTO 3.1.6). Rail must not restrict sight distance at intersections and driveways (NJDOT §8: "Guide
  rail should not restrict sight distance", VERIFIED). The ends at a gap get an end terminal (MTO
  3.1.6, VERIFIED; NJDOT 8.3.4 C shows driveway treatments).
- **Rule (ASSUMED).** For every OSM way that joins the route, on the side it joins, cut
  [s_c − w/2 − c, s_c + w/2 + c]. The way's s_c and its angle come from `route.json` `stubs` plus OSM
  `highway=service`.
  - **w** is the joining way's width: the `width` tag, else driveway 4 m, `service` 5 m,
    `residential/unclassified/track` 7 m, `tertiary` and up 8 m. Divide by sin(angle) for skewed joins.
  - **c** is the clearance: **5 m** for public roads (sight line and the terminal's flare) and **2 m**
    for driveways.
- **Unmapped driveways** cannot be seen. The user can cut them out with `guardrails.json` (see §7.2).

### 5.5 Lateral position and hardware (VERIFIED: RDM 3.1.6, 4.2.1, 4.2.2.1; 2026 proposal record of revisions)

**Where the rail stands:**
- **Type M20 SBGR stands "at the edge of shoulder with 1.0 m rounding".** The rounding breakpoint to a
  3H:1V or flatter slope is at least 0.5 m beyond the edge of shoulder. Next to a 2H:1V slope, the post
  face can sit at the breakpoint if 2.438 m posts are used. Type M30 may stand further out if the
  ground there is 10H:1V or flatter, with a flare up to 10:1.
- **Clearance limits.**
  - At least **4.25 m from the roadway centreline** (snowplough clearance).
  - The 2026 proposal adds "not … more than **4.0 m** from edge of travelled way".
  - Shy-line offsets (Table 3-1: 50 → 1.1 m, 60 → 1.4, 70 → 1.7, 80 → 2.0, 90 → 2.2, 100 → 2.4) are
    desirable, "however barriers may be placed at offsets equal to the recommended shoulder width".
  - The task brief's "0.6-1.2 m beyond the shoulder" is **not** Ontario practice. The face goes *at*
    the shoulder edge. Maine's restricted-site rule is ≥ 4 ft (1.2 m) from the lane edge or ≥ 16 ft
    from the centreline. NJDOT wants ≥ 2 ft from the post back to the slope break.

**What the rail is:**
- **Type M SBGR (MGS) geometry.**
  - Top of rail **785 ± 25 mm** (at least 710 mm after overlays).
  - W-beam 311 mm deep and 2.67 mm thick. Panels are 3.81 m nominal (4.128 m with the lap), spliced
    *between* posts.
  - Posts W150×13, 1.829 m long, every **1.905 m** (0.952 or 0.476 m where deflection must be cut).
  - Routed blocks **200 mm** (M20) or 300 mm (M30), wood or plastic.
  - Reflectors on the posts, every ≤ 20 m on tangents.
  - The legacy standard SBGR had its top at **685 mm**, with splices at the posts.
- **Working width** of Type M20 at TL-3 is 1.4 m, so the hazard edge should be ≥ 1.4 m behind the face.
  Not needed for visuals.
- **Curves.** SBGR works on centreline radii **≥ 45 m**; sharper curves need shop-bent rail (fine for us).
  High-tension cable needs R ≥ 250 m.
- **Terminals.**
  - Approach ends: on divided roads; both ends on undivided roads.
  - SBEATs flare away tangentially "over their nominal length of approximately **15 m** to provide an
    offset of **0.3 m** from the edge of shoulder to the face of rail at post 1".
  - Legacy turned-down ("fishtail") ends exist on old roads and are being replaced.
- **Cable.**
  - Ontario used **low-tension 3-cable rail on wooden posts "since the 1960s"** (12 mm strand).
  - New installs since 2016 are high-tension 3-cable: 19 mm ropes on steel posts every 2 m, working
    width 2.2 m.
  - It is used where SBGR cannot get 4.25 m from the centreline, and on long open runs (less snow
    drifting).
  - **Rule (ASSUMED).** W-beam everywhere by default. Cable is an optional variant only for
    `motorway|trunk|primary` with V_d ≥ 90, a run ≥ 100 m, R_min ≥ 250 m and ≥ 2.2 m of flat ground
    behind the face.

---

## 6. Validation plan (before turning it on by default)

1. Run the warrant pass on the three test roads. Write `out/<track>/data/guardrails_auto.json` (spans +
   reason + DCZ + h + slope per span) and a top-down PNG overlay.
2. Ask the user to mark where real rails are, in `guardrails.json` or via the map picker later.
   Compute precision and recall by length.
3. Tune only these knobs, in this order:
   - the `h_scale` multiplier on the §4.1 heights (1.0 = AASHTO/NJDOT);
   - `min_hazard_m` (10 m);
   - the V_d margin (+10);
   - the AADT class table.
4. Expect over-prediction on old escarpment roads (§1 caveat). If recall is good and precision poor,
   raise `h_scale` before touching anything else.

---

## 7. Rules for the builder

### 7.1 Algorithm

```
inputs: RoadModel m (s, centre, half_width, shoulder, curvature, bridge), per-station OSM tags
        (highway, maxspeed, oneway, lanes, sidewalk), DTM sampler z(xz) (HRDEM 1 m; fall back to
        build._ground_fn), OSM water/waterway/natural layers, joining ways (stubs), railed decks.
output: BarrierSpan(side, s0, s1, kind="guard_rail", source="warrant") + reason per span,
        passed to plan_barriers(..., heuristic=spans) so user > OSM > bridge-approach > warrant.

0. Gate. Skip stations on bridge decks or in junction areas. Skip roads with V_posted <= 40.
   Urban curbed streets (sidewalk=* and V_posted <= 50) keep only: deep water, drops >= 2 m,
   bridge ends.
1. Speeds. V_posted = maxspeed (mph x 1.609), else CLASS_DEFAULT_SPEED[highway].
   V_d = ceil10(V_posted + V_MARGIN); clamp to a row of the DCZ table
   (<=60 | 70-80 | 90 | 100 | >=110) and of the runout table.
2. Traffic. aadt = CLASS_AADT[highway] (per-track override allowed).
3. Edges. Lane edge e(s) = m.half_width (asphalt edge; we have no paved-shoulder data);
   shoulder edge = e(s) + m.shoulder (= RoadModel.outer).
   Undivided two-way: the centreline is the far-lane edge. oneway=yes: one direction only.
4. Profiles. For each station (1 m) and each side: sample z at x = 0 .. X_MAX step 0.5 m beyond the
   lane edge, X_MAX = DCZ_max + 10 m. Use road_columns() so columns never cross on hairpins.
   z_road = z at x = 0 (the built road edge). Smooth along x with a 1.5 m moving mean.
   Smooth along s with a 5-station median.
5. DCZ. Classify the ground from x = 0 to DCZ(10H:1V) (ignoring critical runs) -> table column
   (sec. 3.1). Look up DCZ_t. Outside of curve (side opposite sign(kappa)) with R = 1/|kappa| <= 900 m:
   DCZ = round_0.5(DCZ_t * K_cz(R, V_d)), K = 1.5 beyond the table. Taper K along s at 20:1
   (1/3 on curve, 2/3 on tangent).
   If a mean fill steeper than 4H:1V starts inside DCZ:
   DCZ_eff = toe + max(DCZ_flat - (shoulder + 0.5 * rounding), 3 m); else DCZ_eff = DCZ.
6. Hazards at (s, side). Each has back distance B (capped at DCZ) and a reason:
   a. fall: maximal runs where dz/dx < -1/3 (gaps <= 1 m bridged). For each run:
      top x_t, toe x_b, h = z(x_t) - z(x_b), r = (x_b - x_t) / h (H:V).
      Warrant if x_t <= DCZ_eff and h > H_SCALE * h_max(r) (table in 7.2).
      B = min(x_b, DCZ).
   b. water: deep-water edge distance d_w <= DCZ_eff -> B = DCZ.
   c. rock (flag ROCK_CUTS): rising run dz/dx > 1 (steeper than 1H:1V), height >= 2 m,
      toe <= DCZ(cut column), OSM natural=cliff|bare_rock|scree within 15 m -> B = toe.
7. Hazard runs. Per side, group warranted stations; bridge gaps < 3 m; drop runs < MIN_HAZARD_M
   (10 m) unless the reason is water or a drop >= 2 m. Per run keep Lh = [s0, s1],
   B = min over the run, reasons.
8. Bridge ends. For each railed deck end at station s_e, both sides: a run with Lh = [s_e, s_e]
   on the approach side of the deck, B = DCZ, reason "bridge". Skip the urban exception (sec. 4.3).
9. Length of need per run (sec. 5.1).
   A  = face offset - lane edge.     A' = face offset from the centreline (= lane width + A).
   B' = min(lane width + B, DCZ).    E  = RUNOUT[V_d][aadt].
   La = max(0, E * (1 - A/B)) toward the near traffic's upstream end.
   La' = max(0, E * (1 - A'/B')) toward the far traffic's upstream end
   (undivided two-way only; 0 if oneway).
10. Join. Per side, merge spans with gap <= 50 m (JOIN_M) unless a connection lies in the gap.
11. Cut. For each joining way on that side, remove [s_c - w/2 - c, s_c + w/2 + c]
    (sec. 5.4 widths and clearances; divide w by sin(angle)). Also remove railed decks
    (plan_barriers already does).
12. Minimum length. Pieces shorter than MIN_RUN_M (25 m): extend symmetrically to 25 m if not
    blocked. If blocked and < MIN_CUT_PIECE_M (12 m), drop.
13. Placement and type. Face at the shoulder edge, else lane edge + FACE_GAP_M (0.5 m). Push out
    to >= 4.25 m from the centreline when the ground allows (hinge >= face + 0.5 m). Never more
    than 4.0 m from the lane edge. Kind = W-beam Type M (sec. 5.5). Optional cable variant per 5.5.
    Ends: flare 0.3 m over the last 15 m, or a turned-down end, for both ends on undivided roads.
14. Report. Write guardrails_auto.json: side, from_m and to_m in route distance (same format as
    guardrails.json), reason, V_d, aadt, DCZ, h, r, B, La, La'.
```

### 7.2 Parameter table

| Name | Value | Source / status |
|---|---|---|
| `V_MARGIN` | +10 km/h; **+20 in the code after the §8 validation** | ASSUMED (MTO: +20 historically on provincial highways, +0 on Secondary highways; VERIFIED memo) |
| `CLASS_DEFAULT_SPEED` (no `maxspeed`) | motorway 100, trunk 90, primary 80, secondary 80, tertiary 60, unclassified 60, residential 50, service 30 km/h | ASSUMED (HTA default 50 in local municipalities / built-up areas, VERIFIED) |
| `CLASS_AADT` | motorway 40,000; trunk 12,000; primary 8,000; secondary 3,000; tertiary 1,200; unclassified 400; residential 300; service 100 | ASSUMED (no open AADT for these roads checked) |
| DCZ table | §3.1 (MTO Table 2-2) | VERIFIED |
| `K_cz` | §3.2 (MTO Table 2-3), R ≤ 900 m, outside only, 1.5 beyond the table | VERIFIED; 1.5 cap ASSUMED |
| DCZ taper | 20:1, ⅓ on curve and ⅔ on tangent | VERIFIED (RDM 3.1.2) |
| Recovery beyond the toe | max(DCZ(10H:1V) − shoulder − ½ rounding, 3 m) | VERIFIED (Table 2-2 Note 1) |
| Rounding width | 1.0 m (V_d < 100), 1.5 m (≥ 100) | VERIFIED (Table 2-8, desirable) |
| Critical slope | steeper than 3H:1V (dz/dx < −1/3) | VERIFIED (MTO 2.3.2, AASHTO, NJDOT) |
| `h_max(r)` | r ≤ 1: 0.6 m; 1.5: 0.9; 2: 1.8; 2.5: 2.7; 3: 6.0; > 3: ∞; linear in r between | VERIFIED rows (NJDOT/AASHTO 2011, TxDOT, MTO SI); interpolation ASSUMED |
| `H_SCALE` | 1.0 (first calibration knob) | ASSUMED |
| Deep water | ≥ 1.0 m NWL; proxy: lakes, ponds, reservoirs, river areas, `river`, `canal`; not streams, ditches or intermittent water | Rule VERIFIED (MTO 2.3.10); proxy ASSUMED |
| River / canal half-width without `width` | river 8 m, canal 5 m (else `width`/2) | ASSUMED |
| `ROCK_CUTS` | off; slope > 1V:1H, ≥ 2 m, toe in DCZ, OSM cliff / bare_rock / scree ≤ 15 m | ASSUMED (rough faces: VERIFIED as warranting) |
| `MIN_HAZARD_M` | 10 m (water, drops ≥ 2 m, bridges exempt) | ASSUMED |
| Profile | 0.5 m step to DCZ_max + 10 m; 1.5 m mean along x; 5-station median along s | ASSUMED |
| Runout E | §5.1 (MTO Table 2-16) | VERIFIED |
| La, La′ | E·(1 − A/B), E·(1 − A′/B′) from the centreline; B, B′ ≤ DCZ | VERIFIED (RDM Figs. 2-14/2-15) |
| `JOIN_M` | 50 m (not across a connection) | VERIFIED (RDM 3.1.6) |
| `MIN_RUN_M` | 25 m including terminals (W-beam); 100 m (cable) | VERIFIED (RDM 4.2.2.1, 4.2.1) |
| `MIN_CUT_PIECE_M` | 12 m | ASSUMED |
| Connection clearance `c` | public road 5 m, driveway 2 m beyond the joining way's edge | ASSUMED (principle VERIFIED) |
| Joining-way width `w` | `width` tag, else driveway 4, service 5, residential/unclassified/track 7, tertiary+ 8 m | ASSUMED |
| `FACE_GAP_M` | 0 beyond the shoulder edge; 0.5 m beyond the lane edge if no shoulder | Shoulder edge VERIFIED (Type M20); 0.5 m ASSUMED |
| Centreline clearance | ≥ 4.25 m (when the ground allows) | VERIFIED (RDM 4.2.2.1) |
| Max offset | ≤ 4.0 m from the lane edge | VERIFIED (2026 proposal) |
| Rail geometry | top 0.785 m, W 311 mm × 2.67 mm, posts every 1.905 m (W150, 1.829 m), blocks 0.20 m | VERIFIED (RDM 4.2.2.1). Today's code uses 0.75 m and a 2.0 m spacing: update. |
| Terminal | flare to 0.3 m over 15 m (SBEAT) at both ends on undivided roads | VERIFIED |
| Min radius | W-beam 45 m (shop-bent below; ignore); cable 250 m | VERIFIED |
| Bridge ends | all 4 corners on undivided roads, length from La/La′ then ≥ 25 m; skip urban curbed bridges | Rules VERIFIED; "all 4 corners" and the urban skip ASSUMED |
| Not triggers | trees, poles, signs, mailboxes, buildings, ditches | ASSUMED (deliberate departure from the letter of RDM 3.1.3) |

### 7.3 Hook-up (proposal; app files are owned by another team)

- **New module.** `src/driftmaps/geometry/warrants.py` with `warrant_spans(model, ground_fn | dtm,
  osm_layers, stubs, decks, options) -> list[BarrierSpan]` (source `"warrant"`).
  - `build.py` passes its result as `heuristic=` to `plan_barriers`. The priority stays user > OSM >
    bridge approaches > warrants, and decks are never railed twice.
  - It replaces the ad-hoc `drop_off_spans`, which is §4.1 with a single 3 m/4 m probe.
- **Options.** A build option `guardrails = "real" | "warranted"`, default `"warranted"` after the
  calibration in §6.
- **Per-track control in `guardrails.json`.** `"auto": false` turns warrants off for a track.
  `{"side", "from_m", "to_m", "kind": "none"}` entries erase warrant spans: unmapped driveways, or
  places the user knows have no rail.
- **Overpass.** The water and natural layers need one extra Overpass `around:` query, the same pattern as
  `geo/barriers.py`. Do **not** edit `geo/overpass.py`; it is being changed in parallel.

### Open questions

- **Real rails on the three roads.** Where are they? This is needed for §6 and decides the bridge "all 4
  corners" and rock-cut rules.
- **Traffic counts.** Hamilton and Peel publish traffic counts. Not checked: Hamilton's portal blocked
  scripts in `scenery.md` §5c. Real AADT would replace `CLASS_AADT` for these roads.
- **TAC Ch. 7 text.** It may contain its own embankment warrant figure and low-volume-road relaxations.
  It is paywalled ($82). Buy it only if calibration shows a systematic bias.
- **NCHRP 1097 relative-risk DCZ** for ≥ 90 km/h roads (MTO 2026 proposal). It matters only when we
  build fast highways.

---

## 8. Implementation and validation (2026-10-16)

### 8.1 What was built

- **`src/driftmaps/geometry/warrants.py`** implements §7.1. `warrant_inputs(...)` collects the
  inputs for a built road. `warrant_spans(inputs, WarrantOptions())` returns `BarrierSpan`s
  (kind `guard_rail`, source `"warrant"`), the hazard runs behind them, and a report saying why
  each span exists, e.g. *"fill 1.9H:1V, 10.3 m drop 2.0 m from the lane edge (limit 1.5 m,
  clear zone 6.5 m)"*. It is deterministic. Snake Rd takes about 1.5 s.
- **`build.py`**: `BuildOptions.auto_guardrails = True` (CLI `--auto-guardrails/--no-auto-guardrails`).
  The spans go to `plan_barriers(..., heuristic=...)`, so guardrails.json, then OSM rails, then
  bridge approaches win over them, and railed decks are never railed twice. The DTM is the build's
  own corrected DEM chain (`dtm_height_fn`: HRDEM 1 m where it exists). The track gets
  `data/guardrails_auto.json` (every warranted span with its reasons, V_d, AADT, DCZ, h, slope,
  B, La, La', plus what was actually built). The debug PNG draws automatic rails in yellow.
  `guardrails.json` accepts `"auto": false` and `{"kind": "none"}` erase spans (§7.3).
- **`geo/barriers.py`**: `fetch_roadside` makes one more cached Overpass `around:` query for deep
  water (areas, multipolygons clipped to the route box, rivers and canals), rock and every
  road, track or driveway near the route. The existing API is unchanged.
  `parse_user_file`/`load_user_file` are new; `parse_user_spans` still returns only real spans.
- **`geometry/barriers.py`**: Type M hardware is now the default: top of rail 0.785 m, posts
  every 1.905 m, 200 mm blocks. Free ends flare 0.3 m away from the road over 15 m (over a third
  of the rail if it is shorter), then turn down as before. The single-probe `drop_off_spans` is
  kept for compatibility, but nothing calls it.
- **`tools/validate_guardrails.py`**: the validation below.

**Departures from §7.1** (all ASSUMED):
1. Stations every **2 m** (not 1 m), with a 3-station (6 m) median along the road.
2. Profiles start **1.5 m inside** the lane edge, because a misplaced OSM centreline can hide a
   slope's top there. They reach at least **30 m** out, the toe of a 6 m 3H:1V fill. With the
   first cut (widest DCZ + 10 m), 3:1 fills were truncated and never reached their 6 m row.
3. **3H:1V itself counts as critical** (slope ≤ 1/3 + 0.002). Without that, the 6 m row never
   fires on an exact 3:1 slope.
4. **V ditches are not hazards.** A fall under 2 m deep that climbs back at least 70 % of its
   depth within 6 m of its toe is skipped. This applies §4.6 "ditches are not triggers".
   Without it, every steep 1.5 m lidar ditch warrants a rail.
5. **Non-recoverable runs** (3H:1V-4H:1V) that extend the DCZ to the toe plus the recovery area
   exclude critical segments. If a critical fall counted as "non-recoverable", it pushed the DCZ
   to 20+ m.
6. **DCZ taper** is a 20:1 dilation of the curve widening. All of the taper falls on the tangent
   side, which is up to ⅓ of the taper too wide.
7. The exempt (water, drop ≥ 2 m) hazards still need **2 stations** (about 4 m), so one noisy
   station cannot make a 25 m rail.
8. Rails are placed only on **A..B**, not on the invented run-up and run-off.
9. The rail face is at the **shoulder edge** (`RoadModel.outer`). The 4.25 m-from-centreline
   push-out is not modelled: on our roads it is at most 0.25 m, and the OSM snap logic in
   `span_meshes` would undo it.

### 8.2 Validation against rails mapped in OSM

**Finding test roads.** An Overpass census of `barrier=guard_rail` inside Ontario (south of
46.6° N) found only **568 ways**, mostly on Toronto expressways. Rural or escarpment roads with
rails mapped beside them, all on the HRDEM 1 m mosaic:

| Area | Road | Length A..B | Mapped rail | Character |
|---|---|---|---|---|
| hwy15_seeleys_bay | Highway 15 (primary, 80 km/h) | 5.99 km | 2,662 m | Rideau lakes, fills, rock cuts |
| hwy15_elgin | Highway 15 | 6.00 km | 2,557 m | lakes, wetlands, one bridge |
| governors_road | Governors Rd (secondary, 60) | 1.34 km | 498 m | Dundas valley side |
| black_bridge_road | Black Bridge Rd (tertiary, 50) | 0.75 km | 452 m | 4 m fill, Cambridge |
| mountainview_grimsby | Mountainview Rd N (tertiary) | 0.64 km | 257 m | overpass approach fills |
| mountain_brow | Mountain Brow Blvd (secondary) | 1.76 km | 807 m | 20 m escarpment brow, 2 bridges |

Total: 16.5 km of road, 7.2 km of mapped rail. Two candidates were dropped because the tool's
pin finder found only a 0.25 km / 0.02 km piece of the named road: Appleby Line and Ontario
St, Beamsville.

**Method.** For each area:
- Route along the named road with the project router (`route_from_pins`, read-only), from
  500 m before the first rail to 500 m after the last, capped at 6 km.
- Shape it with `prepare_geometry` and run the warrants.
- Match the OSM rails with `match_barriers`.
- Compare length per side inside A..B, off railed decks. Precision is the predicted length that
  is also mapped; recall is the mapped length that is also predicted. Both are computed strictly
  and with a 10 m tolerance at span ends.

OSM is incomplete (a missing rail is not necessarily a false positive), so **precision is a
lower bound**.

**Before → after** (10 m tolerance; strict in brackets):

| Area | Default V+10: P / R | Chosen V+20: P / R |
|---|---|---|
| hwy15_seeleys_bay | 0.77 / 0.67 | 0.75 / 0.70 |
| hwy15_elgin | 0.79 / 0.78 | 0.75 / 0.81 |
| governors_road | 0.28 / 0.50 | 0.27 / 0.51 |
| black_bridge_road | 1.00 / 0.14 | 1.00 / 0.64 |
| mountainview_grimsby | 0.56 / 1.00 | 0.55 / 1.00 |
| mountain_brow | 0.78 / 0.96 | 0.76 / 0.98 |
| **Total** | **0.71 / 0.71** (0.68 / 0.68), 7.2 km predicted | **0.70 / 0.76** (0.67 / 0.74), 8.0 km predicted |

**Sweep** (total, 10 m tolerance, P / R):

| V margin | h_scale 0.75 | 1.0 | 1.25 | 1.5 | 2.0 |
|---|---|---|---|---|---|
| +0 | 0.70 / 0.68 | 0.74 / 0.65 | 0.80 / 0.60 | 0.83 / 0.55 | 0.84 / 0.48 |
| +10 | 0.67 / 0.74 | 0.71 / 0.71 | 0.76 / 0.66 | 0.79 / 0.61 | 0.81 / 0.55 |
| +20 | 0.66 / 0.79 | 0.70 / 0.76 | 0.74 / 0.73 | 0.77 / 0.67 | 0.80 / 0.60 |

`min_hazard_m` 10 / 20 / 30 m changed the totals by at most 1 %.

**Decision.** The default is now **V margin +20**, with `h_scale` kept at 1.0.
- +20 is MTO's documented historical margin (memo SCB-HDO-2026-01, VERIFIED), so it is a rule
  value, not a fitted one.
- It raises recall on 5 of 6 areas and costs 0.01-0.04 of precision on each.
- Black Bridge Rd shows the mechanism. Its 4 m fills have their hinge 3.7-4.7 m from the lane
  edge. At V_d 60 (posted 50 + 10) the DCZ is 3.5 m and misses them; at V_d 70 it is 5.0 m and
  catches them.
- V+20 with h_scale 1.25 scored F1 0.74 against 0.73. That 0.01 is not worth allowing fills
  higher than every source's table, so `h_scale` stays at 1.0.
- Before the change, precision and recall were balanced. §6 says to raise `h_scale` only when
  precision is poor, which it was not.

**What the pictures show** (`out/guardrails_validate/<area>_default.png` = V+10,
`_chosen.png` = V+20, plus `_detail*.png` crops of the worst disagreements):
- **Highway 15.** Predictions follow the mapped rails along the lake shore and fill sections,
  and none fall on the flat farm stretches.
- **Misses on Highway 15** are mostly on the cut side: a ditch and then a 2-3 m rise that is
  probably rock. That is the §4.5 rock-cut rule, still off. Highway 15's OSM `width` tags
  (15.5-22 m) evidently include paved shoulders, which puts our "lane edge" several metres out.
  The build shares this data issue.
- **Governors Rd**'s "false positives" sit on 3-8 m slopes at 2.1-2.4H:1V right at the road
  edge. By the rules they warrant a rail. Either OSM is incomplete there, or this is the old-road
  caveat of §1.
- **Mountainview Rd** predicts the overpass approach fills on both ends; only one end is
  mapped.
- **Mountain Brow** matches the escarpment brow almost exactly. The extra predictions are bridge
  ends.

### 8.3 Snake Rd with automatic rails

`drift-maps build --route out/ontario/dm_snake_road/route.json --name "Snake Road (Waterdown)"
--id dm_snake_road --out out/guardrails_test --force` builds **8 spans, 1,513 m** of rail on
7.1 km of roadside, in 56 s with trees and buildings. The spans are listed by route metres from
A and checked on the hillshade views `out/guardrails_test/snake_warrants_*.png`:

| Side | From A | Why |
|---|---|---|
| left | 207-322 | 3.7 m fill at 2.0H:1V |
| left | 526-926 | escarpment side, 8-13 m drops at 1.6-2.3H:1V |
| left | 2,420-2,952 | escarpment, 2-8 m at 1.6-2.0H:1V |
| left | 3,471-3,496 | creek bridge approach (25 m minimum) |
| right | 426-587 | ravine, 9-10 m drops |
| right | 1,722-1,899 | escarpment, 10-17 m at 1.5-1.7H:1V |
| right | 3,319-3,383 | ravine beside the road, 9.7 m at 1.9H:1V |
| right | 3,457-3,496 | creek bridge approach (La = 39 m) |

Checked on the images:
- The flat 1.0-1.7 km stretch, and its roadside ditches, get no rail.
- Every rail stops at the side roads.
- The far-side corners of the creek bridge are dropped by the rules: route end B and a junction
  sit about 20 m past the deck, so each corner piece would be under 12 m.

### 8.4 Still uncertain

- **The sample is small.** Six roads and 7.2 km of mapped rail, and Highway 15 is 70 % of it.
  OSM rails are rare in Ontario, so a few hundred metres of the user's own knowledge of the test
  roads (§6 step 2) would be worth more than more OSM areas.
- **Rock cuts** (§4.5) are the main source of misses on Shield roads. The rule exists
  (`WarrantOptions(rock_cuts=True)`) but is not validated.
- **AADT per class and the 50 km/h urban gate** are still ASSUMED. `CLASS_AADT` was not tuned;
  it was the last knob in §6.
- **The DTM sees today's ground**, including the road's own fill. Where OSM is offset by more
  than about 2 m, the profile can start partway down a slope and underestimate the drop. The
  1.5 m inside start covers small offsets only.
- **Unmapped driveways** cannot be cut. The user has to erase them with `"kind": "none"`.
