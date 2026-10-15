# M2 in-game checklist: 3 Ontario roads, now with scenery

Thanks for driving the first versions! These are the same three roads, rebuilt with what you
asked for:

- **Trees where they really stand.** Every tree comes from government lidar scans (laser
  scans from a plane): we find the real tree tops and give each one its real height. Leafy or
  evergreen comes from a national land-cover map.
- **Buildings** where the real houses, barns and sheds are, from OpenStreetMap and Microsoft's
  building outlines, at their real (lidar) height.
- **Real bridges** instead of floating road sheets: a concrete deck, concrete side walls you
  can hit, solid ends (abutments) and support columns (piers) on the long one. The ground under
  a bridge is now the real creek valley.
- **Guardrails only where they really are.** No map we can use shows guardrails on these roads
  (OpenStreetMap has none), so you get rails only next to each bridge for now. **Tell us where
  the real ones are** (section 5) and the next build will have them.

You need: Assetto Corsa on Windows, Content Manager (CM), Custom Shaders Patch (CSP), and
about 45 minutes. At the end there is a reply template you can copy and fill in.

| Track (name in CM) | Length | What's new to look at |
|---|---|---|
| Snake Road (Waterdown) | 3.5 km | Forest on the escarpment climb (about 23,000 trees), houses in Waterdown at the top (144 buildings), the 20 m bridge just before the finish. |
| Sydenham Road (Dundas) | 3.5 km | The 112 m bridge right after the start, about 10 m above the creek, on 4 piers. Forest on the climb, then farmland with houses and lines of trees between the fields (about 12,000 trees, 119 buildings). |
| Forks of the Credit Road (Caledon) | 4.3 km | Thick forest in the river valley (about 28,000 trees), the river bridge about 1 km in, two small culvert bridges around 2.6 and 2.8 km, scattered houses (62). |

**What it looks like on purpose (not bugs):**
- Trees are flat pictures standing crossed in an X (the classic way, same as Kunos' own tracks).
  From some angles you may see that they are flat.
- Trees stop about 400 m from the road, and beyond 150 m there are fewer, bigger ones.
- Right beside the road (about 5 m past the gravel) there are no tree trunks. That's where the
  invisible walls are, and it keeps you from ever driving through a tree. Branches still hang
  over the road.
- Buildings are simple shapes (walls and a flat or pitched roof) with painted-on windows and
  doors. No signs, mailboxes, fences, power lines or road signs yet.
- Guardrails are only on the 20 m before and after each bridge (see section 5).
- The walls are still invisible.

> For the developers: built with `uv run drift-maps build --route out/ontario/<id>/route.json
> --name "<name>" --id <id> --out out/ontario_v2 --zip out/send_v2/drift-maps-<name>.zip`
> (default scenery: trees, buildings, guardrails on, `--scenery-density 1`). Ids unchanged,
> so the install replaces the M1 folders. Debug images (trees, buildings, rails, bridges,
> distance ticks every 250 m from A): `out/ontario_v2/debug/`. Per-track stats are in the
> build output (`trees`, `houses`, `rails`, `bridge` lines).

---

## 1. Install

1. Put the three zips (`drift-maps-snake-road.zip`, `drift-maps-sydenham-road.zip`,
   `drift-maps-forks-of-the-credit.zip`) and `install_tracks.bat` **in the same folder** (for
   example Downloads).
   - The .bat installs **every** `drift-maps*.zip` in that folder. Move the old (M1) zips out
     first.
2. Double-click `install_tracks.bat`. It finds AC through Steam. If it asks for your AC
   folder: in CM go to **Settings → Assetto Corsa**, copy the folder path at the top, and
   paste it in.
3. When it says it's done, press a key to close it.
4. Open CM → **Content → Tracks**. The three tracks keep the same names and **replace** the
   old versions. If CM was already open, restart it.

---

## 2. Trees

Drive each track in **Practice** (any normal road car; the GT86 is great). Look around while
you drive, and stop a few times to look properly.

- [ ] **Trees are where the real forest is.** Wooded stretches look wooded, fields and lawns
      are open. Where you know the real road well: does it feel like the same place?
- [ ] **Trees stand on the ground.** No trees floating in the air or sunk deep into a hill.
- [ ] **No tree on the road or in your way.** No trunk on the asphalt or the gravel, and no
      tree inside a house.
- [ ] **They look like trees, not boxes.** No solid black, white, pink or blocky squares
      around the leaves. (Pink/black/white means a texture failed to load: tell us which.)
- [ ] **Colour and brightness.** Too dark, too bright, or about right next to the grass?
- [ ] **Distant trees.** As you drive, do trees pop in or vanish in a distracting way? Do
      far-away hillsides look too bare?

---

## 3. Buildings

- [ ] **Buildings are where real houses/barns are** (Waterdown at the top of Snake Road, the
      houses along Sydenham Road on top, the farms and houses at Forks).
- [ ] **They sit on the ground.** Not floating, not half-buried, not tilted.
- [ ] **Size and roof look believable** (a house looks like a house, a barn like a barn).
- [ ] **No building on the road** or blocking your way.
- [ ] **Textures:** walls and roofs show siding/brick/roof patterns, no pink/black/white.

---

## 4. Bridges

Drive over each bridge slowly, then stop just past it, get out of the car view (free camera
or replay camera) and look at it from the side and from below if you can.

- [ ] **Not floating any more:** you see a concrete deck with a thickness, solid ends where
      it meets the ground, and on Sydenham Road columns (piers) down to the creek.
- [ ] **Concrete side walls** on both sides of the deck. Drive into one gently: you
      **bounce off** it and never fall off the bridge.
- [ ] **Ground under the bridge** looks like the real creek valley; no ground poking up
      through the deck or the road.
- [ ] **Driving over is smooth:** no bump where the bridge starts or ends.
- [ ] **Forks culverts** (around 2.6 and 2.8 km): short concrete walls on both sides, nothing
      odd.
- [ ] The real bridges: do they have **concrete walls** (like ours) or **steel railings**? Do
      the real roads have **guardrails on the bridge approaches** (we put 20 m on each side)?

---

## 5. Guardrails: where are the real ones?

Our tracks now have rails only next to the bridges. You know these roads: please tell us
where the **real** guardrails are, so we can add exactly those.

For each stretch, write: which track, which side **when driving from the start (A) to the
finish (B)**, and where it starts and ends, any way you can describe it (for example
"Snake Road, right side, from the sharp left to about 200 m before the bridge" or "Sydenham,
both sides, the whole first steep climb"). Rough is fine; we convert it into a small
`guardrails.json` file for each track.

- [ ] Also check: the rails we added at the bridges stand at the edge of the gravel, look
      like a W-shaped steel rail on posts, and you bounce off them.

---

## 6. Smoothness (FPS)

Use whatever FPS counter you normally use (or just tell us how it feels).

- [ ] Your usual FPS on the old (M1) tracks, and on these, **in the thickest forest**
      (Forks, the first kilometre; Snake, the climb).
- [ ] Any stutter when trees or buildings appear?
- [ ] If it's too slow, say so: we can build with fewer trees (`--scenery-density 0.5`), or
      more if it runs fine and you want a denser forest.

---

## 7. Quick re-check of the basics

These worked (or we're waiting for your M1 answers), but the scenery touched the bridges and
the ground, so please re-check quickly:

- [ ] No falling through the road or the grass anywhere, also at the bridges.
- [ ] Invisible walls still hold on both sides (the trees are behind them).
- [ ] Hotlap: the timer starts at A and stops at B.
- [ ] Race with 3 AI: they still drive the road and cross the bridges without crashing.

---

## Reply template

Copy this, fill it in, and send it back with screenshots of anything odd (a screenshot of a
good view helps too!).

```
CSP version: ...
Car used: ...
Install (.bat worked, 3 tracks replaced in CM): yes / no   note:
FPS: old tracks ... / new tracks ... (thickest forest: ...)   stutter: no / yes ...

SNAKE ROAD (Waterdown)
  Trees: in the right places yes/no   on the ground yes/no   look ok yes/no   brightness: dark/ok/bright
         pop-in: ...   any tree on the road/in a house: no/yes ...
  Buildings: right places yes/no   on the ground yes/no   look ok: ...
  Bridge: looks solid yes/no   side walls hold yes/no   ground under it ok yes/no   bump: no/yes
          real bridge has: concrete walls / steel railing   real approach rails: yes/no
  Real guardrails (side as driven A->B, where): ...
  Basics (fall-through, walls, timer, AI): ok / ...

SYDENHAM ROAD (Dundas)
  Trees: in the right places yes/no   on the ground yes/no   look ok yes/no   brightness: dark/ok/bright
         pop-in: ...   any tree on the road/in a house: no/yes ...
  Buildings: right places yes/no   on the ground yes/no   look ok: ...
  Bridge: looks solid yes/no   piers ok yes/no   side walls hold yes/no   ground under it ok yes/no
          bump: no/yes   real bridge has: concrete walls / steel railing   real approach rails: yes/no
  Real guardrails (side as driven A->B, where): ...
  Basics (fall-through, walls, timer, AI): ok / ...

FORKS OF THE CREDIT ROAD (Caledon)
  Trees: in the right places yes/no   on the ground yes/no   look ok yes/no   brightness: dark/ok/bright
         pop-in: ...   any tree on the road/in a house: no/yes ...
  Buildings: right places yes/no   on the ground yes/no   look ok: ...
  River bridge: looks solid yes/no   side walls hold yes/no   ground under it ok yes/no   bump: no/yes
  Culverts (2.6 / 2.8 km): ok / ...
  Real guardrails (side as driven A->B, where): ...
  Basics (fall-through, walls, timer, AI): ok / ...

More or fewer trees next time: more / same / fewer
Anything else: ...
```
