# M5 checklist: the map app, and your own roads

The app is ready to try. Click points on a map (**A**, **B**, **C**...), press **Create map**,
and a few minutes later the real road is installed in Assetto Corsa with its hills, trees,
buildings and bridges. It makes two kinds of track:

- **A to B** (a touge run): from the first point to the last one.
- **A loop** (new): press **Back to start** and the road comes back to A along real streets, so
  you can race it lap after lap (a circuit in Content Manager, with lap times and sectors).

This round has three goals:

1. **The app itself:** does it install, start and work on your PC without help?
2. **Your roads:** build the roads you actually drive and tell us how they come out in game.
3. **Loops:** make a loop round a few blocks you know and race it.

You need: your Windows PC with Assetto Corsa, Content Manager (CM) and Custom Shaders Patch
(CSP), an internet connection, and about an hour. At the end there is a reply template you can
copy and fill in.

**What's normal (not bugs):**
- **Finding a road in a new area can take up to a minute.** The app downloads the roads around
  your pins from OpenStreetMap's free servers. They are sometimes busy: if you see "Couldn't
  reach the map servers", wait a minute and click again.
- **A build takes a few minutes.** On our test PC, Snake Road (3.5 km, about 23,000 trees and
  147 buildings) took 3 minutes and made a 28 MB track. Longer roads take longer. The first
  build in a new area takes longer again, because it downloads elevation and tree data.
- **Detailed (1 m lidar) elevation and lidar trees are only in Canada for now.** The US and France
  also have detailed elevation. Anywhere else the app warns you that the hills will be coarse
  (30 m data).
- **There are no guardrails unless OpenStreetMap has them.** Bridges get their concrete side
  walls.
- **Track folders are named `dm_` + the track name**, for example `dm_snake_road_app`.
- **A loop can't drive the same road twice.** If the only way back to A is back along the road
  you came, the app says "Can't make a loop here - add a point on a different road back to the
  start". Crossing your own route at a junction (a figure 8) is fine.
- **The black window must stay open** while you use the app, and especially during a build.

---

## 1. Install and start the app

Follow [`docs/INSTALL_APP.md`](../INSTALL_APP.md) sections 1 to 3. Then check:

- [ ] Downloaded the zip (from **Releases** or from **Actions**: which one?)
- [ ] Extracted it and double-clicked **AssettoCorsaMapGenerator.exe**
- [ ] SmartScreen ("Windows protected your PC"): **More info > Run anyway** worked
- [ ] Firewall question: did it appear? Did you click **Allow**?
- [ ] The black window opened, and the browser opened the app page by itself
- [ ] Top right of the page says **Assetto Corsa found**. If it says **not found**, paste your
      assettocorsa folder in the box that opens and click **Save**. Did that work?
- [ ] The map shows streets (around Hamilton the first time)

---

## 2. Warm-up: Snake Road (a road we know)

You already drove our Snake Road builds, so this tells us whether the app gives the same result.

1. Type **Snake Road** in the search box at the top of the map and pick **Snake Road,
   Burlington** (OpenStreetMap files Snake Road under Burlington, not Waterdown).
2. Click on the road at the **bottom** of Snake Road, just above Highway 403: point **A**.
3. Click on the road at the **top**, where Snake Road reaches Waterdown: point **B**.
4. Wait for the blue line. It should follow Snake Road all the way, about **3.5 km**, and the
   elevation chart under the map should climb about **100 m**.
5. Type the track name **Snake Road (app)**, so it doesn't replace the copy you already have.
   Leave everything under **More options** as it is (trees, buildings and guardrails on), and
   click **Create map**.
6. When it says **Installed!**, open CM (restart it if it was open), find **Snake Road (app)**
   and drive it.

- [ ] The blue line follows Snake Road from bottom to top, about 3.5 km
- [ ] The elevation chart looks like the real climb
- [ ] The progress bar moved and the build finished. How long did it take? ...
- [ ] **My tracks** (bottom left of the page) lists it as **Installed**
- [ ] CM shows it with a preview picture and outline. **CM's length matches the app's length.**
- [ ] In game it looks and drives like our earlier Snake Road (start at A, timer stops at B,
      trees, houses, the small bridge near the top, no falling through)

---

## 3. Your own roads (the main event)

Pick **one to three roads you really drive** and build each one. For a first try, a road of
2 to 8 km works best.

Tips for picking the road:
- **Search:** type the road name. If road + town finds nothing, try the road name alone,
  because OpenStreetMap often files a road under a neighbouring town. You can also type
  coordinates like `43.30, -79.88`.
- **Zoom in** before you click, so the points land on the road itself.
- **Points are lettered in the order you click them:** A is the start, each click adds the next
  letter at the end, and on an A to B road the last letter is the finish.
- **Wrong way?** If the blue line takes a different road than you want, **Shift+click** on the
  road you want it to use: that adds a point in between (the letters after it move up by one).
  You can also drag any point.
- **Undo** (Ctrl+Z) and **Clear** (top of the panel) fix mistakes. Right-click a point to
  remove it. **Drive it the other way** is under **More options**.
- The grey chips under the length list the roads the route uses: a quick check that it's the
  road you meant.

For each road, check:

- [ ] The route followed your road right away / needed waypoints / couldn't get it right
- [ ] How long "Finding the road" took
- [ ] The build finished. How long did it take? If it failed, copy the red message.
- [ ] In game: the road is where it really is, corners and hills feel like the real thing, and
      the road is the right width
- [ ] Scenery: trees, buildings and bridges are in the right places, on the ground, nothing
      floating or on the road
- [ ] Hotlap: the timer starts at A and stops at B
- [ ] Race with 3 AI: they drive the road without crashing
- [ ] FPS is fine (or what it drops to, and where)

---

## 4. A loop (new)

Pick **a block or a few blocks of streets you know**, about 1 to 3 km round. Downtown
Waterdown works well if you want a known one: Flamboro Street, Dundas Street East, Mill Street
South and Main Street South go round in about 1.4 km.

1. Click **A** on one street, then **B**, **C** (and more if you like) on the other streets
   round the block, in the order you want to drive them.
2. Click **Back to start**. The blue line now comes back to A, the length says **Lap length**,
   and the list ends with "Back to the start". Click it again to go back to A to B.
3. Name it and click **Create map**.
4. In CM, find it in the track list. It should be a **circuit**: Quick Drive offers races with
   more than one lap.

- [ ] The loop line follows the streets you meant and comes back to A
- [ ] **Back to start** on and off switches between the loop and A to B
- [ ] If you tried somewhere it can't loop (a dead end), the message made sense
- [ ] In game: no bump, gap or kink where the lap starts and ends
- [ ] The timer counts laps: the lap time shows at the start/finish line, with sector times
- [ ] Hotlap: you start a little before the line and are at speed when the lap starts
- [ ] Race with 3 AI over 3 laps: they keep lapping without stopping at the line
- [ ] Practice: the car starts in a pit spot behind the grid, on the road

---

## 5. The app itself

Try these. Each one is quick:

- [ ] **Drag a point, Undo, Clear, Drive it the other way** all do what you expect
- [ ] **A third click:** with A and B already down, a plain click adds C at the end (C is the
      new finish). **Shift+click** puts a point in between instead. Is that what you expected?
- [ ] **The screen:** only the points, Back to start, the track name and Create map are shown;
      the scenery switches are under **More options**. Easy to find what you need?
- [ ] **Close the browser tab during a build**, then open <http://127.0.0.1:8765/> again: the
      progress bar picks up where it was
- [ ] **Keep going during a build:** while it builds, pick your next road on the map. The
      **Create map** button comes back when the first map is done
- [ ] **Build without installing:** under **More options**, untick **Install into Assetto
      Corsa when done**. It should say **Built!** with the folder, and My tracks says
      **Not installed**
- [ ] **Delete** a test track in **My tracks**: it disappears from CM (after restarting CM),
      and your other tracks are still there
- [ ] **Quit and restart:** close the black window, start the app again. My tracks still lists
      your tracks
- [ ] **Dark map** button (top right) switches the map style
- [ ] Anything confusing, ugly or slow? A screenshot helps a lot.

---

## 6. If something goes wrong: what to send

- A **screenshot of the app page** (Windows key + Shift + S) showing the message.
- The **error text** (you can select and copy it).
- The app's **log file**: paste `%LOCALAPPDATA%\AssettoCorsaMapGenerator` into the File
  Explorer address bar and send `app.log`. **Copy it before restarting the app**, because each
  start begins a new log.
- For a road that came out wrong: the **coordinates of the points** (shown under each letter
  in the app), or the `route.json` file inside the track's folder
  (`assettocorsa\content\tracks\dm_...\route.json`).
- If the app won't start at all: open Command Prompt in the app's folder, run
  `AssettoCorsaMapGenerator.exe --self-test --network`, and send what it prints.

---

## Reply template

Copy this, fill it in, and send it back with screenshots of anything odd (and of a nice view!).

```
INSTALL
  Downloaded from: Releases / Actions
  SmartScreen: Run anyway worked yes/no   Firewall question: appeared yes/no, clicked Allow yes/no
  Browser opened by itself: yes/no   "Assetto Corsa found": yes / no, pasted the folder and it worked: yes/no
  Problems: ...

SNAKE ROAD (app)
  Route: follows Snake Road yes/no   length shown: ... km   elevation chart ok: yes/no
  Build time: ... min   errors: none / ...
  CM: listed yes/no   preview ok yes/no   length in CM: ... km (app said ... km)
  In game same as our earlier Snake Road: yes / differences: ...

MY ROAD 1: name ...   where (town): ...   length: ... km
  Route: right away / needed waypoints / couldn't get it right: ...
  Finding the road took: ... s   build took: ... min   errors: none / ...
  In game: road in the right place yes/no   corners ok yes/no   hills ok yes/no   width ok yes/no
           scenery ok yes/no: ...   timer A->B ok yes/no   3 AI ok yes/no   FPS: ...
  Best part: ...   Worst part: ...

MY ROAD 2: (same as above)

MY ROAD 3: (same as above)

LOOP: streets ...   where (town): ...   lap length: ... km
  Line comes back to A: yes/no   Back to start on/off ok: yes/no   messages ok: yes/no / ...
  CM shows it as a circuit (multi-lap races): yes/no
  In game: seam where the lap starts yes/no   lap/sector times ok yes/no
           hotlap start ok yes/no   3 AI over 3 laps ok yes/no   FPS: ...

THE APP
  Drag/Undo/Clear/Drive it the other way ok: yes / ...
  Plain click adds the next letter, Shift+click adds one in between: fine / ...
  Simpler screen (More options): fine / missed something: ...
  Closed the tab during a build, progress came back: yes/no
  Picked the next road during a build: worked yes/no
  Build without installing: "Built!" + Not installed yes/no
  Delete: gone from CM yes/no   other tracks untouched yes/no
  Restart: My tracks still there yes/no
  Confusing, ugly or slow: ...

CSP version: ...   Car used: ...
Anything else: ...
```

---

Road data © OpenStreetMap contributors (ODbL). Not affiliated with or endorsed by Kunos
Simulazioni; Assetto Corsa is their trademark, named only to state compatibility.
