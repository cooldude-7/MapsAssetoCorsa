# M5 checklist: the map app, and your own roads

The app is ready to try. Pick your road on a map, press **Create map**, and a few minutes later
the real road is installed in Assetto Corsa with its hills, trees, buildings, bridges and
guardrails. There are two ways to pick the road (the switch at the top of the panel):

- **Turn by turn** (new, the default): click the road you start on, pick which way to drive,
  and at every junction pick the road you want with an arrow (or the arrow keys). You drive the
  route the way you would in the car.
- **Quick A to B:** click points (**A**, **B**, **C**...) and the app finds the roads between
  them.

It makes two kinds of track:

- **A to B** (a touge run): from the start to the finish.
- **A loop:** drive back round to A and press **Back to start**, so you can race it lap after
  lap (a circuit in Content Manager, with lap times and sectors).

This round has four goals:

1. **The app itself:** does it install, start and work on your PC without help?
2. **Turn by turn:** is picking a road junction by junction quick and clear?
3. **Your roads:** build the roads you actually drive and tell us how they come out in game.
4. **Loops:** make a loop round a few blocks you know and race it.

**Which version you have** is shown next to the app's name at the top of the page, for
example **Update 3 · 2026-10-16 · 1a2b3c4**. Please put it in your reply.

You need: your Windows PC with Assetto Corsa, Content Manager (CM) and Custom Shaders Patch
(CSP), an internet connection, and about an hour. At the end there is a reply template you can
copy and fill in.

**What's normal (not bugs):**
- **Roads in a new area can take up to a minute to load.** The app downloads the roads round
  the part of the map you're looking at from OpenStreetMap's free servers, while a
  **Loading roads... N s** chip counts at the bottom of the map. After that they are saved, so
  the same area is instant next time. The servers are sometimes busy: if you see "Couldn't
  reach the map servers", wait a minute and move the map a little to try again.
- **Turn by turn needs you zoomed in** (it says "Zoom in to pick a road" until you are).
- **A build takes a few minutes.** On our test PC, Snake Road (3.5 km, about 23,000 trees and
  147 buildings) took 3 minutes and made a 28 MB track. Longer roads take longer. The first
  build in a new area takes longer again, because it downloads elevation and tree data.
- **Detailed (1 m lidar) elevation and lidar trees are only in Canada for now.** The US and France
  also have detailed elevation. Anywhere else the app warns you that the hills will be coarse
  (30 m data).
- **Guardrails are automatic now.** By default the app puts steel guardrails where Ontario's
  road-design rules call for one (steep drops, high embankments, deep water and bridge ends),
  plus any that OpenStreetMap has. Under **More options**, **Guardrails** can be set to
  **Only where mapped** (the old behaviour) or **Off**. Bridges get their concrete side walls.
- **Turn by turn drives straight past driveways and parking-lot entrances**: it only stops at
  real junctions.
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

## 2. Warm-up: Snake Road, turn by turn (a road we know)

You already drove our Snake Road builds, so this tells us whether the app gives the same result.

1. Type **Snake Road** in the search box at the top of the map and pick **Snake Road,
   Burlington** (OpenStreetMap files Snake Road under Burlington, not Waterdown). Make sure
   **Turn by turn** is selected at the top of the panel.
2. Wait for **Loading roads...** to finish, then click on Snake Road at the **bottom**, just
   above Highway 403. That's **A**.
3. Two arrows appear: click the one pointing **up the hill** (or press the **up arrow key**).
   The blue line runs along Snake Road to the first junction (Old York Road).
4. At each junction, arrows show the roads you can take. Press **up** (straight on) to stay on
   Snake Road. Near the top, when the line reaches Waterdown, press **Enter** (or click
   **Finish here**). That's **B**.
5. The length should be about **3.5 km**, and the elevation chart under the map should climb
   about **100 m**.
6. Type the track name **Snake Road (app)**, so it doesn't replace the copy you already have.
   Leave everything under **More options** as it is, and click **Create map**.
7. When it says **Installed!**, open CM (restart it if it was open), find **Snake Road (app)**
   and drive it.

- [ ] The blue line follows Snake Road from bottom to top, about 3.5 km
- [ ] The elevation chart looks like the real climb
- [ ] Guardrails: where are they? Do they make sense (drops, the bridge), or are there too
      many / too few?
- [ ] The progress bar moved and the build finished. How long did it take? ...
- [ ] **My tracks** (bottom left of the page) lists it as **Installed**
- [ ] CM shows it with a preview picture and outline. **CM's length matches the app's length.**
- [ ] In game it looks and drives like our earlier Snake Road (start at A, timer stops at B,
      trees, houses, the small bridge near the top, no falling through)

---

## 3. Turn by turn: how it works

- **Start:** zoom in and click the road you start on. The road lights up under the mouse
  before you click. Then pick a direction with an arrow (or the arrow key that points that
  way). On a one-way street there is only one way, so it just goes.
- **At a junction:** the arrows (also listed in the panel) are the roads you can take. Click one,
  or press **left / up / right**; when there are more than three roads, the arrows are
  numbered: press **1**, **2**, **3**... A red arrow is a one-way you may not drive into.
- **Skip ahead:** click further along a road (even several junctions on) and the route drives
  straight on to there. A dashed line shows where a click would take you.
- **Undo:** **Backspace** takes the last turn back (also the **Undo** button). **Clear** starts
  again.
- **Finish:** **Enter** or **Finish here** ends the route at the junction you're at.
  **Shift+click** a road ahead (or on the last stretch) ends it right there, mid-road.
- **A loop:** drive back round until you're on A's road again, heading the way you started.
  **Back to start** then lights up and closes the loop (until then it says "Drive back to A to
  close the loop").
- The map follows you when the route runs off the screen, and new roads load as you go.

- [ ] Starting, picking the direction and taking turns with the **mouse** felt natural
- [ ] The same with the **keyboard** (arrows, numbers, Backspace, Enter)
- [ ] Clicking further along a road skipped ahead the way you expected
- [ ] Shift+click to finish mid-road worked
- [ ] How long did "Loading roads" take in a new area? Was it in the way?
- [ ] Anything confusing? (A screenshot helps.)

---

## 4. Your own roads (the main event)

Pick **one to three roads you really drive** and build each one. For a first try, a road of
2 to 8 km works best. Use **Turn by turn** for at least one of them; **Quick A to B** is still
there for the others.

Tips for picking the road:
- **Search:** type the road name. If road + town finds nothing, try the road name alone,
  because OpenStreetMap often files a road under a neighbouring town. You can also type
  coordinates like `43.30, -79.88`.
- **Zoom in** before you click, so the points land on the road itself.
- In **Quick A to B**:
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

## 5. A loop

Pick **a block or a few blocks of streets you know**, about 1 to 3 km round. Downtown
Waterdown works well if you want a known one: Flamboro Street, Dundas Street East, Mill Street
South and Main Street South go round in about 1.4 km.

1. **Turn by turn:** click your start street, pick a direction, and take the turns round the
   block. When the route is back on your start street, click **Back to start**: the line
   closes back to A and the length says **Lap length**. (Backspace opens it again.)
   **Or in Quick A to B:** click **A** on one street, then **B**, **C** on the other streets
   round the block, in the order you want to drive them, and click **Back to start** (click it
   again to go back to A to B).
2. Name it and click **Create map**.
3. In CM, find it in the track list. It should be a **circuit**: Quick Drive offers races with
   more than one lap.

- [ ] The loop line follows the streets you meant and comes back to A
- [ ] Turn by turn: **Back to start** lit up when you were back on A's road, and closed the loop
- [ ] Quick A to B: **Back to start** on and off switches between the loop and A to B
- [ ] If you tried somewhere it can't loop (a dead end), the message made sense
- [ ] In game: no bump, gap or kink where the lap starts and ends
- [ ] The timer counts laps: the lap time shows at the start/finish line, with sector times
- [ ] Hotlap: you start a little before the line and are at speed when the lap starts
- [ ] Race with 3 AI over 3 laps: they keep lapping without stopping at the line
- [ ] Practice: the car starts in a pit spot behind the grid, on the road

---

## 6. The app itself

Try these. Each one is quick:

- [ ] **Switching** between Turn by turn and Quick A to B: each keeps its own route
- [ ] Quick A to B: **Drag a point, Undo, Clear, Drive it the other way** all do what you expect
- [ ] Quick A to B, **a third click:** with A and B already down, a plain click adds C at the end (C is the
      new finish). **Shift+click** puts a point in between instead. Is that what you expected?
- [ ] **The screen:** only the route, Back to start, the track name and Create map are shown;
      the scenery switches and **Guardrails** are under **More options**. Easy to find what
      you need?
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

## 7. If something goes wrong: what to send

- **The version** at the top of the page (Update N · date · code).
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
  Version (top of the page): Update ... · ... · ...
  Downloaded from: Releases / Actions
  SmartScreen: Run anyway worked yes/no   Firewall question: appeared yes/no, clicked Allow yes/no
  Browser opened by itself: yes/no   "Assetto Corsa found": yes / no, pasted the folder and it worked: yes/no
  Problems: ...

SNAKE ROAD (app, turn by turn)
  Route: follows Snake Road yes/no   length shown: ... km   elevation chart ok: yes/no
  Loading roads took: ... s   guardrails make sense: yes / too many / too few: ...
  Build time: ... min   errors: none / ...
  CM: listed yes/no   preview ok yes/no   length in CM: ... km (app said ... km)
  In game same as our earlier Snake Road: yes / differences: ...

TURN BY TURN
  Mouse: natural yes/no   keys (arrows, numbers, Backspace, Enter): ok yes/no
  Click further along to skip ahead: ok yes/no   Shift+click finish mid-road: ok yes/no
  Confusing: ...

MY ROAD 1: name ...   where (town): ...   length: ... km   picked with: turn by turn / quick A to B
  Route: right away / needed waypoints or extra turns / couldn't get it right: ...
  Finding the road took: ... s   build took: ... min   errors: none / ...
  In game: road in the right place yes/no   corners ok yes/no   hills ok yes/no   width ok yes/no
           scenery ok yes/no: ...   timer A->B ok yes/no   3 AI ok yes/no   FPS: ...
  Best part: ...   Worst part: ...

MY ROAD 2: (same as above)

MY ROAD 3: (same as above)

LOOP: streets ...   where (town): ...   lap length: ... km
  Made with: turn by turn / quick A to B
  Line comes back to A: yes/no   Back to start ok: yes/no   messages ok: yes/no / ...
  CM shows it as a circuit (multi-lap races): yes/no
  In game: seam where the lap starts yes/no   lap/sector times ok yes/no
           hotlap start ok yes/no   3 AI over 3 laps ok yes/no   FPS: ...

THE APP
  Switching modes keeps each route: yes/no
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
