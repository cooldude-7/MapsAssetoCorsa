# Assetto Corsa Map Generator: get the app and use it (Windows)

Click two points on a map and the app turns the real road between them into an Assetto Corsa
track, installs it, and you drive it from Content Manager. Everything runs on your own PC.
You don't install Python or anything else: you download a zip, unzip it and double-click.

You need: Windows 10 or 11, Assetto Corsa (Steam), Content Manager and Custom Shaders Patch
(CSP), and an internet connection (the app downloads road and elevation data).

---

## 1. Download the app

GitHub builds the app for you every time the project changes. Get it from one of two places.
Both need you to be signed in to GitHub with an account that can see the project.

### Option A: Releases (easiest, when there is one)

1. Go to <https://github.com/cooldude-7/MapsAssetoCorsa/releases>.
2. Find the newest release at the top. Under **Assets** (click the word to open it if it's
   folded), click **AssettoCorsaMapGenerator-windows.zip**.
3. The zip downloads to your **Downloads** folder.

If the page says there aren't any releases yet, use option B.

### Option B: the latest build (Actions)

1. Go to <https://github.com/cooldude-7/MapsAssetoCorsa/actions/workflows/windows-app.yml>.
2. You see a list of runs. Click the newest one with a **green tick**. (A yellow dot means it is
   still building, about 10-15 minutes. A red cross means that build failed; take an older
   green one.)
3. Scroll down to **Artifacts** at the bottom of the run page and click
   **AssettoCorsaMapGenerator-windows**.
4. It downloads as **AssettoCorsaMapGenerator-windows.zip** to your Downloads folder.

GitHub keeps these builds for 90 days. To make a fresh one: on the page from step 1 click
**Run workflow** (right side), leave the branch as it is, click the green **Run workflow**
button, and wait for the new run to get its green tick.

## 2. Unzip it (anywhere)

1. Open your **Downloads** folder.
2. Right-click **AssettoCorsaMapGenerator-windows.zip** and pick **Extract All...**
3. Choose any folder you like, for example `C:\Games\AssettoCorsaMapGenerator`, and click
   **Extract**.

There's no installer and it doesn't need admin rights. Keep the `_internal` folder next to
`AssettoCorsaMapGenerator.exe`: the app needs everything in it. Don't run the app from inside
the zip without extracting it first.

## 3. Start it

1. In the folder you extracted, double-click **AssettoCorsaMapGenerator.exe** (the icon with a
   road and a red map pin). With the Releases zip it is one folder deeper, inside
   `AssettoCorsaMapGenerator`.
2. **"Windows protected your PC"** (blue SmartScreen box): click **More info**, then
   **Run anyway**. Windows shows this for any app that isn't code-signed (signing costs money).
   It only asks once for each new download.
3. **Firewall question** ("Windows Defender Firewall has blocked some features"): click
   **Allow**. The app only talks to your own PC (address `127.0.0.1`) and downloads map data.
4. A black window opens and says:

   > Assetto Corsa Map Generator is running - keep this window open. Close it to quit.

5. Your web browser opens the app page by itself. If it doesn't, type the address shown in the
   black window into your browser, usually <http://127.0.0.1:8765/>.

The first time, the app looks for Assetto Corsa. The top bar says **Assetto Corsa found** or
**Assetto Corsa not found**. If it's not found, see [Troubleshooting](#assetto-corsa-not-found).

## 4. Make a track in 6 steps

1. **Find your road.** Type a road or a town in the search box at the top of the map (or press
   `/`), or drag and zoom the map.
2. **Click your points.** Click where the road should start (point **A**), then further along
   the road (**B**), and more points if you like (**C**, **D**...): each click adds the next
   letter at the end, and the last one is the finish. The app follows the real roads through
   them in that order and draws the route. To force it along a certain road, **Shift+click**
   on that road to add a point in between. Drag a point to move it, right-click it to remove
   it; **Undo** and **Clear** are at the top of the panel.
3. **A loop?** Click **Back to start** and the road comes back to A: a circuit you can race lap
   after lap. Click it again for A to B. Check the line follows the roads you want; the panel
   shows the length and an elevation chart of the hills.
4. **Name it.** Type a **Track name**. Trees, buildings, guardrails, the scenery density,
   **Drive it the other way** and **Install into Assetto Corsa when done** are under
   **More options** (all on by default).
5. **Click "Create map".** A progress bar shows what it's doing. It takes a few minutes; the
   first map in a new area takes longer because it downloads map and elevation data. Keep the
   black window open until it's done.
6. **Drive it.** When it says **Installed!**, open Content Manager, go to **Drive**, pick the
   track by its name in the track list, choose your car and go. If Content Manager was already
   open and the track isn't there, close and reopen Content Manager.

Tracks you've built are listed under **My tracks** in the app, where you can also delete them.
The app only ever lists or deletes tracks it made itself; your other tracks are never touched.

## Where tracks go

- **Installed tracks:** your Assetto Corsa tracks folder, for example
  `C:\Program Files (x86)\Steam\steamapps\common\assettocorsa\content\tracks\dm_snake_road`
  (folder names start with `dm_`).
- **The app's own folder:** `%LOCALAPPDATA%\AssettoCorsaMapGenerator` (paste that into the
  File Explorer address bar). It holds your settings, downloaded map data (`cache`), tracks
  built without installing (`tracks`) and the app's log file (`app.log`).

Your settings and tracks are kept there, not next to the .exe, so updating the app doesn't lose
anything.

## Quit, update, remove

- **Quit:** close the black window (the X at its top right), or click in it and press
  `Ctrl+C`. Closing only the browser tab does not stop the app. Closing the window during a
  build stops that build.
- **Opened it twice?** Double-clicking the .exe while it's already running just opens the page
  again.
- **Update:** download the new zip and extract it over the old folder (or into a new one).
- **Remove:** delete the folder you extracted. To also free the downloaded map data, delete
  `%LOCALAPPDATA%\AssettoCorsaMapGenerator`. Installed tracks stay in Assetto Corsa until you
  delete them (in **My tracks**, or the `dm_...` folders in `content\tracks`).

---

## Troubleshooting

### Firewall question
Click **Allow**. If you clicked Cancel by mistake, the app usually still works because it only
talks to your own PC. If the page won't load, open **Windows Security > Firewall & network
protection > Allow an app through firewall**, and tick **AssettoCorsaMapGenerator**.

### SmartScreen or antivirus blocked it
SmartScreen: **More info > Run anyway**. If Windows Security removed the .exe, open
**Windows Security > Virus & threat protection > Protection history**, pick the entry for
AssettoCorsaMapGenerator, and choose **Allow** or **Restore**. Then extract the zip again.

### Assetto Corsa not found
1. Find your Assetto Corsa folder: in Steam, right-click **Assetto Corsa > Manage > Browse local
   files**. A folder opens (it contains `acs.exe`).
2. Click that folder's address bar at the top of File Explorer and copy it (`Ctrl+C`).
   It looks like `C:\Program Files (x86)\Steam\steamapps\common\assettocorsa`.
3. In the app, click **Assetto Corsa not found** at the top, paste the folder (`Ctrl+V`) and
   click **Save**. The `content` or `content\tracks` folder inside it works too.

### Map servers busy, or the route won't load
Road data comes from free, shared OpenStreetMap servers, and elevation from government data
servers. They are sometimes busy or slow. Wait a minute and try again (move a point again or
press **Create map** again). Also check that your internet works. Very long routes take longer;
try a shorter one first.

### The browser didn't open, or the page won't load
Type the address shown in the black window into your browser (usually
<http://127.0.0.1:8765/>; if that port was busy the app picks the next one, like 8766). Make
sure the black window is still open.

### The black window closes right away, or shows an error
- Make sure you extracted the zip (step 2) and that the `_internal` folder is next to the .exe.
- Look at `%LOCALAPPDATA%\AssettoCorsaMapGenerator\app.log`, and send it along when you ask
  for help.
- For whoever helps you: open Command Prompt in the app folder and run
  `AssettoCorsaMapGenerator.exe --self-test` (add `--network` to also test the internet
  downloads). Every line should say `ok`.

### The track isn't in Content Manager
Close and reopen Content Manager. Check that **Install into Assetto Corsa when done** was ticked.
If **My tracks** in the app shows the track as **Not installed**, copy that track's folder from
`%LOCALAPPDATA%\AssettoCorsaMapGenerator\tracks` into `assettocorsa\content\tracks`.

---

Road data © OpenStreetMap contributors (ODbL). Not affiliated with or endorsed by Kunos
Simulazioni; Assetto Corsa is their trademark, named only to state compatibility.
