# Assetto Corsa Map Generator: how to install it

Pick a real road on a map and the app turns it into a track you can drive in Assetto Corsa.

## What you need

- Windows 10 or 11.
- Assetto Corsa from Steam.
- Content Manager (recommended): the easiest way to find and drive the new tracks.
- Custom Shaders Patch: only for the "touge" race mode. Normal tracks work without it.

## Download

1. Open https://github.com/cooldude-7/MapsAssetoCorsa/releases/latest
2. Under **Assets**, click **AssettoCorsaMapGenerator-Setup-(version).exe**.
   (The `.zip` is a portable copy that doesn't need installing. Most people want the Setup.)

## Two warnings you will probably see

The app is free and made by one person, and its .exe isn't "code-signed" (signing costs money
every year). Windows and browsers warn about any unsigned download that few people have
fetched yet. That is all these warnings mean.

- **Browser** ("isn't commonly downloaded", "uncommon", or "could be dangerous"):
  - Edge: hover over the download, click **...**, then **Keep**. If it asks again, click
    **Show more**, then **Keep anyway**.
  - Chrome: click **Keep** (or open the downloads list and click **Keep**).
- **Windows SmartScreen** ("Windows protected your PC") when you open the Setup:
  click **More info**, then **Run anyway**.

The Setup doesn't need administrator rights and doesn't ask for them.

## Checking the file (optional)

Each release lists the files' fingerprints in **SHA256SUMS.txt**. To check yours, open
PowerShell in your Downloads folder and run:

```powershell
Get-FileHash .\AssettoCorsaMapGenerator-Setup-0.4.0.exe -Algorithm SHA256
```

(Use your file's real name.) The long code it prints must match the line for that file in
SHA256SUMS.txt (capital or small letters don't matter).

With the GitHub command line tool you can also check that GitHub itself built the file from
this project's code:

```powershell
gh attestation verify .\AssettoCorsaMapGenerator-Setup-0.4.0.exe -R cooldude-7/MapsAssetoCorsa
```

## Privacy and safety, in plain words

- The app runs only on your PC. There is no account and no sign-in.
- It opens a page in your web browser, but that page comes from the app itself at
  `http://127.0.0.1`. Only your own computer can reach it, not other people on your network
  or the internet.
- To make a track it downloads public map data: roads from OpenStreetMap, ground heights
  (lidar) from Natural Resources Canada, building outlines from Microsoft and Ontario towns,
  street signs and poles from Mapillary, and the map pictures you see.
- Mapillary works out of the box with the app's built-in shared access. If you prefer, you
  can paste your own free Mapillary token in the settings; it stays on your PC. You can also
  switch Mapillary off.
- It asks GitHub once a day whether there's a newer version (you can switch that off in the
  settings). Nothing about you or your PC is sent.
- It only writes tracks into Assetto Corsa's `content\tracks` folder, and only its own tracks.
  Its settings and downloads live in `%LOCALAPPDATA%\AssettoCorsaMapGenerator`.
- To uninstall: Windows **Settings > Apps > Installed apps > Assetto Corsa Map Generator >
  Uninstall**. It asks whether to delete your settings and downloads too. Tracks you already
  installed in Assetto Corsa stay.

## First steps

1. Open **Assetto Corsa Map Generator** from the Start menu. A black window opens (keep it
   open while you use the app) and the map page opens in your browser.
2. Check the top right says **Assetto Corsa found**. If not, click it and paste your
   `assettocorsa` folder (in Steam: right-click Assetto Corsa > Manage > Browse local files).
3. Optional, Ontario only: click **Download Ontario roads** in the same panel (about 1 GB, once)
   so roads load instantly. Or skip it and just pick a road.
4. Pick your road on the map: click the road to start on, then choose each turn.
5. Give the track a name and click **Create map**. Wait for it to finish.
6. Open Content Manager, find the track under its name, and drive.

## Updates

After the first install you never have to download it again: when a new version is out, a
button at the top says **Update available: vX.Y.Z — Install**. Click it and the app updates
and reopens by itself, keeping your settings, tracks and downloads.
