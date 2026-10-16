# Touge Battle in-game checklist (CSP New Mode, v0.1)

Touge Battle is a new game mode: you race **one AI car** from A to B on a road this tool built.
The first car past B wins. Then the AI brakes and parks before the end of the road, your car is
braked too, and a result screen shows the winner and the gap. Click **Race again** for another
run from A.

We could not test any of this in the game, so this is the first real test. Every step below
says what should happen. Please tell us what actually happens; the reply template is at the
end.

You need: Assetto Corsa on Windows, Content Manager (CM), **Custom Shaders Patch (CSP) 0.2.11
or newer** (the public version is fine) and about 30 minutes.

## 1. Build or rebuild a track

Use an **A to B** track (not a loop) built **with this version** of the app or CLI. New builds
have a file `data/acmg_touge.json` in the track folder, which holds the B gate and the start
spots. Older A to B tracks still work, but the chase start (section 4) needs that file.

Pick a road of 2 to 5 km with a few corners for the first test.

## 2. Install the mode

From the command line (the app gets a button later):

```
drift-maps install-touge
```

It finds Assetto Corsa the same way `drift-maps find-ac` does. If that fails, add
`--ac-root "C:\...\steamapps\common\assettocorsa"`. It copies the mode to
`<AC>\extension\lua\new-modes\acmg-touge\`. It refuses to run if CSP isn't installed.

To install it by hand instead, copy the repo folder `csp/new-modes/acmg-touge` to
`<AC>\extension\lua\new-modes\acmg-touge`.

To remove it: `drift-maps install-touge --uninstall`.

## 3. Start a Touge Battle in Content Manager

New Modes are part of CSP. Content Manager lists them with the normal modes. We have not
seen these screens ourselves, so please tell us the real menu names.

1. CM > **Drive** > Quick Drive. Open the **mode** list (Practice, Race, Hotlap ...). CSP's
   own modes, such as *Rally Stage*, *Overtake* and *Traffic*, should be in it too. Pick
   **Touge Battle (ACMG)**. If no CSP modes show up at all, update CM and CSP, and check CM >
   Settings > Custom Shaders Patch for a "new modes" or "custom modes" switch.
2. Track: your A to B track. Only tracks with the `touge` tag can be picked; our A to B tracks
   have it. Other tracks should show "A touge road made by Assetto Corsa Map Generator".
3. Car: any. **Opponents: 1**. For a fair first test, give the opponent the same car as yours.
4. Mode settings, if CM shows them (they come from the mode's `settings.ini`):
   - AI level, % (default 95)
   - AI aggression, % (default 50)
   - AI tyre grip, % (default 120, which is CSP's normal AI grip; 100 means the same grip as you)
   - Start: Side by side / Chase (AI leads, you start one car length behind) / Lead (you lead,
     the AI chases)
   - Countdown for reruns, in seconds
   - Brake your car automatically after B
5. Press **Go**.

## 4. What to check

| # | Do | What should happen |
|---|---|---|
| T1 | Wait on the grid. | Both cars are side by side just behind A, and the text "Touge Battle: First car past B wins" is shown. AC's start lights count down. Nobody moves before green. |
| T2 | Race to B. | At green you see "GO!". At the top is a strip: who leads and by how many metres, the distance to B, a timer and the score. The AI drives the whole road at race speed and does **not** slow down or stop before B. |
| T3 | Cross B (win or lose). | "You win!" or "AI wins" appears. The AI brakes hard after B and stops in the run-off, without hitting the end wall. After B your car brakes on its own; you can still steer. For a few seconds the cars pass through each other, so nobody gets rear-ended. |
| T4 | Wait. | A result window: You win / You lose, the gap (in seconds, or "more than 8 s (N m short of B)"), the times, the runs won, two sliders (AI level, AI aggression), three start buttons and **Race again**, **Restart session**, **End**. Move the mouse if you don't see the pointer. |
| T5 | Click **Race again** (or press **Shift+R**). | Both cars jump back to A, repaired. The AI is in your driving lane and you are beside it. A 3-2-1 countdown runs and both cars stay put, even on a slope. Then GO, and you can drive and steer normally. |
| T6 | On the result screen, pick **Chase (AI leads)**, then Race again. | The AI starts in front, you one car length behind it in the same lane. Try **Lead (AI chases)** as well. |
| T7 | Move the AI level slider down (say to 80) and up (say to 110), then race. | The AI is clearly slower, then faster. Same for aggression: does it defend or try to pass more? |
| T8 | Click **Restart session**, then later **End**. | Restart: the session starts over from AC's grid (the score goes back to 0). End: CSP's end-of-session screen with "You N, AI M". |
| T9 | Stop and wait more than 10 s at any point while racing (or let the AI crash and get stuck). | If the AI is stuck for 10 s, you win with the gap "AI did not finish". |

**If T1 or T2 fails** (the race doesn't start, cars spawn in odd places, or the AI stops
before B), please also try the Track Day copy of the mode:

```
drift-maps install-touge --trackday-variant
```

Then pick **Touge Battle (ACMG) - Track Day test** in CM and repeat T1 to T5. In this copy,
the first run already starts with our own countdown.

## 5. Send us

1. Your CSP version (CM > Settings > Custom Shaders Patch) and CM version.
2. The track and the two cars.
3. For each of T1 to T9: OK, or what happened instead. A short video or screenshots of the
   start, the finish and the result window help a lot.
4. The mode's log lines. Close the game, open
   `Documents\Assetto Corsa\logs\custom_shaders_patch.log` and copy every line containing
   `[touge]`. We think this is where CSP writes them; if you can't find them there, CSP's
   *Lua Debug* app shows them live. Look for `touge.state`, `touge.player_to_B_m` and
   `touge.ai_to_B_m` while you drive.

```
CSP version:            CM version:
Track:                  My car:              AI car:
Mode (Race / Track Day test):
T1 start:
T2 AI to B:
T3 finish and stopping:
T4 result window:
T5 race again:
T6 chase / lead:
T7 AI level / aggression:
T8 restart / end:
T9 stuck AI:
Anything else:
[touge] log lines:
```

---

### For the app and packaging teams (not for players)

- **Install button:** the map app could get an "Install touge mode" button (for example in My
  tracks, next to an A to B track) that calls
  `driftmaps.csp_modes.install_touge(find_ac_root())` and shows the returned folders. On a
  `ModeInstallError` it should show the message: no CSP, or a folder with the same name that
  this tool didn't install. The CLI `drift-maps install-touge` does the same.
- **Packaging:** the frozen app finds the mode in `<bundle>/csp/new-modes` (or through
  `DRIFTMAPS_CSP_MODES`). `packaging/acmg.spec` should add
  `(str(ROOT / "csp" / "new-modes"), "csp/new-modes")` to `datas`. Until then
  `install_touge` raises `ModeInstallError` ("mode files were not found") in the `.exe`.
