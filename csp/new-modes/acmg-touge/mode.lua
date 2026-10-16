--[[
  Touge Battle (ACMG) v0.1: a CSP New Mode for A to B roads made by Assetto Corsa Map Generator.

  1v1 against one stock AC AI car driving the track's own fast_lane.ai:
  - Start side by side at A (AC's own race start the first time; our countdown for reruns), or
    a chase start (AI ahead, you one car length behind, or the other way round).
  - The first car past the B gate wins. The loser may keep racing to B for a few seconds so the
    gap is measured; then the AI is braked to a stop and parked (instead of driving into the
    wall at the end of the run-off), your car is braked too after B, and a result screen shows
    the winner, the gap and "Race again".
  - AI level, aggression and grip come from settings.ini and can be changed between runs.

  Track data: data/acmg_touge.json (gates, grid and chase start, written by the generator); the
  AC_AB_FINISH_L/R dummies are the fallback on other A to B tracks.

  Only API present in the CSP 0.2.11 New Modes SDK (extension/internal/lua-sdk/ac_new_modes) is
  used. The physics calls need ALLOW_PHYSICS_ALTERATIONS=1 (manifest.ini).
]]

local sim = ac.getSim()
local PLAYER, AI = 0, 1

local LOSER_GRACE_S = 8 -- the loser may race on to B this long after the winner (to time the gap)
local RESULT_AFTER_S = 6 -- open the result screen at the latest this long after that
local STOPPED_KMH = 3
local AI_STUCK_S = 10 -- an AI slower than STOPPED_KMH this long while racing did not finish
local GATE_SLACK_M = 3 -- a car this far beyond the gate's ends still counts as passing B

local settings = ac.INIConfig.scriptSettings():mapSection('SETTINGS', {
  AI_LEVEL = 95,
  AI_AGGRESSION = 50,
  AI_GRIP = 120,
  START_STYLE = 'SIDE_BY_SIDE',
  COUNTDOWN = 3,
  BRAKE_AFTER_FINISH = true,
})

local STYLES = { 'SIDE_BY_SIDE', 'CHASE', 'LEAD' }
local STYLE_NAMES = { SIDE_BY_SIDE = 'Side by side', CHASE = 'Chase (AI leads)', LEAD = 'Lead (AI chases)' }

local function clamp(x, lo, hi)
  x = tonumber(x) or lo
  if x < lo then return lo end
  if x > hi then return hi end
  return x
end

-- Values for the next run (the result screen can change them).
local tune = {
  level = clamp(settings.AI_LEVEL, 50, 120),
  aggression = clamp(settings.AI_AGGRESSION, 0, 100),
  grip = clamp(settings.AI_GRIP, 80, 150),
  style = STYLE_NAMES[settings.START_STYLE] and settings.START_STYLE or 'SIDE_BY_SIDE',
  countdown = clamp(settings.COUNTDOWN, 1, 10),
}

local function log(fmt, ...)
  ac.log('[touge] ' .. string.format(fmt, ...))
end

-- ---------------------------------------------------------------------------------------------
-- Track data: the B gate and the start poses
-- ---------------------------------------------------------------------------------------------

local function v3(a)
  return vec3(tonumber(a[1]) or 0, tonumber(a[2]) or 0, tonumber(a[3]) or 0)
end

local function loadTrackData()
  local ok, data = pcall(function ()
    local raw = io.load(ac.getTrackDataFilename('acmg_touge.json'))
    return raw and JSON.parse(raw) or nil
  end)
  if ok and type(data) == 'table' and type(data.finish_gate) == 'table' then
    return data
  end
  return nil
end

-- A gate as a line across the road: its centre, the horizontal forward unit vector (fx, fz)
-- and half its length (reach).
local function flatGate(centre, forward, reach)
  local len = math.sqrt(forward.x * forward.x + forward.z * forward.z)
  if len < 1e-6 then return nil end
  return { centre = centre, fx = forward.x / len, fz = forward.z / len, reach = reach }
end

local function gateFromData(g)
  if type(g.centre) ~= 'table' or type(g.forward) ~= 'table' then return nil end
  return flatGate(v3(g.centre), v3(g.forward), tonumber(g.reach_m) or 10)
end

local function gateFromDummies(prefix)
  local l, r = ac.findNodes(prefix .. '_L'), ac.findNodes(prefix .. '_R')
  if l:empty() or r:empty() then return nil end
  local pl, pr = l:getPosition(), r:getPosition()
  local dx, dz = pr.x - pl.x, pr.z - pl.z
  -- _L is on the driver's left, so forward = up x (R - L) = (dz, 0, -dx).
  return flatGate((pl + pr) * 0.5, vec3(dz, 0, -dx), math.sqrt(dx * dx + dz * dz) / 2)
end

-- Signed distance past the gate along the road (negative before it) and across it.
local function along(g, p)
  return (p.x - g.centre.x) * g.fx + (p.z - g.centre.z) * g.fz
end

local function across(g, p)
  return (p.z - g.centre.z) * g.fx - (p.x - g.centre.x) * g.fz
end

local function poseFromData(p)
  if type(p) ~= 'table' or type(p.position) ~= 'table' or type(p.forward) ~= 'table' then
    return nil
  end
  return { pos = v3(p.position), dir = v3(p.forward) }
end

local data = loadTrackData()
local finishGate = data and gateFromData(data.finish_gate) or gateFromDummies('AC_AB_FINISH')
local finishProgress = finishGate and ac.worldCoordinateToTrackProgress(finishGate.centre) or nil
if finishProgress and finishProgress < 0 then
  finishProgress = nil -- no AI spline: distances fall back to straight lines
end

-- ---------------------------------------------------------------------------------------------
-- State
-- ---------------------------------------------------------------------------------------------

local state = 'init' -- init, grid, countdown, racing, finishing, result, error
local errorText = nil
local raceTime = 0 -- seconds since GO (paused time excluded)
local countdownLeft = 0
local decidedAt = nil -- raceTime when the race was decided
local crossedAt = {} -- [car index] = raceTime when it passed B
local prevAlong = {}
local winner = nil -- PLAYER or AI
local aiDnf = false
local aiSlowFor = 0
local aiParked = false
local resultGap = nil
local score = { player = 0, ai = 0 }
local initialPoses = nil
local controls = nil
local requestRun, requestRestart, requestEnd, requestReinit = false, false, false, false
local raceAgainButton = ac.ControlButton('acmg.touge/Race again', {
  keyboard = { key = ui.KeyIndex.R, shift = true },
})

local function setState(s)
  if state ~= s then
    log('state %s -> %s (race time %.2f s)', state, s, raceTime)
  end
  state = s
  ac.debug('touge.state', s)
end

local function fail(text)
  errorText = text
  setState('error')
  ac.setMessage('Touge Battle', text)
  log('error: %s', text)
end

local function applyAI()
  physics.setAILevel(AI, tune.level / 100)
  -- the launcher's aggression is multiplied by 0.95 (SDK note), so 100% is 0.95 here
  physics.setAIAggression(AI, tune.aggression / 100 * 0.95)
  physics.setExtraAIGrip(AI, tune.grip / 100)
  log('AI level %.0f%%, aggression %.0f%%, grip %.0f%%', tune.level, tune.aggression, tune.grip)
end

-- Pedal overrides for your car are max()-combined with your own pedals (SDK), so this only adds
-- braking; steering is left to you (math.huge = keep the original steering).
local function playerBrake(brake)
  if not controls then
    if brake <= 0 then return end
    controls = ac.overrideCarControls(PLAYER)
    if not controls then return end
  end
  controls.steer = math.huge
  controls.gas = 0
  controls.clutch = 1
  controls.handbrake = 0
  controls.brake = brake
end

local function currentPose(i)
  local c = ac.getCar(i)
  return { pos = c.position:clone(), dir = c.look:clone() }
end

local function startPoses()
  if data then
    local chase = type(data.chase) == 'table' and data.chase or {}
    local grid = type(data.grid) == 'table' and data.grid or {}
    local leader, chaser = poseFromData(chase.leader), poseFromData(chase.chaser)
    if tune.style == 'CHASE' and leader and chaser then return { player = chaser, ai = leader } end
    if tune.style == 'LEAD' and leader and chaser then return { player = leader, ai = chaser } end
    -- Side by side: the AI on its own line (AC_START_0, the driving lane), you beside it, the
    -- same as AC's grid with you starting last.
    local g0, g1 = poseFromData(grid[1]), poseFromData(grid[2])
    if g0 and g1 then return { player = g1, ai = g0 } end
  end
  return initialPoses
end

local function place(poses)
  physics.setCarPosition(PLAYER, poses.player.pos, poses.player.dir)
  physics.setCarVelocity(PLAYER, vec3(0, 0, 0))
  physics.setAICarPosition(AI, poses.ai.pos, poses.ai.dir)
  physics.setCarVelocity(AI, vec3(0, 0, 0))
end

local function resetRace()
  raceTime, decidedAt, winner, resultGap = 0, nil, nil, nil
  aiDnf, aiSlowFor, aiParked = false, 0, false
  crossedAt, prevAlong = {}, {}
end

-- A rerun: repair both cars, put them on the start and count down (they are held meanwhile).
local function beginRun()
  resetRace()
  physics.disableCarCollisions(AI, false)
  physics.setAINoInput(AI, false)
  physics.resetCarState(PLAYER)
  physics.resetCarState(AI)
  applyAI()
  local poses = startPoses()
  if poses then
    place(poses)
  end
  countdownLeft = tune.countdown
  setState('countdown')
end

local function go()
  physics.setAIStopCounter(AI, 0)
  physics.lockUserControlsFor(0)
  playerBrake(0)
  raceTime = 0
  setState('racing')
end

local function init()
  local player, ai = ac.getCar(PLAYER), ac.getCar(AI)
  if sim.carsCount < 2 or not ai then
    return fail('Touge Battle needs one AI opponent: set opponents to 1 and pick a car.')
  end
  if not finishGate then
    return fail('No B gate on this track (data/acmg_touge.json or AC_AB_FINISH_L/R): '
      .. 'build it as an A to B road with Assetto Corsa Map Generator.')
  end
  if not player.physicsAvailable then
    return -- try again next frame
  end
  initialPoses = { player = currentPose(PLAYER), ai = currentPose(AI) }
  -- Leftovers of a previous run in this session (if the script was reloaded on a restart).
  physics.disableCarCollisions(AI, false)
  physics.setAINoInput(AI, false)
  log('track data: %s; B gate at (%.1f, %.1f, %.1f), reach %.1f m; B at spline %.4f of %.0f m',
    data and 'acmg_touge.json' or 'AC_AB_FINISH dummies', finishGate.centre.x, finishGate.centre.y,
    finishGate.centre.z, finishGate.reach, finishProgress or -1, sim.trackLengthM)
  log('session type %s, started %s, start style %s', tostring(sim.raceSessionType),
    tostring(sim.isSessionStarted), tune.style)
  applyAI()
  if sim.raceSessionType == ac.SessionType.Race and not sim.isSessionStarted then
    -- First run: AC's own race start (grid and lights). Only a chase start moves the cars.
    if tune.style ~= 'SIDE_BY_SIDE' then
      local poses = startPoses()
      if poses then place(poses) end
    end
    setState('grid')
  else
    beginRun()
  end
end

-- ---------------------------------------------------------------------------------------------
-- Racing, finish and stopping
-- ---------------------------------------------------------------------------------------------

-- Returns true when car `i` passed B this frame; its crossing time is interpolated in the frame.
local function checkCrossing(i, dt)
  local p = ac.getCar(i).position
  local d = along(finishGate, p)
  local prev = prevAlong[i]
  prevAlong[i] = d
  if crossedAt[i] or not prev then return false end
  if prev < 0 and d >= 0 and d - prev < 30
      and math.abs(across(finishGate, p)) < finishGate.reach + GATE_SLACK_M then
    crossedAt[i] = raceTime - dt * (d / (d - prev))
    log('car %d passed B at %.3f s', i, crossedAt[i])
    return true
  end
  return false
end

local function decide(w)
  winner = w
  decidedAt = raceTime
  if w == PLAYER then score.player = score.player + 1 else score.ai = score.ai + 1 end
  -- The cars pass through each other from now on: nobody rear-ends a braking car after B.
  physics.disableCarCollisions(AI, true)
  setState('finishing')
end

local function gapText()
  if aiDnf then return 'AI did not finish' end
  local loser = winner == PLAYER and AI or PLAYER
  if crossedAt[loser] and crossedAt[winner] then
    return string.format('%.3f s', crossedAt[loser] - crossedAt[winner])
  end
  if finishProgress then
    local metres = (finishProgress - ac.getCar(loser).splinePosition) * sim.trackLengthM
    if metres > 0 then
      return string.format('more than %d s (%d m short of B)', LOSER_GRACE_S, math.floor(metres + 0.5))
    end
  end
  return string.format('more than %d s', LOSER_GRACE_S)
end

local function stopCars()
  local since = raceTime - decidedAt
  if crossedAt[AI] or aiDnf or since > LOSER_GRACE_S then
    if not aiParked and ac.getCar(AI).speedKmh > STOPPED_KMH then
      physics.setAIStopCounter(AI, 1) -- brake along its line, refreshed every frame
    elseif not aiParked then
      physics.setAINoInput(AI, true, false) -- parked, engine running
      aiParked = true
      log('AI parked')
    end
  end
  if settings.BRAKE_AFTER_FINISH and crossedAt[PLAYER] then
    playerBrake(math.min(1, 0.3 + (raceTime - crossedAt[PLAYER]) * 0.7))
  end
end

local function updateRace(dt)
  raceTime = raceTime + dt
  local newly = {}
  for _, i in ipairs({ PLAYER, AI }) do
    if checkCrossing(i, dt) then newly[#newly + 1] = i end
  end
  if not winner and #newly > 0 then
    local w = newly[1]
    if #newly > 1 and crossedAt[newly[2]] < crossedAt[w] then w = newly[2] end
    decide(w)
  end

  if state == 'racing' then
    local ai = ac.getCar(AI)
    if raceTime > 5 and ai.speedKmh < STOPPED_KMH then
      aiSlowFor = aiSlowFor + dt
    else
      aiSlowFor = 0
    end
    if aiSlowFor > AI_STUCK_S then
      log('AI stuck for %.0f s: it did not finish', aiSlowFor)
      aiDnf = true
      decide(PLAYER)
    end
    return
  end

  stopCars()
  if state == 'finishing' then
    local since = raceTime - decidedAt
    local loser = winner == PLAYER and AI or PLAYER
    local loserDone = crossedAt[loser] or aiDnf or since > LOSER_GRACE_S
    local aiStopped = aiParked or ac.getCar(AI).speedKmh < STOPPED_KMH
    if loserDone and (aiStopped or since > LOSER_GRACE_S + RESULT_AFTER_S) then
      resultGap = gapText()
      log('result: %s won, gap %s; score you %d, AI %d', winner == PLAYER and 'you' or 'AI',
        resultGap, score.player, score.ai)
      setState('result')
    end
  end
end

local function endSession()
  local place = score.player > score.ai and 1 or (score.player == score.ai and 2 or 4)
  local summary = string.format('You %d, AI %d', score.player, score.ai)
  ac.endSession('Touge Battle: ' .. summary, true, {
    summary = summary,
    message = string.format('Runs won: you %d, AI %d.\nLast run: %s, gap %s.', score.player,
      score.ai, winner == PLAYER and 'you won' or 'AI won', resultGap or '?'),
    place = place,
  })
end

-- A restarted session (our "Restart session" button or AC's menu) starts over from AC's grid.
ac.onSessionStart(function (sessionIndex, restarted)
  requestReinit = true
end)

function script.update(dt)
  if sim.isReplayActive then return end
  dt = sim.dt -- 0 while paused
  if requestReinit then
    requestReinit = false
    requestRun, requestRestart, requestEnd = false, false, false
    if state ~= 'error' and state ~= 'init' then
      log('session restarted')
      physics.setAIStopCounter(AI, 0)
      physics.lockUserControlsFor(0)
      playerBrake(0)
      resetRace()
      score = { player = 0, ai = 0 }
      setState('init')
    end
  end
  if state == 'init' then
    init()
    return
  end
  if state == 'error' then return end

  if raceAgainButton:pressed() and state ~= 'countdown' and state ~= 'grid' then
    requestRun = true
  end
  if requestRestart then
    requestRestart = false
    if not ac.tryToRestartSession() then
      ac.setMessage('Touge Battle', 'Could not restart the session; use Race again instead.')
    end
    return
  end
  if requestEnd then
    requestEnd = false
    endSession()
    return
  end
  if requestRun then
    requestRun = false
    beginRun()
    return
  end

  if state ~= 'racing' then
    physics.preventAIFromRetiring(AI)
  end
  if state == 'grid' then
    if sim.isSessionStarted then
      resetRace()
      go()
    end
  elseif state == 'countdown' then
    countdownLeft = countdownLeft - dt
    local hold = math.max(countdownLeft, 0)
    physics.setAIStopCounter(AI, hold + 1)
    physics.lockUserControlsFor(hold + 0.5)
    playerBrake(1)
    if countdownLeft <= 0 then go() end
  elseif state == 'racing' or state == 'finishing' then
    updateRace(dt)
  elseif state == 'result' then
    raceTime = raceTime + dt
    checkCrossing(PLAYER, dt) -- a late loser still gets braked after B
    stopCars()
  end

  if finishGate then
    ac.debug('touge.player_to_B_m', -along(finishGate, ac.getCar(PLAYER).position))
    ac.debug('touge.ai_to_B_m', -along(finishGate, ac.getCar(AI).position))
    ac.debug('touge.race_time_s', raceTime)
  end
end

-- ---------------------------------------------------------------------------------------------
-- UI
-- ---------------------------------------------------------------------------------------------

local function drawBig(text, sub)
  local ws = ac.getUI().windowSize
  local w = 640
  ui.transparentWindow('acmgTougeBig', vec2((ws.x - w) / 2, ws.y * 0.2), vec2(w, 170), true, false,
    function ()
      ui.pushFont(ui.Font.Huge)
      ui.textAligned(text, vec2(0.5, 0.5), vec2(w, 110))
      ui.popFont()
      if sub then
        ui.textAligned(sub, vec2(0.5, 0), vec2(w, 50))
      end
    end)
end

local function drawHud()
  local ws = ac.getUI().windowSize
  local w = 520
  local p, a = ac.getCar(PLAYER), ac.getCar(AI)
  local gap = (p.splinePosition - a.splinePosition) * sim.trackLengthM
  local lead = gap >= 0 and string.format('You lead by %d m', math.floor(gap + 0.5))
    or string.format('AI leads by %d m', math.floor(-gap + 0.5))
  local toB = -along(finishGate, p.position) -- straight line; along the road when possible
  if finishProgress then
    toB = (finishProgress - p.splinePosition) * sim.trackLengthM
  end
  toB = math.max(0, toB)
  ui.transparentWindow('acmgTougeHud', vec2((ws.x - w) / 2, 8), vec2(w, 54), true, false, function ()
    ui.textAligned(string.format('TOUGE BATTLE   %s   B in %.2f km   %s', lead, toB / 1000,
      ac.lapTimeToString(raceTime * 1000)), vec2(0.5, 0.5), vec2(w, 26))
    ui.textAligned(string.format('You %d : %d AI   (Shift+R: race again)', score.player, score.ai),
      vec2(0.5, 0.5), vec2(w, 22))
  end)
end

local function drawResult()
  local ws = ac.getUI().windowSize
  local w, h = 460, 360
  ui.toolWindow('acmgTougeResult', vec2((ws.x - w) / 2, ws.y * 0.16), vec2(w, h), false, true,
    function ()
      ui.pushFont(ui.Font.Title)
      ui.text(winner == PLAYER and 'You win!' or 'You lose')
      ui.popFont()
      ui.text('Gap: ' .. (resultGap or '?'))
      if crossedAt[PLAYER] then
        ui.text('Your time from the start: ' .. ac.lapTimeToString(crossedAt[PLAYER] * 1000))
      end
      if crossedAt[AI] then
        ui.text('AI time from the start: ' .. ac.lapTimeToString(crossedAt[AI] * 1000))
      end
      ui.text(string.format('Runs won: you %d, AI %d', score.player, score.ai))
      ui.separator()
      ui.text('Next run')
      ui.setNextItemWidth(w - 40)
      local level, levelChanged = ui.slider('##level', tune.level, 50, 120, 'AI level: %.0f%%')
      if levelChanged then tune.level = math.floor(level + 0.5) end
      ui.setNextItemWidth(w - 40)
      local aggr, aggrChanged = ui.slider('##aggression', tune.aggression, 0, 100, 'AI aggression: %.0f%%')
      if aggrChanged then tune.aggression = math.floor(aggr + 0.5) end
      for k, s in ipairs(STYLES) do
        if k > 1 then ui.sameLine() end
        if ui.radioButton(STYLE_NAMES[s], tune.style == s) then tune.style = s end
      end
      ui.separator()
      if ui.button('Race again (Shift+R)') then requestRun = true end
      ui.sameLine()
      if ui.button('Restart session') then requestRestart = true end
      ui.sameLine()
      if ui.button('End') then requestEnd = true end
    end)
end

function script.drawUI()
  if sim.isInMainMenu then return end
  if state == 'error' then
    drawBig('Touge Battle', errorText)
  elseif state == 'grid' then
    drawBig('Touge Battle', 'First car past B wins. Go on the green light.')
  elseif state == 'countdown' then
    drawBig(tostring(math.max(1, math.ceil(countdownLeft))), STYLE_NAMES[tune.style])
  elseif state == 'racing' or state == 'finishing' then
    if state == 'racing' and raceTime < 1 then
      drawBig('GO!')
    elseif state == 'finishing' then
      drawBig(winner == PLAYER and 'You win!' or 'AI wins')
    end
    drawHud()
  elseif state == 'result' then
    drawResult()
  end
end
