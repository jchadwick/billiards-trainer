-- Hardware-free tests for the visualizer's incoming coordinate boundary.
-- Run from frontend/visualizer with: lua test_coordinates.lua
package.path = "./?.lua;./?/init.lua;" .. package.path

local now = 10
love = {
    timer = {getTime = function() return now end},
    graphics = {
        getDimensions = function() return 1000, 500 end,
        setColor = function() end,
        circle = function() end,
        ellipse = function() end,
        rectangle = function() end,
        print = function() end
    },
    math = {
        newTransform = function()
            return {translate = function() end, scale = function() end}
        end
    }
}

local Coordinates = require("core.coordinates")
local StateManager = require("core.state_manager")
local Calibration = require("rendering.calibration")
local Trajectory = require("modules.trajectory.init")
local Connection = require("modules.network.connection")

local function equal(actual, expected, message)
    assert(actual == expected, (message or "values differ") ..
        ": expected " .. tostring(expected) .. ", got " .. tostring(actual))
end

local function close(actual, expected, message)
    assert(math.abs(actual - expected) < 1e-9,
        (message or "values differ") ..
        ": expected " .. tostring(expected) .. ", got " .. tostring(actual))
end

local point, pointError = Coordinates.toCanonicalPoint(
    {x = 100, y = 200, scale = {2, 3}}, nil, nil
)
assert(point, pointError)
close(point.x, 200)
close(point.y, 600)
equal(point.scale[1], 1)
equal(point.scale[2], 1)
equal(point.coordinate_space, "4k_canonical")

local rejected = Coordinates.toCanonicalPoint({x = 1, y = 2}, nil, nil)
assert(not rejected, "untagged image points must be rejected")
rejected = Coordinates.toCanonicalPoint(
    {x = 1, y = 2, coordinate_space = Coordinates.CANONICAL_SPACE}, nil, nil
)
assert(not rejected, "canonical image points still require explicit identity scale")
rejected = Coordinates.toCanonicalPoint({x = 1, y = 2, scale = {0, 1}}, nil, nil)
assert(not rejected, "non-positive scale factors must be rejected")
rejected = Coordinates.toCanonicalPoint(
    {x = 1, y = 2, coordinate_scale = {1, 1}}, nil, nil
)
assert(not rejected, "coordinate_scale must not alias the canonical scale field")
rejected = Coordinates.toCanonicalPoint(
    {x = 1, y = 2, units = "meters", unit = "meters"}, nil, nil
)
assert(not rejected, "plural units metadata must be rejected")
rejected = Coordinates.toCanonicalPoint({100, 200, scale = {2, 3}}, nil, nil)
assert(not rejected, "array point coordinates must be rejected")
rejected = Coordinates.toCanonicalPoint({
    x = 1, y = 2, scale = {1, 1}, coordinate_space = "camera_pixels"
}, nil, nil)
assert(not rejected, "only backend pixel and canonical coordinate spaces are supported")
local backendPixelPoint = Coordinates.toCanonicalPoint({
    x = 100, y = 200, scale = {2, 3}, coordinate_space = "pixel"
})
assert(backendPixelPoint, "backend pixel coordinate space should be accepted")
close(backendPixelPoint.x, 200)
close(backendPixelPoint.y, 600)
rejected = Coordinates.toCanonicalPoint({x = 0/0, y = 2, scale = {1, 1}}, nil, nil)
assert(not rejected, "non-finite coordinates must be rejected")
rejected = Coordinates.toCanonicalPoint({
    x = 1, y = 2, scale = {1, 1}, coordinate_space = "canonical_4k"
}, nil, nil)
assert(not rejected, "noncanonical coordinate-space aliases must be rejected")
local malformedVector = Coordinates.normalizeGeometry({
    position = {1, 0/0, scale = {1, 1}}
})
assert(not malformedVector, "malformed coordinate arrays must be rejected")
local arrayBoundary = Coordinates.normalizeGeometry({
    scale = {1, 1},
    playing_area_corners = {{1, 2}, {3, 2}, {3, 4}, {1, 4}}
})
assert(not arrayBoundary, "table corners must be scaled x/y objects")
local legacyScaleGeometry = Coordinates.normalizeGeometry({
    position = {x = 1, y = 2, coordinate_scale = {1, 1}}
})
assert(not legacyScaleGeometry, "coordinate_scale must not pass through nested geometry")
local legacyUnitsGeometry = Coordinates.normalizeGeometry({
    position = {x = 1, y = 2, scale = {1, 1}, units = "meters", unit = "meters"}
})
assert(not legacyUnitsGeometry, "plural units must not pass through nested geometry")
local ambiguousBoundary = Coordinates.normalizeGeometry({
    unrecognized_geometry = {x = 1, y = 2}
})
assert(not ambiguousBoundary, "unscaled coordinate records must not pass through")
local unscaledCanonicalBounds = Coordinates.normalizeGeometry({
    bounds = {
        left = 1, top = 2, right = 3, bottom = 4,
        coordinate_space = Coordinates.CANONICAL_SPACE
    }
})
assert(not unscaledCanonicalBounds,
    "canonical bounds still require explicit identity scale")
local inheritedCanonicalScale = Coordinates.normalizeGeometry({
    coordinate_space = Coordinates.CANONICAL_SPACE,
    scale = {1, 1},
    bounds = {left = 1, top = 2, right = 3, bottom = 4}
})
assert(inheritedCanonicalScale,
    "bounds may inherit an explicit parent canonical scale")
assert(not Coordinates.normalizeGeometry({
    scale = {1, 1},
    bounds = {1, 2, 3, 4}
}), "array-shaped table boundaries must be rejected")

local tableGeometry, geometryError = Coordinates.normalizeGeometry({
    coordinate_space = "pixel",
    scale = {2, 3},
    playing_area_corners = {
        {x = 10, y = 20}, {x = 30, y = 20},
        {x = 30, y = 40}, {x = 10, y = 40}
    },
    bounds = {left = 10, top = 20, right = 30, bottom = 40},
    detection_region = {x = 10, y = 20, width = 30, height = 40, scale = {2, 3}},
    width = 2.1,
    height = 1.05
})
assert(tableGeometry, geometryError)
close(tableGeometry.playing_area_corners[3].x, 60)
close(tableGeometry.playing_area_corners[3].y, 120)
close(tableGeometry.bounds.left, 20)
close(tableGeometry.bounds.top, 60)
close(tableGeometry.bounds.right, 60)
close(tableGeometry.bounds.bottom, 120)
equal(tableGeometry.bounds.scale[1], 1)
close(tableGeometry.detection_region.width, 60)
close(tableGeometry.detection_region.height, 120)
close(tableGeometry.width, 2.1) -- physical dimensions remain meters

local scaledPixelDimensions, dimensionsError = Coordinates.normalizeGeometry({
    coordinate_space = "pixel",
    dimensions_unit = "pixels",
    dimensions_scale = {2, 3},
    width = 100,
    height = 40
})
assert(scaledPixelDimensions, dimensionsError)
close(scaledPixelDimensions.width, 200)
close(scaledPixelDimensions.height, 120)
equal(scaledPixelDimensions.dimensions_scale[1], 1)
equal(scaledPixelDimensions.dimensions_scale[2], 1)
equal(scaledPixelDimensions.dimensions_unit, "canonical_pixels")
local scaledPixelAliases = Coordinates.normalizeGeometry({
    dimensions_unit = "pixels",
    dimensions_scale = {2, 3},
    width_px = 50,
    height_px = 25
})
assert(not scaledPixelAliases,
    "pixel table extents must use the canonical width and height fields")
assert(not Coordinates.normalizeGeometry({width_px = 50, height_px = 25}),
    "untyped pixel extent aliases must not pass through geometry normalization")
rejected = Coordinates.toCanonicalPoint({
    x = 1, y = 2, width_px = 5, scale = {2, 2}
}, nil, nil)
assert(not rejected, "pixel point extents must use canonical width and height fields")
local physicalDimensions = Coordinates.normalizeGeometry({
    dimensions_unit = "meters",
    dimensions_scale = {2, 3},
    width = 2.1,
    height = 1.05
})
close(physicalDimensions.width, 2.1)
close(physicalDimensions.height, 1.05)
local missingDimensionScale = Coordinates.normalizeGeometry({
    dimensions_unit = "pixels",
    width = 100,
    height = 40
})
assert(not missingDimensionScale,
    "pixel table extents must not pass through without their scale")
local incompatibleDimensionSpace = Coordinates.normalizeGeometry({
    coordinate_space = "normalized_table",
    dimensions_unit = "pixels",
    dimensions_scale = {2, 2},
    width = 100,
    height = 40
})
assert(not incompatibleDimensionSpace,
    "pixel extents cannot be declared in normalized table space")
local missingCanonicalDimensionScale = Coordinates.normalizeGeometry({
    dimensions_unit = "canonical_pixels",
    width = 100,
    height = 40
})
assert(not missingCanonicalDimensionScale,
    "canonical pixel extents require an explicit identity scale")

local function canonicalPoint(x, y)
    return {
        x = x,
        y = y,
        scale = {1, 1},
        coordinate_space = Coordinates.CANONICAL_SPACE
    }
end

local playingArea = {
    canonicalPoint(384, 216),
    canonicalPoint(3456, 216),
    canonicalPoint(3072, 1944),
    canonicalPoint(768, 1944)
}
local cornerTargets = {{0, 0}, {1, 0}, {1, 1}, {0, 1}}
for index, corner in ipairs(playingArea) do
    local u, v = Coordinates.toTablePlane(corner, playingArea)
    close(u, cornerTargets[index][1], "playing-area corner u mapping")
    close(v, cornerTargets[index][2], "playing-area corner v mapping")
end
-- This point is the projective image of local table point (0.75, 0.25) for the
-- trapezoid above. Its expected location differs from full-camera normalization.
local projectiveX, projectiveY = 2628.9230769231, 747.6923076923
local projectiveU, projectiveV = Coordinates.toTablePlane(
    canonicalPoint(projectiveX, projectiveY), playingArea
)
close(projectiveU, 0.75, "projective quadrilateral u mapping")
close(projectiveV, 0.25, "projective quadrilateral v mapping")
local centerX, centerY = Coordinates.toTablePlane(
    canonicalPoint(1920, 1203.4285714286), playingArea
)
close(centerX, 0.5, "projective quadrilateral center u")
close(centerY, 0.5, "projective quadrilateral center v")
local missingScale = {x = 384, y = 216, coordinate_space = Coordinates.CANONICAL_SPACE}
assert(not Coordinates.toTablePlane(missingScale, playingArea),
    "projected image points must carry an explicit scale")
assert(not Coordinates.toTablePlane(canonicalPoint(100, 100), nil),
    "image points must not project without calibrated table bounds")
local crossedArea = {playingArea[1], playingArea[3], playingArea[2], playingArea[4]}
assert(not Coordinates.toTablePlane(canonicalPoint(1920, 1080), crossedArea),
    "crossed playing-area corners must be rejected")

local world = Coordinates.normalizeGeometry({
    x = 1.2, y = 0.4, unit = "meters"
})
close(world.x, 1.2)
close(world.y, 0.4)
equal(world.unit, "meters")
local worldBounds = Coordinates.normalizeGeometry({
    bounds = {left = 0.2, top = 0.1, right = 1.2, bottom = 0.8},
    coordinate_space = "world_meters"
})
close(worldBounds.bounds.right, 1.2)
close(worldBounds.bounds.bottom, 0.8)
local normalizedBounds = Coordinates.normalizeGeometry({
    bounds = {
        left = 0.1,
        top = 0.2,
        right = 0.9,
        bottom = 0.8,
        scale = {1, 1},
        coordinate_space = "normalized_table"
    }
})
equal(normalizedBounds.bounds.coordinate_space, "normalized_table")
close(normalizedBounds.bounds.right, 0.9)
local scaledNormalizedBounds = Coordinates.normalizeGeometry({
    bounds = {
        left = 0.1,
        top = 0.2,
        right = 0.9,
        bottom = 0.8,
        scale = {2, 2},
        coordinate_space = "normalized_table"
    }
})
assert(not scaledNormalizedBounds,
    "normalized table bounds must reject non-identity scaling")
local normalizedTablePoint, normalizedTableError = Coordinates.normalizeGeometry({
    x = 0.25,
    y = 0.75,
    scale = {1, 1},
    coordinate_space = "normalized_table"
})
assert(normalizedTablePoint, normalizedTableError)
equal(normalizedTablePoint.coordinate_space, "normalized_table")
local missingNormalizedScale = Coordinates.normalizeGeometry({
    x = 0.25,
    y = 0.75,
    coordinate_space = "normalized_table"
})
assert(not missingNormalizedScale,
    "normalized table coordinates require explicit identity scale")
local mixedPhysicalPoint = Coordinates.normalizeGeometry({
    x = 0.25,
    y = 0.75,
    unit = "meters",
    coordinate_space = "pixel"
})
assert(not mixedPhysicalPoint,
    "physical meter points cannot be tagged as image pixels")

local state = StateManager.new()
state:updateFromStateMessage({
    balls = {
        {
            id = "scaled",
            position = {x = 100, y = 200, scale = {2, 3}},
            velocity = {x = 4, y = 5, scale = {2, 3}},
            radius = 0.028575,
            radius_unit = "meters",
            is_cue_ball = true
        },
        {
            id = "ambiguous",
            position = {x = 10, y = 20},
            velocity = {x = 1, y = 1}
        },
        {
            id = "scaled_radius",
            position = {x = 100, y = 200, scale = {2, 3}},
            velocity = {x = 0, y = 0, scale = {2, 3}},
            radius = 12,
            radius_unit = "pixels",
            radius_scale = {2, 2}
        },
        {
            id = "untyped_radius",
            position = {x = 100, y = 200, scale = {2, 3}},
            velocity = {x = 0, y = 0, scale = {2, 3}},
            radius = 12
        }
    },
    table = {
        width = 2.1,
        height = 1.05,
        playing_area_corners = {
            {x = 1, y = 2, scale = {2, 3}},
            {x = 3, y = 2, scale = {2, 3}},
            {x = 3, y = 4, scale = {2, 3}},
            {x = 1, y = 4, scale = {2, 3}}
        },
        pocket_positions = {{x = 5, y = 6, scale = {2, 3}}}
    }
})
local ball = state:getBallById("scaled")
assert(ball, "valid scale-tagged ball should be retained")
close(ball.position.x, 200)
close(ball.position.y, 600)
close(ball.velocity.x, 8)
close(ball.velocity.y, 15)
close(ball.radius, 0.028575)
local scaledRadiusBall = state:getBallById("scaled_radius")
assert(scaledRadiusBall, "valid scaled-radius ball should be retained")
close(scaledRadiusBall.radius, 24)
equal(scaledRadiusBall.radius_unit, "canonical_pixels")
equal(scaledRadiusBall.radius_scale[1], 1)
equal(state:getBallById("untyped_radius"), nil,
    "untyped radius must not survive state normalization")
equal(state:getBallById("ambiguous"), nil)
close(state:getTable().playing_area_corners[4].x, 2)
close(state:getTable().playing_area_corners[4].y, 12)
close(state:getTable().pocket_positions[1].x, 10)
close(state:getTable().pocket_positions[1].y, 18)

state:updateFromMotionMessage({
    ball_id = "scaled",
    position = {x = 50, y = 60, scale = {4, 2}},
    velocity = {x = 2, y = 3, scale = {4, 2}},
    is_moving = true
})
close(ball.position.x, 200)
close(ball.position.y, 120)
close(ball.velocity.x, 8)
close(ball.velocity.y, 6)

Trajectory:updateTrajectory({
    scale = {2, 2},
    lines = {{
        start = {x = 960, y = 540},
        ["end"] = {x = 1920, y = 1080},
        type = "primary"
    }},
    collisions = {{
        position = {x = 100, y = 200, scale = {2, 3}},
        scale = {2, 3},
        type = "ball"
    }}
})
equal(#Trajectory.paths, 1)
close(Trajectory.paths[1].points[1].x, 1920)
close(Trajectory.paths[1].points[1].y, 1080)
close(Trajectory.paths[1].points[2].x, 3840)
close(Trajectory.paths[1].points[2].y, 2160)
equal(Trajectory.paths[1].points[1].scale[1], 1)
equal(#Trajectory.collisions, 1)
close(Trajectory.collisions[1].position.x, 200)
close(Trajectory.collisions[1].position.y, 600)
equal(Trajectory.ghostBalls, nil,
    "the backend trajectory contract does not include ghost-ball data")

local renderedTrajectory
_G.StateManager = state
_G.Calibration = Calibration
_G.Renderer = {
    drawTrajectory = function(_, points) renderedTrajectory = points end,
    drawDashedLine = function() end,
    reset = function() end
}
Trajectory.colors = {
    primary = {1, 1, 1, 1},
    collision = {1, 1, 1, 1},
    ghost = {1, 1, 1, 1},
    aimLine = {1, 1, 1, 1}
}
Calibration:calculateTransform()

-- State corners arrive tagged in their source camera resolution and are
-- canonicalized before a trajectory is projected into the local table plane.
local sourceCorners = {
    {x = 192, y = 108, scale = {2, 2}},
    {x = 1728, y = 108, scale = {2, 2}},
    {x = 1536, y = 972, scale = {2, 2}},
    {x = 384, y = 972, scale = {2, 2}}
}
state:updateFromStateMessage({
    table = {
        width = 3072,
        height = 1728,
        dimensions_scale = {1, 1},
        playing_area_corners = sourceCorners,
        real_world_dimensions = {width = 2, height = 1, unit = "meters"}
    }
})
Trajectory:updateTrajectory({
    lines = {{
        start = {x = 1314.4615384616, y = 373.8461538462, scale = {2, 2}},
        ["end"] = {x = 960, y = 601.7142857143, scale = {2, 2}}
    }}
})
Trajectory:draw()
assert(renderedTrajectory and #renderedTrajectory == 2,
    "scaled trajectory over calibrated bounds should be rendered")
close(renderedTrajectory[1].x, 700, "projective point projector x")
close(renderedTrajectory[1].y, 150, "projective point projector y")
close(renderedTrajectory[2].x, 500, "table center projector x")
close(renderedTrajectory[2].y, 250, "table center projector y")

local firstResolution = {renderedTrajectory[2].x, renderedTrajectory[2].y}
Trajectory:updateTrajectory({
    lines = {{
        start = {x = 657.2307692308, y = 186.9230769231, scale = {4, 4}},
        ["end"] = {x = 480, y = 300.8571428572, scale = {4, 4}}
    }}
})
Trajectory:draw()
close(renderedTrajectory[2].x, firstResolution[1],
    "equivalent 960p point must have same projector x")
close(renderedTrajectory[2].y, firstResolution[2],
    "equivalent 960p point must have same projector y")

local dimensionsOnly = {
    width = 9999,
    height = 8888,
    table_meter_origin = Coordinates.TABLE_METER_ORIGIN,
    real_world_dimensions = {width = 2, height = 1, unit = "meters"}
}
state:updateFromStateMessage({table = dimensionsOnly})
local meterX, meterY = Trajectory.projectPoint(
    {x = 1, y = 0.5, unit = "meters", origin = Coordinates.TABLE_METER_ORIGIN},
    Calibration
)
close(meterX, 500, "meter coordinate uses physical width")
close(meterY, 250, "meter coordinate uses physical height")
local noMeterDimensions = {width = 2, height = 1}
assert(not Coordinates.toPhysicalTablePlane(
    {x = 1, y = 0.5, unit = "meters", origin = Coordinates.TABLE_METER_ORIGIN},
    noMeterDimensions
), "pixel table extents must not stand in for meter dimensions")
assert(not Coordinates.toPhysicalTablePlane(
    {x = 1, y = 0.5, unit = "meters", origin = "camera_origin"}, dimensionsOnly
), "meter points with an incompatible origin must be rejected")
assert(not Coordinates.toPhysicalTablePlane(
    {x = 1, y = 0.5, unit = "meters"},
    {real_world_dimensions = {width = 2, height = 1, unit = "meters"}}
), "meter points without an explicit origin must be rejected")

-- Without calibrated corners, canonical camera points fail closed instead of
-- being normalized against the full 4K image canvas.
state:updateFromStateMessage({table = dimensionsOnly})
Trajectory:updateTrajectory({
    lines = {{
        start = canonicalPoint(1920, 1080),
        ["end"] = canonicalPoint(1930, 1090)
    }}
})
Trajectory:draw()
assert(#renderedTrajectory == 0, "image points must fail closed without bounds")

Trajectory:updateTrajectory({
    lines = {{start = {x = 1, y = 1}, ["end"] = {x = 2, y = 2}}},
    collisions = {{x = 1, y = 2}}
})
equal(#Trajectory.paths, 0)
equal(#Trajectory.collisions, 0)

Trajectory:updateAimLine({
    start = {x = 960, y = 540},
    finish = {x = 1920, y = 1080},
    scale = {2, 2}
})
assert(Trajectory.aimLine)
close(Trajectory.aimLine.start.x, 1920)
close(Trajectory.aimLine.finish.y, 2160)
Trajectory:updateAimLine({x1 = 1, y1 = 2, x2 = 3, y2 = 4, scale = {1, 1}})
equal(Trajectory.aimLine, nil)

Trajectory:updateTrajectory({
    scale = {3, 4},
    collisions = {{position = {x = 2, y = 3}}}
})
equal(#Trajectory.collisions, 1,
    "nested collision positions may inherit an explicit envelope scale")
close(Trajectory.collisions[1].position.x, 6)
close(Trajectory.collisions[1].position.y, 12)
Trajectory:updateTrajectory({collisions = {{x = 2, y = 3, scale = {1, 1}}}})
equal(#Trajectory.collisions, 0,
    "scalar collision coordinates must be rejected without a nested position")

local flatMeterDimensions = {
    real_world_dimensions_unit = "meters",
    real_width_meters = 2,
    real_height_meters = 1
}
local flatWidth, flatHeight = Coordinates.realWorldDimensions(flatMeterDimensions)
equal(flatWidth, nil, "flat legacy meter dimensions must be rejected")
equal(flatHeight, nil, "flat legacy meter dimensions must be rejected")

local connection = Connection.new({
    reconnectInitialDelay = 2,
    reconnectMaxDelay = 8,
    heartbeatTimeout = 12,
    heartbeatInterval = 3
})
equal(connection.reconnectInitialDelay, 2)
equal(connection.reconnectMaxDelay, 8)
equal(connection.heartbeatTimeout, 12)
equal(connection.heartbeatInterval, 3)
connection:configure({reconnectInitialDelay = 4, heartbeatTimeout = 20})
equal(connection.reconnectInitialDelay, 4)
equal(connection.heartbeatTimeout, 20)
connection:configure({reconnectInterval = 100, connectionTimeout = 200})
equal(connection.reconnectInitialDelay, 4,
    "legacy reconnectInterval must not configure the connection manager")
equal(connection.heartbeatTimeout, 20,
    "legacy connectionTimeout must not configure the connection manager")

local savedConfig = _G.Config
local savedMessageHandler = _G.MessageHandler
local savedWebSocketLibrary = package.preload["lib.websocket"]
local savedNetworkWebSocket = package.loaded["modules.network.websocket"]
package.preload["lib.websocket"] = function()
    return {new = function() return {} end}
end
local networkConfig = {
    ["network.websocket_url"] = "ws://127.0.0.1:8000/api/v1/websocket/ws",
    ["network.auto_connect"] = false,
    ["network.reconnect_enabled"] = false,
    ["network.reconnect_delay"] = 1500,
    ["network.max_reconnect_delay"] = 45000,
    ["network.heartbeat_interval"] = 25000
}
_G.Config = {get = function(_, key) return networkConfig[key] end}
_G.MessageHandler = {}
local Network = require("modules.network.init")
Network:init()
equal(Network.config.reconnectInitialDelay, 1.5,
    "millisecond reconnect configuration must convert to seconds")
equal(Network.config.reconnectMaxDelay, 45,
    "millisecond maximum reconnect configuration must convert to seconds")
equal(Network.config.heartbeatInterval, 25,
    "millisecond heartbeat configuration must convert to seconds")
equal(Network.connection.autoReconnect, false,
    "an explicit false reconnect setting must be preserved")
equal(Network.websocket.reconnect_delay, 1500,
    "WebSocket reconnect delay must remain in milliseconds")
equal(Network.websocket.max_reconnect_delay, 45000,
    "WebSocket maximum reconnect delay must remain in milliseconds")
equal(Network.websocket.host, "127.0.0.1")
equal(Network.websocket.port, 8000)
equal(Network.websocket.path, "/api/v1/websocket/ws")
Network:configure({websocket_url = "ws://example.test:8123/api/v1/websocket/ws"})
equal(Network.config.host, "example.test",
    "configured WebSocket URLs must preserve their host")
equal(Network.config.port, 8123,
    "configured WebSocket URLs must preserve their port")
equal(Network.websocket.path, "/api/v1/websocket/ws",
    "configured WebSocket URLs must preserve their API path")
Network:configure({
    host = "127.0.0.1",
    port = 8000,
    path = "/api/v1/websocket/ws"
})
equal(Network.config.websocket_url, nil,
    "host/port/path updates must replace a prior full URL")
Network:configure({
    reconnectInitialDelay = 2.5,
    reconnectMaxDelay = 20,
    heartbeatInterval = 4,
    heartbeatTimeout = 8
})
equal(Network.connection.reconnectInitialDelay, 2.5)
equal(Network.connection.reconnectMaxDelay, 20)
equal(Network.connection.heartbeatInterval, 4)
equal(Network.connection.heartbeatTimeout, 8)
_G.Config = savedConfig
_G.MessageHandler = savedMessageHandler
package.preload["lib.websocket"] = savedWebSocketLibrary
package.loaded["modules.network.websocket"] = savedNetworkWebSocket
package.loaded["modules.network.init"] = nil

Calibration:calculateTransform()
local missingTableX = Calibration:transformCanonical(1920, 1080)
assert(not missingTableX, "canonical calibration must fail without table bounds")
state:updateFromStateMessage({table = {playing_area_corners = playingArea}})
local screenX, screenY = Calibration:transformCanonical(1920, 1203.4285714286)
close(screenX, 500)
close(screenY, 250)
local outputScale = Calibration:getOutputScale()
close(outputScale.width, 800)
close(outputScale.height, 400)
equal(Calibration.corners[1].scale[1], 1)
equal(Calibration.corners[1].coordinate_space, "normalized_projector")

Trajectory:updateTrajectory({
    lines = {{
        type = "primary",
        coordinate_space = "normalized_table",
        scale = {1, 1},
        start = {x = 0.25, y = 0.75},
        ["end"] = {x = 0.5, y = 0.5}
    }}
})
Trajectory:draw()
close(renderedTrajectory[1].x, 300,
    "normalized table path maps through projector calibration")
close(renderedTrajectory[1].y, 350,
    "normalized table path preserves its own coordinate space")

state:updateFromStateMessage({table = {
    playing_area_corners = {
        canonicalPoint(384, 216), canonicalPoint(3456, 216),
        canonicalPoint(3456, 1944), canonicalPoint(384, 1944)
    }
}})
Trajectory:updateTrajectory({
    paths = {{points = {canonicalPoint(1920, 1080), canonicalPoint(1930, 1090)}}},
    ghostBalls = {{x = 1920, y = 1080, radius = 12}}
})
equal(#Trajectory.paths, 0, "legacy path arrays must not be accepted")
equal(Trajectory.ghostBalls, nil, "unsupported ghost-ball payloads must be ignored")

local unitX, unitY = Coordinates.toUnitPlane(point)
close(unitX, 200 / Coordinates.CANONICAL_WIDTH)
close(unitY, 600 / Coordinates.CANONICAL_HEIGHT)

Calibration.corners = {
    {x = 0.1, y = 0.2, scale = {1, 1}, coordinate_space = "normalized_projector"},
    {x = 0.8, y = 0.1, scale = {1, 1}, coordinate_space = "normalized_projector"},
    {x = 0.9, y = 0.85, scale = {1, 1}, coordinate_space = "normalized_projector"},
    {x = 0.2, y = 0.9, scale = {1, 1}, coordinate_space = "normalized_projector"}
}
local calibrationTargets = {{100, 100}, {800, 50}, {900, 425}, {200, 450}}
local calibrationInputs = {{0, 0}, {1, 0}, {1, 1}, {0, 1}}
for index, target in ipairs(calibrationTargets) do
    local input = calibrationInputs[index]
    local screenX, screenY = Calibration:transform(input[1], input[2])
    close(screenX, target[1], "projector calibration corner x")
    close(screenY, target[2], "projector calibration corner y")
end

-- Exercise the executable WebSocket example with a deterministic socket stub
-- so its real payloads stay covered without a server or the optional socket
-- library installed.
local exampleSockets = {}
local previousWebSocket = package.loaded["modules.network.websocket"]
package.loaded["modules.network.websocket"] = {
    new = function(options)
        local socket = {options = options, sent = {}}
        function socket:connect()
            if self.options.onopen then self.options.onopen() end
        end
        function socket:send(message)
            table.insert(self.sent, message)
        end
        function socket:getURL() return self.options.url or "ws://stub" end
        function socket:getState() return "stub" end
        function socket:isConnected() return false end
        function socket:getStats()
            return {
                messages_sent = #self.sent,
                messages_received = 0,
                bytes_sent = 0,
                bytes_received = 0,
                uptime = 0,
                reconnect_attempts = 0
            }
        end
        table.insert(exampleSockets, socket)
        return socket
    end
}
dofile("modules/network/websocket_example.lua")
package.loaded["modules.network.websocket"] = previousWebSocket
local exampleDataSocket = exampleSockets[5]
assert(exampleDataSocket and exampleDataSocket.options.onopen,
    "WebSocket example should define its data message callback")
local exampleStateMessage = exampleDataSocket.sent[1]
local exampleTrajectoryMessage = exampleDataSocket.sent[2]
assert(exampleStateMessage.type == "state")
local exampleState = StateManager.new()
exampleState:updateFromStateMessage(exampleStateMessage.data)
close(exampleState:getBallById(1).position.x, 960,
    "example ball point should scale from source image pixels")
close(exampleState:getTable().playing_area_corners[1].x, 384,
    "example table corner should scale from source image pixels")
close(exampleState:getTable().bounds.width, 3072,
    "example table boundary extent should scale from source image pixels")
close(exampleState:getTable().width, 3072,
    "explicit canonical table extents should retain their declared scale")
close(exampleState:getTable().real_world_dimensions.width, 2.1,
    "example physical table dimensions should remain in meters")
assert(exampleTrajectoryMessage.type == "trajectory")
Trajectory:updateTrajectory(exampleTrajectoryMessage.data)
close(Trajectory.paths[1].points[1].x, 960,
    "example trajectory points should scale from source image pixels")
close(Trajectory.collisions[1].position.y, 1080,
    "example collision point should scale from source image pixels")
equal(exampleStateMessage.data.coordinate_space, "pixel",
    "example should use the backend-supported source pixel space")
equal(exampleTrajectoryMessage.data.coordinate_space, "pixel",
    "trajectory example should use the backend-supported source pixel space")
equal(Trajectory.ghostBalls, nil,
    "WebSocket example should match the backend line/collision contract")

print("All visualizer coordinate tests passed")
