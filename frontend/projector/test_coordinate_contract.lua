-- Hardware-free coordinate contract tests. Run with:
-- lua frontend/projector/test_coordinate_contract.lua

local projectRoot = "frontend/projector/"
package.path = projectRoot .. "?.lua;" .. projectRoot .. "?/init.lua;" .. package.path

local CoordinateContract = require("core.coordinate_contract")
local Trajectory = require("modules.trajectory.init")

love = {
    timer = {
        getTime = function() return 42 end
    },
    graphics = {
        getDimensions = function() return 1920, 1080 end
    },
    math = {
        newTransform = function()
            return {
                translate = function() end,
                scale = function() end
            }
        end
    }
}

local function assertValid(valid, errorMessage)
    assert(valid, errorMessage)
end

local function assertInvalid(valid, errorMessage)
    assert(not valid, "expected coordinate payload to be rejected")
    assert(type(errorMessage) == "string" and #errorMessage > 0)
end

local function tableMetadata()
    return {
        coordinate_space = "normalized_table",
        coordinate_scale = {1, 1}
    }
end

local function validTrajectory()
    local data = tableMetadata()
    data.table_boundary = {
        {x = 0, y = 0},
        {x = 1, y = 0},
        {x = 1, y = 1},
        {x = 0, y = 1}
    }
    data.bounds = {x = 0, y = 0, width = 1, height = 1}
    data.paths = {{points = {{x = 0.2, y = 0.3}, {x = 0.8, y = 0.7}}}}
    data.collisions = {{x = 0.5, y = 0.5, type = "cushion"}}
    data.ghostBalls = {{
        x = 0.7,
        y = 0.4,
        radius = 15,
        radius_space = "projector_pixels",
        radius_scale = {1, 1}
    }}
    return data
end

assertValid(CoordinateContract.validateTrajectory(validTrajectory()))
assertValid(CoordinateContract.validateBounds({x = 0.1, y = 0.2, width = 0.5, height = 0.6}))
assertValid(CoordinateContract.validateProjectorCorners({
    {x = 0.1, y = 0.1},
    {x = 0.9, y = 0.1},
    {x = 0.9, y = 0.9},
    {x = 0.1, y = 0.9}
}, "normalized_projector", {1, 1}))

local invalid = validTrajectory()
invalid.coordinate_scale = nil
assertInvalid(CoordinateContract.validateTrajectory(invalid))

invalid = validTrajectory()
invalid.coordinate_space = "pixels"
assertInvalid(CoordinateContract.validateTrajectory(invalid))

invalid = validTrajectory()
invalid.coordinate_scale = {0.5, 0.5}
assertInvalid(CoordinateContract.validateTrajectory(invalid))

invalid = validTrajectory()
invalid.coordinate_scale = {1, 1, 1}
assertInvalid(CoordinateContract.validateTrajectory(invalid))

invalid = validTrajectory()
invalid.paths[1].points[1].scale = {2, 2}
assertInvalid(CoordinateContract.validateTrajectory(invalid))

invalid = validTrajectory()
invalid.table_boundary[1].scale = {1, 1}
assertInvalid(CoordinateContract.validateTrajectory(invalid))

invalid = validTrajectory()
invalid.collisions[1].scale = {1, 1}
assertInvalid(CoordinateContract.validateTrajectory(invalid))

invalid = validTrajectory()
invalid.bounds.scale = {1, 1}
assertInvalid(CoordinateContract.validateTrajectory(invalid))

for _, invalidValue in ipairs({-0.01, 1.01, math.huge, -math.huge, 0 / 0, "0.5"}) do
    invalid = validTrajectory()
    invalid.paths[1].points[2].x = invalidValue
    assertInvalid(CoordinateContract.validateTrajectory(invalid))
end

invalid = validTrajectory()
invalid.table_boundary[3].y = 1.01
assertInvalid(CoordinateContract.validateTrajectory(invalid))

invalid = validTrajectory()
invalid.table_boundary[4] = nil
assertInvalid(CoordinateContract.validateTrajectory(invalid))

invalid = validTrajectory()
invalid.bounds.width = 1.1
assertInvalid(CoordinateContract.validateTrajectory(invalid))

invalid = validTrajectory()
invalid.bounds.height = 0
assertInvalid(CoordinateContract.validateTrajectory(invalid))

invalid = validTrajectory()
invalid.bounds.unit = "meters"
assertInvalid(CoordinateContract.validateTrajectory(invalid))

invalid = validTrajectory()
invalid.ghostBalls[1].radius_space = nil
assertInvalid(CoordinateContract.validateTrajectory(invalid))

invalid = validTrajectory()
invalid.ghostBalls[1].radius_scale = {2, 2}
assertInvalid(CoordinateContract.validateTrajectory(invalid))

invalid = validTrajectory()
invalid.ghostBalls[1].radius = -1
assertInvalid(CoordinateContract.validateTrajectory(invalid))

invalid = validTrajectory()
invalid.ghostBalls[1].radius = 0 / 0
assertInvalid(CoordinateContract.validateTrajectory(invalid))

local collision = tableMetadata()
collision.x = 0.3
collision.y = 0.4
assertValid(CoordinateContract.validateCollision(collision))
collision.x = -1
assertInvalid(CoordinateContract.validateCollision(collision))

local aim = tableMetadata()
aim.x1, aim.y1, aim.x2, aim.y2 = 0.2, 0.3, 0.8, 0.7
assertValid(CoordinateContract.validateAim(aim))
aim.x2 = 1.2
assertInvalid(CoordinateContract.validateAim(aim))

local state = tableMetadata()
state.balls = {{x = 0.4, y = 0.6, radius = 12, radius_space = "projector_pixels", radius_scale = {1, 1}}}
state.cue = {tip_x = 0.2, tip_y = 0.5}
state.table = {
    corners = {
        {x = 0, y = 0}, {x = 1, y = 0}, {x = 1, y = 1}, {x = 0, y = 1}
    },
    playing_area = {x = 0, y = 0, width = 1, height = 1}
}
assertValid(CoordinateContract.validateState(state))
state.balls[1].radius_scale = {0.5, 0.5}
assertInvalid(CoordinateContract.validateState(state))
state.balls[1].radius_scale = {1, 1}
state.table.corners[4].x = -0.1
assertInvalid(CoordinateContract.validateMessage("state", state))
assertValid(CoordinateContract.validateState({fps = 60}))

local nestedState = tableMetadata()
nestedState.balls = {{
    id = "cue",
    position = {x = 0.4, y = 0.6},
    radius = 14,
    radius_space = "projector_pixels",
    radius_scale = {1, 1},
    velocity = {
        x = 1.5,
        y = -0.25,
        velocity_space = "normalized_table_per_second",
        velocity_scale = {1, 1}
    }
}}
nestedState.cue = {position = {x = 0.2, y = 0.5}, tip = {x = 0.3, y = 0.5}}
nestedState.table = {
    corners = {{x = 0, y = 0}, {x = 1, y = 0}, {x = 1, y = 1}, {x = 0, y = 1}},
    pockets = {
        {center = {x = 0, y = 0}},
        {center = {x = -0.02, y = 0.01, unit = "meters"}, radius = 0.03, radius_unit = "meters"}
    },
    playing_area = {x = 0.05, y = 0.05, width = 0.9, height = 0.9},
    real_world_dimensions = {width = 2.54, height = 1.27, unit = "meters"}
}
nestedState.room_reference = {
    x = -0.3,
    y = 1.4,
    unit = "meters",
    coordinate_space = "world_meters"
}
assertValid(CoordinateContract.validateState(nestedState))

invalid = {
    coordinate_space = "normalized_table",
    coordinate_scale = {1, 1},
    balls = {{position = {x = 0.4, y = 0.6, unit = "meters", coordinate_space = "world_meters"}}}
}
assertValid(CoordinateContract.validateState(invalid))
invalid = {
    coordinate_space = "normalized_table",
    coordinate_scale = {1, 1},
    balls = {{position = {x = 0.4, y = 0.6, coordinate_space = "4k_canonical", coordinate_scale = {1, 1}}}}
}
assertInvalid(CoordinateContract.validateState(invalid))
invalid = tableMetadata()
invalid.table = {real_world_dimensions = {width = 2.54, height = 1.27}}
assertInvalid(CoordinateContract.validateState(invalid))
invalid = tableMetadata()
invalid.table = {real_world_dimensions = {width = 2.54, height = 1.27, unit = "meters", extent_scale = {1, 1}}}
assertInvalid(CoordinateContract.validateState(invalid))
invalid = tableMetadata()
invalid.balls = {{position = {x = 0.4, y = 0.6, unit = "meters", coordinate_space = "normalized_table"}}}
assertInvalid(CoordinateContract.validateState(invalid))
invalid = tableMetadata()
invalid.balls = {{position = {x = 0.4, y = 0.6, unit = "pixels"}}}
assertInvalid(CoordinateContract.validateState(invalid))
invalid = tableMetadata()
invalid.table = {corners = {
    {x = 0.1, y = 0.1, unit = "meters"}, {x = 1, y = 0}, {x = 1, y = 1}, {x = 0, y = 1}
}}
assertInvalid(CoordinateContract.validateState(invalid))
invalid = tableMetadata()
invalid.balls = {{position = {x = 0.4, y = 0.6, coordinate_space = "normalized_table"}}}
assertInvalid(CoordinateContract.validateState(invalid))
invalid = tableMetadata()
invalid.table = {playing_area = {x = 0, y = 0, width = 1.1, height = 0.5}}
assertInvalid(CoordinateContract.validateState(invalid))
invalid = tableMetadata()
invalid.table = {corners = {{x = 0, y = 0}, {x = 1, y = 0}, {x = 1, y = 1}, {x = 0, y = 1}}, width = 2, height = 1}
assertInvalid(CoordinateContract.validateState(invalid))

local motion = tableMetadata()
motion.position = {x = 0.25, y = 0.75}
motion.velocity = {
    x = -1.5,
    y = 0.25,
    velocity_space = "normalized_table_per_second",
    velocity_scale = {1, 1}
}
assertValid(CoordinateContract.validateMotion(motion))
invalid = tableMetadata()
invalid.position = {x = 0.25, y = 0.75}
invalid.velocity = {x = 1.5, y = 0.25, velocity_space = "normalized_table_per_second"}
assertInvalid(CoordinateContract.validateMotion(invalid))
invalid = motion
invalid = {
    coordinate_space = "normalized_table",
    position = {x = 0.25, y = 0.75},
    velocity = {x = 1.5, y = 0.25, velocity_space = "normalized_table_per_second", velocity_scale = {1, 1}}
}
assertInvalid(CoordinateContract.validateMessage("motion", invalid))

local event = tableMetadata()
event.name = "ball_contact"
event.position = {x = 0.7, y = 0.2}
event.cushion = {position = {x = 0.9, y = 0.5}}
assertValid(CoordinateContract.validateMessage("event", event))
event.cushion.position.x = 1.1
assertInvalid(CoordinateContract.validateMessage("event", event))
assertValid(CoordinateContract.validateMessage("event", {name = "timeout", duration = 0.5}))
assertInvalid(CoordinateContract.validateMessage("vendor_event", {position = {x = 0.2, y = 0.3}}))

assertValid(CoordinateContract.validateMessage("show_calibration_chessboard", {
    coordinate_space = "projector_pixels",
    coordinate_scale = {1, 1},
    squareSize = 80,
    offsetX = 15,
    offsetY = -20
}))
assertInvalid(CoordinateContract.validateMessage("show_calibration_chessboard", {squareSize = 80}))
assertInvalid(CoordinateContract.validateMessage("show_calibration_chessboard", {
    coordinate_space = "projector_pixels",
    coordinate_scale = {2, 2},
    squareSize = 80
}))
assertInvalid(CoordinateContract.validateMessage("show_calibration_chessboard", {
    coordinate_space = "projector_pixels",
    coordinate_scale = {1, 1},
    offsetX = math.huge
}))
assertInvalid(CoordinateContract.validateMessage("show_calibration_chessboard", {
    coordinate_space = "projector_pixels",
    coordinate_scale = {1, 1},
    position = {x = 10, y = 20}
}))
assertValid(CoordinateContract.validateMessage("hide_calibration_chessboard", {}))
assertInvalid(CoordinateContract.validateMessage("hide_calibration_chessboard", {x = 10, y = 20}))
assertValid(CoordinateContract.validateMessage("frame", {fps = 30, latency = 4}))
assertValid(CoordinateContract.validateMessage("frame", {
    width = 1920,
    height = 1080,
    dimensions_space = "image_pixels",
    dimensions_scale = {1, 1}
}))
assertInvalid(CoordinateContract.validateMessage("frame", {width = 1920, height = 1080}))
assertInvalid(CoordinateContract.validateMessage("frame", {
    width = 1920,
    height = 1080,
    dimensions_space = "image_pixels",
    dimensions_scale = {1, 1},
    position = {x = 10, y = 20}
}))

local badCorners = {
    {x = 0, y = 0}, {x = 1, y = 0}, {x = 1, y = 1}, {x = 0, y = math.huge}
}
assertInvalid(CoordinateContract.validateProjectorCorners(badCorners, "normalized_projector", {1, 1}))
assertInvalid(CoordinateContract.validateProjectorCorners(badCorners, "normalized_projector", {0.5, 0.5}))

local Calibration = require("core.calibration")
local savedCalibration
local calibrationData = {
    coordinate_space = "normalized_projector",
    coordinate_scale = {1, 1},
    corners = {
        {x = 0.05, y = 0.05},
        {x = 0.95, y = 0.05},
        {x = 0.95, y = 0.95},
        {x = 0.05, y = 0.95}
    }
}
package.loaded["lib.json"] = {
    decode = function() return calibrationData end,
    encode = function(value)
        savedCalibration = value
        return "calibration"
    end
}
love.filesystem = {
    getInfo = function() return {type = "file"} end,
    read = function() return "calibration" end,
    write = function() return true end
}
assert(Calibration:load())
assert(Calibration.corners[1].x == 0.05)
calibrationData.coordinate_scale = {0.5, 0.5}
assert(not Calibration:load())
assert(Calibration.corners[1].x == 0.1)
Calibration:dragCorner(-10, -10)
assert(Calibration.corners[1].x == 0 and Calibration.corners[1].y == 0)
assert(Calibration:save())
assert(savedCalibration.coordinate_space == "normalized_projector")
assert(savedCalibration.coordinate_scale[1] == 1 and savedCalibration.coordinate_scale[2] == 1)
Calibration.corners[1].x = math.huge
assert(not Calibration:save())
assert(not Calibration:calculateTransform())
Calibration:reset()
assert(Calibration:calculateTransform())
local beforeDragX = Calibration:transform(0.5, 0.5)
assert(Calibration:dragCorner(384, 108))
local afterDragX = Calibration:transform(0.5, 0.5)
assert(afterDragX > beforeDragX)

-- The module revalidates direct calls and leaves its last good frame intact.
local initial = validTrajectory()
assert(Trajectory:updateTrajectory(initial))
local retainedPath = Trajectory.paths[1]
invalid = validTrajectory()
invalid.paths[1].points[1].x = -0.1
local accepted = Trajectory:updateTrajectory(invalid)
assert(not accepted)
assert(Trajectory.paths[1] == retainedPath)

local priorCollisionCount = #Trajectory.collisions
invalid = tableMetadata()
invalid.x, invalid.y = 0.2, 1.5
assert(not Trajectory:addCollision(invalid))
assert(#Trajectory.collisions == priorCollisionCount)

-- Network admission validation rejects malformed coordinates before queuing.
package.preload["socket"] = function() return {} end
package.preload["lib.json"] = function() return {} end
local Network = require("core.network")
assert(Network.websocketUrl == nil)
assert(Network:getStatus().websocket == nil)
Network.status.messagesReceived = 0
Network.messageQueue = {}
assert(Network:processMessage({type = "trajectory", data = validTrajectory()}, "127.0.0.1", 1))
assert(#Network.messageQueue == 1)
assert(Network.status.messagesReceived == 1)
invalid = validTrajectory()
invalid.coordinate_space = nil
assert(not Network:processMessage({type = "trajectory", data = invalid}, "127.0.0.1", 1))
assert(#Network.messageQueue == 1)
assert(Network.status.messagesReceived == 1)
local stateCallbackInvoked = false
Network:registerCallback("state", function()
    stateCallbackInvoked = true
end)
assert(not Network:processMessage({type = "state", data = {
    coordinate_space = "normalized_table",
    coordinate_scale = {1, 1},
    balls = {{position = {x = 0.2, y = 1.2}}}
}}, "127.0.0.1", 1))
assert(#Network.messageQueue == 1)
assert(Network.status.messagesReceived == 1)
assert(not stateCallbackInvoked)

print("Projector coordinate contract tests passed")
