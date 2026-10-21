-- State Manager: Manages game state received from backend
-- All data comes from WebSocket messages - no direct camera or ball access

local StateManager = {}
StateManager.__index = StateManager
local Coordinates = require("core.coordinates")

local function normalize_record(value, scale, coordinateSpace)
    if type(value) ~= "table" then return nil end
    local normalized = Coordinates.normalizeGeometry(value, scale, coordinateSpace)
    return normalized
end

function StateManager.new()
    local self = setmetatable({}, StateManager)
    self.balls = {}  -- Indexed by ball_id
    self.cue = nil
    self.table = nil
    self.last_update = 0
    self.sequence_number = 0
    return self
end

-- Update from periodic state message (every 500ms)
function StateManager:updateFromStateMessage(data)
    if type(data) ~= "table" then return end
    if type(data.balls) == "table" then
        self.balls = {}
        for _, ball in ipairs(data.balls) do
            if type(ball) == "table" and ball.id ~= nil then
                local normalized = normalize_record(
                    ball, ball.scale,
                    ball.coordinate_space or data.coordinate_space
                )
                if normalized and type(normalized.position) == "table"
                    and type(normalized.velocity) == "table" then
                    self.balls[ball.id] = normalized
                end
            end
        end
    end

    if data.cue ~= nil then
        self.cue = normalize_record(data.cue, data.scale, data.coordinate_space)
    end

    if data.table ~= nil then
        self.table = normalize_record(data.table, data.scale, data.coordinate_space)
    end

    self.last_update = love.timer.getTime()
end

-- Update from immediate motion event
function StateManager:updateFromMotionMessage(data)
    if type(data) ~= "table" then return end
    local ball_id = data.ball_id
    if self.balls[ball_id] then
        local normalized = normalize_record(
            data, data.scale, data.coordinate_space
        )
        if normalized and type(normalized.position) == "table"
            and type(normalized.velocity) == "table" then
            self.balls[ball_id].position = normalized.position
            self.balls[ball_id].velocity = normalized.velocity
            self.balls[ball_id].is_moving = data.is_moving
        end
    end
    self.last_update = love.timer.getTime()
end

function StateManager:getBalls()
    return self.balls
end

function StateManager:getBallById(id)
    return self.balls[id]
end

function StateManager:getCueBall()
    for _, ball in pairs(self.balls) do
        if ball.is_cue_ball then
            return ball
        end
    end
    return nil
end

function StateManager:getCue()
    return self.cue
end

function StateManager:getTable()
    return self.table
end

function StateManager:setSequenceNumber(seq)
    self.sequence_number = seq
end

function StateManager:getSequenceNumber()
    return self.sequence_number
end

function StateManager:getLastUpdateTime()
    return self.last_update
end

return StateManager
