-- State Manager: Manages game state received from backend
-- All data comes from WebSocket messages - no direct camera or ball access

local StateManager = {}
StateManager.__index = StateManager

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
    if data.balls then
        self.balls = {}
        for _, ball in ipairs(data.balls) do
            self.balls[ball.id] = {
                id = ball.id,
                position = {x = ball.position[1], y = ball.position[2]},
                velocity = {x = ball.velocity[1], y = ball.velocity[2]},
                radius = ball.radius,
                is_moving = ball.is_moving,
                number = ball.number,
                is_cue_ball = ball.is_cue_ball,
                confidence = ball.confidence
            }
        end
    end

    if data.cue then
        self.cue = data.cue
    end

    if data.table then
        self.table = data.table
    end

    self.last_update = love.timer.getTime()
end

-- Update from immediate motion event
function StateManager:updateFromMotionMessage(data)
    local ball_id = data.ball_id
    if self.balls[ball_id] then
        self.balls[ball_id].position = {x = data.position[1], y = data.position[2]}
        self.balls[ball_id].velocity = {x = data.velocity[1], y = data.velocity[2]}
        self.balls[ball_id].is_moving = data.is_moving
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
