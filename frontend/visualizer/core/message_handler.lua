-- Message Handler: Parses and routes WebSocket messages
-- Routes different message types to appropriate handlers

local MessageHandler = {}
MessageHandler.__index = MessageHandler

function MessageHandler.new(state_manager)
    local self = setmetatable({}, MessageHandler)
    self.state_manager = state_manager
    self.trajectory_callback = nil
    self.frame_callback = nil
    self.alert_callback = nil
    self.config_callback = nil
    self.last_sequence = 0
    self.sequence_gaps = 0
    return self
end

-- Main entry point: parse JSON and route
function MessageHandler:handleMessage(json_string)
    local json = require("lib.json")
    local success, message = pcall(json.decode, json_string)

    if not success then
        print("Error parsing JSON: " .. tostring(message))
        return
    end

    if not message.type then
        print("Message missing type field")
        return
    end

    -- Check sequence number
    if message.sequence then
        self:checkSequence(message.sequence)
    end

    -- Route based on message type
    if message.type == "state" then
        self:handleStateMessage(message.data)
    elseif message.type == "motion" then
        self:handleMotionMessage(message.data)
    elseif message.type == "trajectory" then
        self:handleTrajectoryMessage(message.data)
    elseif message.type == "frame" then
        self:handleFrameMessage(message.data)
    elseif message.type == "alert" then
        self:handleAlertMessage(message.data)
    elseif message.type == "config" then
        self:handleConfigMessage(message.data)
    else
        print("Unknown message type: " .. message.type)
    end
end

function MessageHandler:handleStateMessage(data)
    self.state_manager:updateFromStateMessage(data)
end

function MessageHandler:handleMotionMessage(data)
    self.state_manager:updateFromMotionMessage(data)
end

function MessageHandler:handleTrajectoryMessage(data)
    if self.trajectory_callback then
        self.trajectory_callback(data)
    end
end

function MessageHandler:handleFrameMessage(data)
    if self.frame_callback then
        self.frame_callback(data)
    end
end

function MessageHandler:handleAlertMessage(data)
    if self.alert_callback then
        self.alert_callback(data)
    end
end

function MessageHandler:handleConfigMessage(data)
    if self.config_callback then
        self.config_callback(data)
    end
end

function MessageHandler:checkSequence(sequence)
    if self.last_sequence > 0 and sequence > self.last_sequence + 1 then
        local gap = sequence - self.last_sequence - 1
        self.sequence_gaps = self.sequence_gaps + gap
        print(string.format("Sequence gap detected: expected %d, got %d (gap: %d)",
            self.last_sequence + 1, sequence, gap))
    end
    self.last_sequence = sequence
    self.state_manager:setSequenceNumber(sequence)
end

function MessageHandler:setTrajectoryCallback(callback)
    self.trajectory_callback = callback
end

function MessageHandler:setFrameCallback(callback)
    self.frame_callback = callback
end

function MessageHandler:setAlertCallback(callback)
    self.alert_callback = callback
end

function MessageHandler:setConfigCallback(callback)
    self.config_callback = callback
end

function MessageHandler:getSequenceGaps()
    return self.sequence_gaps
end

return MessageHandler
