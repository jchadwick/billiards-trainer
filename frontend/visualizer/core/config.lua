-- Configuration management module for the visualizer
-- Loads default.json and provides get/set methods with validation

local json = require("lib.json")

local Config = {}
Config.__index = Config

-- Configuration schema for validation
local CONFIG_SCHEMA = {
    display = {
        width = "number",
        height = "number",
        fullscreen = "boolean",
        vsync = "boolean"
    },
    network = {
        websocket_url = "string",
        auto_connect = "boolean",
        reconnect_enabled = "boolean",
        reconnect_delay = "number",
        max_reconnect_delay = "number",
        heartbeat_interval = "number"
    },
    calibration = {
        profile = "string",
        auto_load = "boolean"
    },
    rendering = {
        fps_target = "number",
        line_thickness = "number",
        anti_aliasing = "boolean",
        theme = "string"
    },
    video_feed = {
        enabled = "boolean",
        opacity = "number",
        layer = "string",
        subscribe_on_start = "boolean",
        quality = "number",
        fps = "number"
    },
    debug_hud = {
        enabled = "boolean",
        position = "string",
        opacity = "number",
        font_size = "number",
        color = "table",
        background = "table",
        sections = {
            connection = "boolean",
            balls = "boolean",
            cue = "boolean",
            table = "boolean",
            performance = "boolean"
        },
        layout = "string",
        update_rate = "number"
    },
    colors = {
        table_felt = "table",
        auto_adapt = "boolean",
        trajectory_primary = "table_or_nil",
        trajectory_secondary = "table_or_nil"
    }
}

-- Create a new Config instance
function Config.new()
    local self = setmetatable({}, Config)
    self.config = {}
    self.default_config = {}
    self.user_overrides = {}
    self.initialized = false
    return self
end

-- Deep copy a table
local function deep_copy(obj, seen)
    if type(obj) ~= 'table' then return obj end
    if seen and seen[obj] then return seen[obj] end

    local s = seen or {}
    local res = setmetatable({}, getmetatable(obj))
    s[obj] = res

    for k, v in pairs(obj) do
        res[deep_copy(k, s)] = deep_copy(v, s)
    end

    return res
end

-- Deep merge two tables (source into target)
local function deep_merge(target, source)
    for k, v in pairs(source) do
        if type(v) == "table" and type(target[k]) == "table" then
            deep_merge(target[k], v)
        else
            target[k] = v
        end
    end
    return target
end

-- Validate a value against a schema type
local function validate_type(value, expected_type, path)
    if expected_type == "table_or_nil" then
        return value == nil or type(value) == "table"
    end

    if expected_type == "table" then
        return type(value) == "table"
    end

    return type(value) == expected_type
end

-- Recursively validate configuration against schema
local function validate_config(config, schema, path)
    path = path or "config"

    for key, expected in pairs(schema) do
        local value = config[key]
        local current_path = path .. "." .. key

        if value == nil and expected ~= "table_or_nil" then
            return false, string.format("Missing required key: %s", current_path)
        end

        if type(expected) == "table" then
            if value ~= nil and type(value) ~= "table" then
                return false, string.format("Expected table at %s, got %s", current_path, type(value))
            end
            if value ~= nil then
                local ok, err = validate_config(value, expected, current_path)
                if not ok then
                    return false, err
                end
            end
        else
            if value ~= nil and not validate_type(value, expected, current_path) then
                return false, string.format("Type mismatch at %s: expected %s, got %s",
                    current_path, expected, type(value))
            end
        end
    end

    return true
end

-- Load JSON file using LOVE2D filesystem API
local function load_json_file(filepath)
    -- Check if file exists
    local info = love.filesystem.getInfo(filepath)
    if not info then
        return nil, string.format("Could not find file: %s", filepath)
    end

    -- Read file content
    local content, err = love.filesystem.read(filepath)
    if not content then
        return nil, string.format("Could not read file: %s (%s)", filepath, err or "unknown error")
    end

    -- Parse JSON
    local ok, decoded = pcall(json.decode, content)
    if not ok then
        return nil, string.format("Failed to parse JSON: %s", decoded)
    end

    return decoded
end

-- Initialize the configuration system
function Config:init()
    if self.initialized then
        return true
    end

    -- Load default configuration
    local default_path = "config/default.json"
    local defaults, err = load_json_file(default_path)
    if not defaults then
        return false, string.format("Failed to load default config: %s", err)
    end

    -- Validate default configuration
    local ok, validation_err = validate_config(defaults, CONFIG_SCHEMA)
    if not ok then
        return false, string.format("Invalid default configuration: %s", validation_err)
    end

    self.default_config = defaults
    self.config = deep_copy(defaults)

    -- Try to load user overrides (optional)
    local user_path = "config/user.json"
    local overrides, _ = load_json_file(user_path)
    if overrides then
        -- Validate user overrides before merging
        local override_ok, override_err = validate_config(overrides, CONFIG_SCHEMA)
        if override_ok then
            self.user_overrides = overrides
            deep_merge(self.config, overrides)
        else
            print(string.format("Warning: Invalid user config, ignoring: %s", override_err))
        end
    end

    self.initialized = true
    return true
end

-- Get a configuration value by path (e.g., "display.width")
function Config:get(path)
    if not self.initialized then
        local ok, err = self:init()
        if not ok then
            error(err)
        end
    end

    local parts = {}
    for part in string.gmatch(path, "[^.]+") do
        table.insert(parts, part)
    end

    local current = self.config
    for _, part in ipairs(parts) do
        if type(current) ~= "table" then
            return nil
        end
        current = current[part]
        if current == nil then
            return nil
        end
    end

    return current
end

-- Set a configuration value by path (e.g., "display.width", 1920)
function Config:set(path, value)
    if not self.initialized then
        local ok, err = self:init()
        if not ok then
            error(err)
        end
    end

    local parts = {}
    for part in string.gmatch(path, "[^.]+") do
        table.insert(parts, part)
    end

    if #parts == 0 then
        return false, "Invalid path"
    end

    -- Navigate to the parent table
    local current = self.config
    for i = 1, #parts - 1 do
        local part = parts[i]
        if type(current[part]) ~= "table" then
            return false, string.format("Invalid path: %s is not a table", part)
        end
        current = current[part]
    end

    -- Set the value
    local key = parts[#parts]

    -- Validate the new value against the schema
    local schema_current = CONFIG_SCHEMA
    for i = 1, #parts - 1 do
        schema_current = schema_current[parts[i]]
        if not schema_current then
            return false, "Path not in schema"
        end
    end

    local expected_type = schema_current[key]
    if not expected_type then
        return false, "Key not in schema"
    end

    if not validate_type(value, expected_type, path) then
        return false, string.format("Type mismatch: expected %s, got %s", expected_type, type(value))
    end

    current[key] = value
    return true
end

-- Get the entire configuration table
function Config:get_all()
    if not self.initialized then
        local ok, err = self:init()
        if not ok then
            error(err)
        end
    end

    return deep_copy(self.config)
end

-- Reset configuration to defaults
function Config:reset()
    if not self.initialized then
        local ok, err = self:init()
        if not ok then
            error(err)
        end
    end

    self.config = deep_copy(self.default_config)
    -- Reapply user overrides
    if next(self.user_overrides) ~= nil then
        deep_merge(self.config, self.user_overrides)
    end
    return true
end

-- Save current configuration to user.json
function Config:save_user_config()
    if not self.initialized then
        return false, "Config not initialized"
    end

    -- Calculate differences from default
    local differences = {}

    local function find_differences(current, default, diff, path)
        for key, value in pairs(current) do
            local default_value = default[key]
            if type(value) == "table" and type(default_value) == "table" then
                diff[key] = diff[key] or {}
                find_differences(value, default_value, diff[key], path .. "." .. key)
                if next(diff[key]) == nil then
                    diff[key] = nil
                end
            elseif value ~= default_value then
                diff[key] = value
            end
        end
    end

    find_differences(self.config, self.default_config, differences, "config")

    -- Only save if there are differences
    if next(differences) == nil then
        return true  -- Nothing to save
    end

    local encoded = json.encode(differences)

    -- Use LOVE2D filesystem API for writing
    local success, err = love.filesystem.write("config/user.json", encoded)
    if not success then
        return false, string.format("Could not write user config: %s", err or "unknown error")
    end

    return true
end

-- Global instance
local config_instance = nil

-- Get the global config instance
function Config.get_instance()
    if not config_instance then
        config_instance = Config.new()
        local ok, err = config_instance:init()
        if not ok then
            error(string.format("Failed to initialize config: %s", err))
        end
    end
    return config_instance
end

return Config
