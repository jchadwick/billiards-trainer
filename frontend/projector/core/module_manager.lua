-- Module Manager
-- Handles dynamic loading, unloading, and management of modules

local ModuleManager = {
    modules = {},
    moduleOrder = {},
    moduleDirectory = "modules"
}

-- Load all modules from the modules directory
function ModuleManager:loadModules()
    self.modules = {}
    self.moduleOrder = {}

    -- Get list of directories in modules folder
    local modulePath = self.moduleDirectory
    local moduleList = love.filesystem.getDirectoryItems(modulePath)

    for _, moduleName in ipairs(moduleList) do
        local fullPath = modulePath .. "/" .. moduleName
        local info = love.filesystem.getInfo(fullPath)

        if info and info.type == "directory" then
            self:loadModule(moduleName)
        end
    end

    -- Sort modules by priority
    self:sortModules()

    print(string.format("Loaded %d modules", #self.moduleOrder))
end

-- Load a single module
function ModuleManager:loadModule(moduleName)
    local modulePath = self.moduleDirectory .. "/" .. moduleName .. "/init.lua"

    -- Check if module file exists
    if not love.filesystem.getInfo(modulePath) then
        print(string.format("Module %s missing init.lua", moduleName))
        return false
    end

    -- Load module
    local success, moduleOrError = pcall(function()
        -- Clear any cached version (use dots for require paths)
        local requirePath = self.moduleDirectory .. "." .. moduleName .. ".init"
        package.loaded[requirePath] = nil
        return require(requirePath)
    end)

    if not success then
        print(string.format("Failed to load module %s: %s", moduleName, moduleOrError))
        return false
    end

    local module = moduleOrError

    -- Validate module interface
    if type(module) ~= "table" then
        print(string.format("Module %s must return a table", moduleName))
        return false
    end

    -- Set default values
    module.name = module.name or moduleName
    module.priority = module.priority or 100
    module.enabled = module.enabled ~= false

    -- Initialize module
    if module.init and type(module.init) == "function" then
        local initSuccess, initError = pcall(function()
            module:init()
        end)

        if not initSuccess then
            print(string.format("Module %s initialization failed: %s", moduleName, initError))
            return false
        end
    end

    -- Store module
    self.modules[moduleName] = module
    table.insert(self.moduleOrder, module)

    print(string.format("Loaded module: %s (priority: %d)", module.name, module.priority))
    return true
end

-- Sort modules by priority (lower number = higher priority = drawn first)
function ModuleManager:sortModules()
    table.sort(self.moduleOrder, function(a, b)
        return a.priority < b.priority
    end)
end

-- Reload all modules
function ModuleManager:reloadModules()
    -- Cleanup existing modules
    self:cleanup()

    -- Reload all modules
    self:loadModules()
end

-- Update all modules
function ModuleManager:update(dt)
    for _, module in ipairs(self.moduleOrder) do
        if module.enabled and module.update and type(module.update) == "function" then
            local success, err = pcall(function()
                module:update(dt)
            end)

            if not success then
                print(string.format("Module %s update error: %s", module.name, err))
            end
        end
    end
end

-- Draw all modules
function ModuleManager:draw()
    for _, module in ipairs(self.moduleOrder) do
        if module.enabled and module.draw and type(module.draw) == "function" then
            local success, err = pcall(function()
                module:draw()
            end)

            if not success then
                print(string.format("Module %s draw error: %s", module.name, err))
            end
        end
    end
end

-- Broadcast an event to all modules
function ModuleManager:broadcast(eventName, ...)
    local args = {...}
    for _, module in ipairs(self.moduleOrder) do
        if module.enabled and module[eventName] and type(module[eventName]) == "function" then
            local success, err = pcall(function()
                module[eventName](module, unpack(args))
            end)

            if not success then
                print(string.format("Module %s event %s error: %s", module.name, eventName, err))
            end
        end
    end
end

-- Send a message to modules (typically from network)
function ModuleManager:sendMessage(messageType, data)
    self:broadcast("onMessage", messageType, data)
end

-- Cleanup all modules
function ModuleManager:cleanup()
    for _, module in ipairs(self.moduleOrder) do
        if module.cleanup and type(module.cleanup) == "function" then
            local success, err = pcall(function()
                module:cleanup()
            end)

            if not success then
                print(string.format("Module %s cleanup error: %s", module.name, err))
            end
        end
    end

    self.modules = {}
    self.moduleOrder = {}
end

-- Get module by name
function ModuleManager:getModule(name)
    return self.modules[name]
end

-- Enable/disable a module
function ModuleManager:setModuleEnabled(name, enabled)
    local module = self.modules[name]
    if module then
        module.enabled = enabled
        return true
    end
    return false
end

-- Get module count
function ModuleManager:getModuleCount()
    return #self.moduleOrder
end

-- List all modules
function ModuleManager:listModules()
    local list = {}
    for _, module in ipairs(self.moduleOrder) do
        table.insert(list, {
            name = module.name,
            priority = module.priority,
            enabled = module.enabled
        })
    end
    return list
end

return ModuleManager
