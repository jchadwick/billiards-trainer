-- Test script to validate module initialization
-- Run with: cd frontend/visualizer && love . --test-init
-- This will be executed as a replacement for main.lua when testing

-- Test module loading
local function test_module(name, path)
    io.write(string.format("Testing: %s ... ", name))
    io.flush()
    local success, result = pcall(require, path)
    if success then
        print("\27[32m✓ PASS\27[0m")
        return true, result
    else
        print(string.format("\27[31m✗ FAIL\27[0m\n  Error: %s", tostring(result)))
        return false, result
    end
end

-- LOVE2D callback to run tests
function love.load()
    print("=== Visualizer Module Initialization Test ===")
    print("Running in LOVE2D environment")
    print()

    -- Test order matches main.lua load sequence
    local tests = {
        {"Config", "core.config"},
        {"Renderer", "core.renderer"},
        {"Calibration", "rendering.calibration"},
        {"Colors", "modules.colors.init"},
        {"StateManager", "core.state_manager"},
        {"MessageHandler", "core.message_handler"},
        {"Trajectory", "modules.trajectory.init"},
        {"Network", "modules.network.init"}
    }

    local passed = 0
    local failed = 0
    local results = {}

    for i, test in ipairs(tests) do
        local success, result = test_module(test[1], test[2])
        if success then
            passed = passed + 1
            table.insert(results, {name = test[1], status = "PASS", module = result})
        else
            failed = failed + 1
            table.insert(results, {name = test[1], status = "FAIL", error = result})
        end
    end

    print()
    print("=== Test Results ===")
    print(string.format("Passed: %d/%d", passed, #tests))
    print(string.format("Failed: %d/%d", failed, #tests))
    print()

    if failed == 0 then
        print("\27[32m✓ All modules can be loaded successfully!\27[0m")
        print()
        print("Testing initialization sequence:")

        -- Try to initialize key modules like main.lua does
        local init_success, init_err = pcall(function()
            -- Initialize Config
            _G.Config = require("core.config").get_instance()
            _G.Config:init()
            print("  ✓ Config initialized")

            -- Initialize Renderer
            local RendererModule = require("core.renderer")
            _G.Renderer = RendererModule
            _G.Renderer:init()
            print("  ✓ Renderer initialized")

            -- Load Calibration
            _G.Calibration = require("rendering.calibration")
            _G.Calibration:load()
            print("  ✓ Calibration loaded")

            -- Initialize Colors
            local Colors = require("modules.colors.init")
            _G.Colors = Colors
            Colors:init()
            print("  ✓ Colors initialized")

            -- Create StateManager
            local StateManager = require("core.state_manager")
            local stateManager = StateManager.new()
            print("  ✓ StateManager created")

            -- Create MessageHandler
            local MessageHandler = require("core.message_handler")
            local messageHandler = MessageHandler.new(stateManager)
            print("  ✓ MessageHandler created")

            -- Load Trajectory module
            local trajectoryModule = require("modules.trajectory.init")
            print("  ✓ Trajectory module loaded")

            -- Initialize Network
            local Network = require("modules.network.init")
            Network:init()
            print("  ✓ Network initialized")
        end)

        print()
        if init_success then
            print("\27[32m✓ All modules initialized successfully!\27[0m")
            os.exit(0)
        else
            print("\27[31m✗ Initialization failed:\27[0m")
            print("  " .. tostring(init_err))
            os.exit(1)
        end
    else
        print("\27[31m✗ Some modules failed to load. Check errors above.\27[0m")
        os.exit(1)
    end
end

-- Minimal LOVE2D callbacks to keep it running
function love.update(dt)
    -- Nothing needed for tests
end

function love.draw()
    -- Nothing needed for tests
end
