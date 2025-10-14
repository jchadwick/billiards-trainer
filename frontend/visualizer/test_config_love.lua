-- Minimal LOVE2D test for config system
local Config = require('core.config')

function love.load()
    print("\n=== Configuration System Test (LOVE2D) ===\n")

    local config = Config.get_instance()

    -- Test basic get
    local width = config:get('display.width')
    local height = config:get('display.height')
    print(string.format("Display: %dx%d", width, height))

    -- Test nested get
    local ws_url = config:get('network.websocket_url')
    print(string.format("WebSocket URL: %s", ws_url))

    -- Test set
    local ok = config:set('display.width', 1920)
    print(string.format("Set display.width: %s (new value: %d)", tostring(ok), config:get('display.width')))

    -- Test type validation
    local ok2, err = config:set('display.width', 'invalid')
    print(string.format("Type validation: %s", ok2 and "FAILED" or "PASSED"))

    print("\n=== Test Complete ===\n")

    -- Exit after test
    love.event.quit()
end

function love.draw()
    love.graphics.print("Running config tests...", 10, 10)
end
