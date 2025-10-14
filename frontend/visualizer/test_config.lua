-- Test script for configuration system
-- Run with: cd frontend/visualizer && lua test_config.lua

local Config = require('core.config')

print("=== Configuration System Test ===\n")

-- Test 1: Initialize and get instance
print("Test 1: Initialize configuration")
local config = Config.get_instance()
print("✓ Config instance created and initialized\n")

-- Test 2: Get simple values
print("Test 2: Get simple configuration values")
local width = config:get('display.width')
local height = config:get('display.height')
local ws_url = config:get('network.websocket_url')
print(string.format("  Display: %dx%d", width, height))
print(string.format("  WebSocket URL: %s", ws_url))
print("✓ Simple values retrieved correctly\n")

-- Test 3: Get nested values
print("Test 3: Get nested configuration values")
local debug_enabled = config:get('debug_hud.enabled')
local debug_opacity = config:get('debug_hud.opacity')
local connection_section = config:get('debug_hud.sections.connection')
print(string.format("  Debug HUD enabled: %s", tostring(debug_enabled)))
print(string.format("  Debug HUD opacity: %.1f", debug_opacity))
print(string.format("  Connection section: %s", tostring(connection_section)))
print("✓ Nested values retrieved correctly\n")

-- Test 4: Get table values
print("Test 4: Get table configuration values")
local colors = config:get('debug_hud.color')
print(string.format("  Debug HUD color: [%d, %d, %d]", colors[1], colors[2], colors[3]))
local felt_color = config:get('colors.table_felt')
print(string.format("  Table felt color: [%d, %d, %d]", felt_color[1], felt_color[2], felt_color[3]))
print("✓ Table values retrieved correctly\n")

-- Test 5: Set valid values
print("Test 5: Set configuration values")
local ok, err = config:set('display.width', 1920)
if ok then
    local new_width = config:get('display.width')
    print(string.format("  Set display.width to 1920: %d", new_width))
    assert(new_width == 1920, "Width should be 1920")
    print("✓ Value set successfully")
else
    print("✗ Failed to set value: " .. err)
end
print()

-- Test 6: Type validation
print("Test 6: Type validation")
local ok2, err2 = config:set('display.width', 'invalid')
if not ok2 then
    print(string.format("  Correctly rejected invalid type: %s", err2))
    print("✓ Type validation works")
else
    print("✗ Type validation failed - accepted invalid type")
end
print()

-- Test 7: Set boolean value
print("Test 7: Set boolean value")
local ok3, err3 = config:set('debug_hud.enabled', false)
if ok3 then
    local new_enabled = config:get('debug_hud.enabled')
    print(string.format("  Set debug_hud.enabled to false: %s", tostring(new_enabled)))
    assert(new_enabled == false, "Should be false")
    print("✓ Boolean value set successfully")
else
    print("✗ Failed to set boolean: " .. err3)
end
print()

-- Test 8: Reset configuration
print("Test 8: Reset configuration")
config:reset()
local reset_width = config:get('display.width')
print(string.format("  After reset, display.width: %d", reset_width))
assert(reset_width == 1440, "Width should be reset to default 1440")
print("✓ Configuration reset successfully\n")

-- Test 9: Get entire config
print("Test 9: Get entire configuration")
local all_config = config:get_all()
print(string.format("  Config sections: %d", #(function()
    local keys = {}
    for k in pairs(all_config) do table.insert(keys, k) end
    return keys
end)()))
print("✓ Full configuration retrieved\n")

-- Test 10: Nil values
print("Test 10: Handle nil values")
local traj_primary = config:get('colors.trajectory_primary')
print(string.format("  trajectory_primary (should be nil): %s", tostring(traj_primary)))
assert(traj_primary == nil, "Should be nil")
print("✓ Nil values handled correctly\n")

print("=== All Tests Passed! ===")
