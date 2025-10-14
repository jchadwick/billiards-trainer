--- Adaptive Color Management Module
-- Provides intelligent color selection based on table felt color
-- Ensures optimal visibility and contrast for trajectory visualization
--
-- This module dynamically adjusts trajectory colors to maintain
-- high contrast ratios against the table felt background.

local contrast = require("modules.colors.contrast")
local conversion = require("modules.colors.conversion")
local adaptive = require("modules.colors.adaptive")

local ColorManager = {
    name = "colors",
    priority = 50, -- Load early, before trajectory module
    tableFeltColor = {r = 0.13, g = 0.55, b = 0.13}, -- Default green felt RGB [0-1]
    cachedPalette = nil,
    autoAdapt = true, -- Automatically regenerate palette on color change
}

--- Initialize the color management module
-- @param config table Configuration options
-- @return boolean Success status
function ColorManager.init(config)
    config = config or {}

    -- Apply configuration if provided
    if config.tableFeltColor then
        ColorManager.setTableFeltColor(config.tableFeltColor)
    end

    if config.preset then
        local presetColor = ColorManager.getPresetFelt(config.preset)
        if presetColor then
            ColorManager.setTableFeltColor(presetColor)
        end
    end

    if config.auto_adapt ~= nil then
        ColorManager.autoAdapt = config.auto_adapt
    end

    -- Generate initial palette
    local success, result = pcall(function()
        return ColorManager.getColorPalette()
    end)

    if not success then
        print("[ColorManager] Error: Failed to generate initial palette: " .. tostring(result))
        return false
    end

    -- Validate initial palette
    local isValid, messages = adaptive.validatePalette(result)

    print("[ColorManager] Initialized with table felt color:",
          ColorManager.tableFeltColor.r,
          ColorManager.tableFeltColor.g,
          ColorManager.tableFeltColor.b)

    if not isValid then
        print("[ColorManager] Warning: Initial palette has contrast issues")
    end

    return true
end

--- Update the table felt color and optionally regenerate trajectory colors
-- @param rgb table RGB color values {r=, g=, b=} in range [0, 1]
-- @return table Updated or current palette
function ColorManager.setTableFeltColor(rgb)
    assert(type(rgb) == "table", "RGB must be a table")
    assert(rgb.r and rgb.g and rgb.b, "RGB must have r, g, b fields")

    -- Validate color values are in valid range [0, 1]
    local function clamp(v)
        return math.max(0, math.min(1, v))
    end

    ColorManager.tableFeltColor = {
        r = clamp(rgb.r),
        g = clamp(rgb.g),
        b = clamp(rgb.b)
    }

    -- Clear cached palette to trigger regeneration
    ColorManager.cachedPalette = nil

    -- Regenerate palette if auto-adapt is enabled
    if ColorManager.autoAdapt then
        local success, result = pcall(function()
            return ColorManager.getColorPalette()
        end)

        if not success then
            print("[ColorManager] Error: Failed to regenerate palette: " .. tostring(result))
        end

        return result
    end

    return ColorManager.cachedPalette
end

--- Get optimized trajectory colors based on current table felt color
-- Returns colors that maintain high contrast against the table surface
-- @param tableFeltRGB table Optional RGB override {r=, g=, b=}
-- @return table Primary trajectory color {r=, g=, b=}
-- @return table Secondary trajectory color {r=, g=, b=}
function ColorManager.getTrajectoryColors(tableFeltRGB)
    local baseColor = tableFeltRGB or ColorManager.tableFeltColor

    -- Get or generate palette
    local palette = ColorManager.getColorPalette(baseColor)

    return palette.primary, palette.secondary
end

--- Calculate WCAG 2.0 contrast ratio between two colors
-- @param color1 table First color {r=, g=, b=}
-- @param color2 table Second color {r=, g=, b=}
-- @return number Contrast ratio (1:1 to 21:1)
function ColorManager.getContrastRatio(color1, color2)
    assert(type(color1) == "table", "color1 must be a table")
    assert(type(color2) == "table", "color2 must be a table")
    assert(color1.r and color1.g and color1.b, "color1 must have r, g, b fields")
    assert(color2.r and color2.g and color2.b, "color2 must have r, g, b fields")

    return contrast.getContrastRatio(
        color1.r, color1.g, color1.b,
        color2.r, color2.g, color2.b
    )
end

--- Get a complete color palette for the current table felt
-- @param feltRgb table Optional felt color override {r=, g=, b=}
-- @return table Color palette with primary, secondary, collision, ghost, aimLine
function ColorManager.getColorPalette(feltRgb)
    feltRgb = feltRgb or ColorManager.tableFeltColor

    -- Return cached palette if available and no override provided
    if not feltRgb and ColorManager.cachedPalette then
        return ColorManager.cachedPalette
    end

    -- Generate new palette
    local palette = adaptive.generatePalette(feltRgb.r, feltRgb.g, feltRgb.b)

    -- Validate palette
    local isValid, messages = adaptive.validatePalette(palette)

    if not isValid then
        print("[ColorManager] Warning: Generated palette does not meet all contrast requirements:")
        for _, msg in ipairs(messages) do
            print("  " .. msg)
        end
    else
        print("[ColorManager] Generated valid color palette:")
        for _, msg in ipairs(messages) do
            print("  " .. msg)
        end
    end

    -- Cache the palette if no override was provided
    if not feltRgb then
        ColorManager.cachedPalette = palette
    end

    return palette
end

--- Get a preset felt color
-- @param presetName string Preset name: "green", "blue", "red", "burgundy", "black", "purple"
-- @return table RGB color {r=, g=, b=}
function ColorManager.getPresetFelt(presetName)
    return adaptive.getPresetFelt(presetName)
end

--- Configure color management system
-- Accepts configuration updates for felt color and behavior
-- @param config table Configuration object with optional fields:
--   - table_felt: RGB color in [0-1] or [0-255] (auto-detected) as {r=, g=, b=}
--   - auto_adapt: boolean - Enable/disable automatic palette regeneration
--   - preset: string - Use preset felt color ("green", "blue", "red", etc.)
-- @return boolean Success status
function ColorManager.configure(config)
    if type(config) ~= "table" then
        error("Configuration must be a table")
    end

    local success = true

    -- Handle preset felt color
    if config.preset then
        local presetColor = ColorManager.getPresetFelt(config.preset)
        if presetColor then
            ColorManager.setTableFeltColor(presetColor)
        else
            print("[ColorManager] Warning: Unknown preset '" .. tostring(config.preset) .. "', using current color")
            success = false
        end
    end

    -- Handle custom felt color
    if config.table_felt then
        local felt = config.table_felt

        -- Validate structure
        if type(felt) ~= "table" then
            error("table_felt must be a table with r, g, b values")
        end

        local r = felt.r
        local g = felt.g
        local b = felt.b

        if not r or not g or not b then
            error("table_felt must contain r, g, b values")
        end

        -- Auto-detect if values are in [0-255] range and normalize if needed
        if r > 1 or g > 1 or b > 1 then
            local normalized = conversion.normalize255(r, g, b)
            r = normalized.r
            g = normalized.g
            b = normalized.b
        end

        ColorManager.setTableFeltColor({r = r, g = g, b = b})
    end

    -- Handle auto-adapt setting
    if config.auto_adapt ~= nil then
        if type(config.auto_adapt) ~= "boolean" then
            error("auto_adapt must be a boolean")
        end
        ColorManager.autoAdapt = config.auto_adapt
    end

    return success
end

--- Regenerate palette with current felt color
-- Forces palette regeneration even if auto_adapt is disabled
-- @return table New color palette
function ColorManager.regeneratePalette()
    ColorManager.cachedPalette = nil
    return ColorManager.getColorPalette()
end

--- Validate current palette against WCAG standards
-- @return boolean True if palette meets all contrast requirements
-- @return table Array of validation messages
function ColorManager.validatePalette()
    if not ColorManager.cachedPalette then
        -- Generate palette if not cached
        ColorManager.getColorPalette()
    end

    return adaptive.validatePalette(ColorManager.cachedPalette)
end

--- Get module information for debugging
-- @return table Module info with version, state, etc.
function ColorManager.getInfo()
    return {
        name = ColorManager.name,
        priority = ColorManager.priority,
        initialized = ColorManager.cachedPalette ~= nil,
        autoAdapt = ColorManager.autoAdapt,
        feltColor = ColorManager.tableFeltColor,
        paletteValid = ColorManager.cachedPalette and adaptive.validatePalette(ColorManager.cachedPalette) or false
    }
end

--- Update module state
-- @param dt number Delta time in seconds
function ColorManager.update(dt)
    -- TODO: Handle any animations or transitions
    -- TODO: Monitor for table color changes from video feed
end

--- Cleanup module resources
function ColorManager.cleanup()
    -- Clear cached color palette
    ColorManager.cachedPalette = nil

    print("[ColorManager] Cleaned up")
end

return ColorManager
