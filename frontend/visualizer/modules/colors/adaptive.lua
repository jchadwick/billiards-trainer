--[[
Adaptive Color Palette Generation

Generates accessible color palettes that adapt to table felt color.
Uses perceptually uniform color spaces and ensures WCAG contrast compliance.

Palette includes:
- primary: Main trajectory color (complementary to felt)
- secondary: Ghost ball and secondary elements
- collision: Collision point highlights
- ghost: Ghost ball outline
- aimLine: Aiming assistance line
]]

local conversion = require("modules.colors.conversion")
local contrast = require("modules.colors.contrast")

local Adaptive = {}

-- Helper: Clamp value to range
local function clamp(value, min, max)
    return math.max(min, math.min(max, value))
end

--[[
Calculate color distance in LAB space
LAB space is perceptually uniform, so Euclidean distance represents perceived difference

@param lab1 table {l, a, b} First color in LAB
@param lab2 table {l, a, b} Second color in LAB
@return number Distance between colors
]]
local function labDistance(lab1, lab2)
    local dl = lab1.l - lab2.l
    local da = lab1.a - lab2.a
    local db = lab1.b - lab2.b
    return math.sqrt(dl * dl + da * da + db * db)
end

--[[
Generate a complementary hue with high saturation
Ensures maximum visual distinction from the base color

@param baseHsl table {h, s, l} Base color in HSL
@return number Complementary hue in [0-360]
]]
local function getComplementaryHue(baseHsl)
    -- Use 180-degree offset for maximum distinction
    return (baseHsl.h + 180) % 360
end

--[[
Generate an analogous hue
Creates harmonious colors near the base hue

@param baseHue number Base hue [0-360]
@param offset number Hue offset in degrees (default: 30)
@return number Analogous hue in [0-360]
]]
local function getAnalogousHue(baseHue, offset)
    offset = offset or 30
    return (baseHue + offset) % 360
end

--[[
Ensure color meets minimum contrast with background
Adjusts lightness while preserving hue and saturation

@param rgb table {r, g, b} Color to adjust [0-1]
@param bgRgb table {r, g, b} Background color [0-1]
@param minRatio number Minimum contrast ratio (default: 4.5)
@return table {r, g, b} Adjusted color [0-1]
]]
local function ensureContrast(rgb, bgRgb, minRatio)
    minRatio = minRatio or 4.5

    local adjustedRgb, ratio = contrast.adjustForContrast(
        rgb.r, rgb.g, rgb.b,
        bgRgb.r, bgRgb.g, bgRgb.b,
        minRatio
    )

    return adjustedRgb
end

--[[
Generate primary trajectory color
Uses complementary hue with high saturation for maximum visibility

@param feltRgb table {r, g, b} Table felt color [0-1]
@return table {r, g, b} Primary color [0-1]
]]
local function generatePrimaryColor(feltRgb)
    -- Convert felt to HSL
    local feltHsl = conversion.rgbToHsl(feltRgb.r, feltRgb.g, feltRgb.b)

    -- Generate complementary hue with high saturation
    local primaryHue = getComplementaryHue(feltHsl)
    local primarySat = 0.85  -- High saturation for visibility
    local primaryLight = 0.55  -- Mid lightness for balance

    -- Convert to RGB
    local primaryRgb = conversion.hslToRgb(primaryHue, primarySat, primaryLight)

    -- Ensure contrast with felt
    return ensureContrast(primaryRgb, feltRgb, 4.5)
end

--[[
Generate secondary color for ghost ball and supporting elements
Uses analogous hue to primary for harmony

@param primaryRgb table {r, g, b} Primary color [0-1]
@param feltRgb table {r, g, b} Table felt color [0-1]
@return table {r, g, b} Secondary color [0-1]
]]
local function generateSecondaryColor(primaryRgb, feltRgb)
    -- Convert primary to HSL
    local primaryHsl = conversion.rgbToHsl(primaryRgb.r, primaryRgb.g, primaryRgb.b)

    -- Generate analogous hue (60 degrees offset for distinction)
    local secondaryHue = getAnalogousHue(primaryHsl.h, 60)
    local secondarySat = 0.75  -- Slightly lower saturation
    local secondaryLight = 0.60  -- Slightly lighter

    -- Convert to RGB
    local secondaryRgb = conversion.hslToRgb(secondaryHue, secondarySat, secondaryLight)

    -- Ensure contrast with felt
    return ensureContrast(secondaryRgb, feltRgb, 4.0)
end

--[[
Generate collision point color
High contrast, warm color for attention

@param feltRgb table {r, g, b} Table felt color [0-1]
@return table {r, g, b} Collision color [0-1]
]]
local function generateCollisionColor(feltRgb)
    -- Use warm colors (red-orange range) for collision points
    local collisionHue = 15  -- Orange-red
    local collisionSat = 0.90  -- Very high saturation
    local collisionLight = 0.60  -- Bright

    local collisionRgb = conversion.hslToRgb(collisionHue, collisionSat, collisionLight)

    -- Ensure high contrast for visibility
    return ensureContrast(collisionRgb, feltRgb, 5.0)
end

--[[
Generate ghost ball color
Semi-transparent, lighter version of secondary

@param secondaryRgb table {r, g, b} Secondary color [0-1]
@param feltRgb table {r, g, b} Table felt color [0-1]
@return table {r, g, b} Ghost color [0-1]
]]
local function generateGhostColor(secondaryRgb, feltRgb)
    -- Convert to HSL and lighten
    local secondaryHsl = conversion.rgbToHsl(secondaryRgb.r, secondaryRgb.g, secondaryRgb.b)

    -- Make lighter and slightly less saturated
    local ghostHue = secondaryHsl.h
    local ghostSat = secondaryHsl.s * 0.8
    local ghostLight = math.min(secondaryHsl.l + 0.15, 0.85)

    local ghostRgb = conversion.hslToRgb(ghostHue, ghostSat, ghostLight)

    -- Ensure minimum contrast
    return ensureContrast(ghostRgb, feltRgb, 3.5)
end

--[[
Generate aim line color
Subtle but visible, uses desaturated primary

@param primaryRgb table {r, g, b} Primary color [0-1]
@param feltRgb table {r, g, b} Table felt color [0-1]
@return table {r, g, b} Aim line color [0-1]
]]
local function generateAimLineColor(primaryRgb, feltRgb)
    -- Convert to HSL and desaturate
    local primaryHsl = conversion.rgbToHsl(primaryRgb.r, primaryRgb.g, primaryRgb.b)

    -- Desaturate and adjust lightness for subtlety
    local aimHue = primaryHsl.h
    local aimSat = primaryHsl.s * 0.6
    local aimLight = primaryHsl.l * 1.1

    local aimRgb = conversion.hslToRgb(aimHue, aimSat, aimLight)

    -- Ensure minimum contrast
    return ensureContrast(aimRgb, feltRgb, 3.0)
end

--[[
Generate complete adaptive color palette
Creates a harmonious, accessible palette based on table felt color

@param feltR number Felt red component [0-1]
@param feltG number Felt green component [0-1]
@param feltB number Felt blue component [0-1]
@return table Palette with primary, secondary, collision, ghost, aimLine colors
]]
function Adaptive.generatePalette(feltR, feltG, feltB)
    -- Validate inputs
    if type(feltR) ~= "number" or type(feltG) ~= "number" or type(feltB) ~= "number" then
        error("Felt RGB values must be numbers")
    end

    -- Clamp values
    feltR = clamp(feltR, 0, 1)
    feltG = clamp(feltG, 0, 1)
    feltB = clamp(feltB, 0, 1)

    local feltRgb = {r = feltR, g = feltG, b = feltB}

    -- Generate palette
    local primary = generatePrimaryColor(feltRgb)
    local secondary = generateSecondaryColor(primary, feltRgb)
    local collision = generateCollisionColor(feltRgb)
    local ghost = generateGhostColor(secondary, feltRgb)
    local aimLine = generateAimLineColor(primary, feltRgb)

    return {
        primary = primary,
        secondary = secondary,
        collision = collision,
        ghost = ghost,
        aimLine = aimLine,
        felt = feltRgb
    }
end

--[[
Validate that a palette meets minimum contrast requirements
@param palette table Palette from generatePalette
@return boolean True if palette is valid
@return table Array of validation messages
]]
function Adaptive.validatePalette(palette)
    if not palette or type(palette) ~= "table" then
        return false, {"Invalid palette structure"}
    end

    local messages = {}
    local isValid = true

    -- Check each color against felt background
    local colors = {"primary", "secondary", "collision", "ghost", "aimLine"}
    local minRatios = {
        primary = 4.5,
        secondary = 4.0,
        collision = 5.0,
        ghost = 3.5,
        aimLine = 3.0
    }

    for _, colorName in ipairs(colors) do
        local color = palette[colorName]
        if not color then
            isValid = false
            table.insert(messages, string.format("Missing %s color", colorName))
        else
            local ratio = contrast.getContrastRatio(
                color.r, color.g, color.b,
                palette.felt.r, palette.felt.g, palette.felt.b
            )

            local minRatio = minRatios[colorName]
            if ratio < minRatio then
                isValid = false
                table.insert(messages, string.format(
                    "%s contrast ratio %.2f is below minimum %.2f",
                    colorName, ratio, minRatio
                ))
            else
                table.insert(messages, string.format(
                    "%s contrast ratio %.2f meets minimum %.2f",
                    colorName, ratio, minRatio
                ))
            end
        end
    end

    return isValid, messages
end

--[[
Get a preset palette for common felt colors
@param presetName string Preset name: "green", "blue", "red", "burgundy"
@return table RGB color {r, g, b} in [0-1] range
]]
function Adaptive.getPresetFelt(presetName)
    local presets = {
        green = {r = 0.13, g = 0.55, b = 0.13},      -- Traditional green felt
        blue = {r = 0.10, g = 0.20, b = 0.50},        -- Blue felt
        red = {r = 0.55, g = 0.10, b = 0.10},         -- Red felt
        burgundy = {r = 0.40, g = 0.10, b = 0.20},    -- Burgundy felt
        black = {r = 0.08, g = 0.08, b = 0.08},       -- Black felt
        purple = {r = 0.30, g = 0.10, b = 0.40}       -- Purple felt
    }

    return presets[presetName] or presets.green
end

-- ============================================================================
-- Public API Wrappers (matching requested interface)
-- ============================================================================

--[[
  Generate a high-contrast color based on table felt color.

  Public wrapper that accepts table parameter for compatibility.
  Generates a color with complementary hue and ensures 4.5:1 contrast ratio.

  @param baseColorRGB table Base color (table felt) as {r, g, b} in [0-1]
  @return table Color as {r, g, b, a} with a=0.9
]]
function Adaptive.generateHighContrastColor(baseColorRGB)
    -- Validate input
    if type(baseColorRGB) ~= "table" or
       type(baseColorRGB.r) ~= "number" or
       type(baseColorRGB.g) ~= "number" or
       type(baseColorRGB.b) ~= "number" then
        error("baseColorRGB must be a table with r, g, b numeric fields")
    end

    -- Generate primary color (which is the high contrast color)
    local result = generatePrimaryColor(baseColorRGB)

    -- Add alpha channel
    return {
        r = result.r,
        g = result.g,
        b = result.b,
        a = 0.9
    }
end

--[[
  Generate a complementary color with adjustments for visibility.

  Public wrapper that accepts table parameter for compatibility.
  Uses 180° hue rotation with adjustments for contrast.

  @param baseColorRGB table Base color as {r, g, b} in [0-1]
  @return table Complementary color as {r, g, b, a} with a=0.9
]]
function Adaptive.generateComplementaryColor(baseColorRGB)
    -- Validate input
    if type(baseColorRGB) ~= "table" or
       type(baseColorRGB.r) ~= "number" or
       type(baseColorRGB.g) ~= "number" or
       type(baseColorRGB.b) ~= "number" then
        error("baseColorRGB must be a table with r, g, b numeric fields")
    end

    -- Convert base color to HSL
    local baseHsl = conversion.rgbToHsl(baseColorRGB.r, baseColorRGB.g, baseColorRGB.b)

    -- Get complementary hue
    local compHue = getComplementaryHue(baseHsl)

    -- Calculate base luminance
    local baseLuminance = contrast.getLuminance(baseColorRGB.r, baseColorRGB.g, baseColorRGB.b)

    -- Set saturation and lightness based on background
    local saturation = 0.85
    local lightness = baseLuminance < 0.5 and 0.75 or 0.35

    -- Generate complementary color
    local compColor = conversion.hslToRgb(compHue, saturation, lightness)

    -- Ensure contrast
    local result = ensureContrast(compColor, baseColorRGB, 4.5)

    -- Add alpha channel
    return {
        r = result.r,
        g = result.g,
        b = result.b,
        a = 0.9
    }
end

return Adaptive
