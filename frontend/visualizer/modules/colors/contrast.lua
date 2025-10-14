--[[
Contrast Module

Provides WCAG 2.1 compliant contrast ratio calculations.
Uses relative luminance to determine contrast between colors.

WCAG contrast ratio guidelines:
- AA Normal Text: 4.5:1 minimum
- AA Large Text: 3:1 minimum
- AAA Normal Text: 7:1 minimum
- AAA Large Text: 4.5:1 minimum
]]

local Contrast = {}

-- Helper: Clamp value to range
local function clamp(value, min, max)
    return math.max(min, math.min(max, value))
end

-- Helper: Convert sRGB to linear RGB for luminance calculation
-- WCAG 2.0 specifies 0.03928 as the threshold value
-- This is the correct value per the specification:
-- https://www.w3.org/TR/WCAG20/#relativeluminancedef
local function srgbToLinear(c)
    if c <= 0.03928 then
        return c / 12.92
    else
        return math.pow((c + 0.055) / 1.055, 2.4)
    end
end

--[[
Calculate relative luminance of an RGB color
Uses WCAG 2.1 formula: L = 0.2126 * R + 0.7152 * G + 0.0722 * B
where R, G, B are linearized sRGB values

@param r number Red component [0-1]
@param g number Green component [0-1]
@param b number Blue component [0-1]
@return number Relative luminance [0-1]
]]
function Contrast.getLuminance(r, g, b)
    -- Validate inputs
    if type(r) ~= "number" or type(g) ~= "number" or type(b) ~= "number" then
        error("RGB values must be numbers")
    end

    r = clamp(r, 0, 1)
    g = clamp(g, 0, 1)
    b = clamp(b, 0, 1)

    -- Convert to linear RGB
    local rLinear = srgbToLinear(r)
    local gLinear = srgbToLinear(g)
    local bLinear = srgbToLinear(b)

    -- Calculate relative luminance
    return 0.2126 * rLinear + 0.7152 * gLinear + 0.0722 * bLinear
end

--[[
Calculate contrast ratio between two colors
Uses WCAG 2.1 formula: (L1 + 0.05) / (L2 + 0.05)
where L1 is the lighter color's luminance and L2 is the darker

@param r1 number First color red component [0-1]
@param g1 number First color green component [0-1]
@param b1 number First color blue component [0-1]
@param r2 number Second color red component [0-1]
@param g2 number Second color green component [0-1]
@param b2 number Second color blue component [0-1]
@return number Contrast ratio [1-21]
]]
function Contrast.getContrastRatio(r1, g1, b1, r2, g2, b2)
    -- Calculate luminance for both colors
    local l1 = Contrast.getLuminance(r1, g1, b1)
    local l2 = Contrast.getLuminance(r2, g2, b2)

    -- Ensure l1 is the lighter color
    if l1 < l2 then
        l1, l2 = l2, l1
    end

    -- Calculate contrast ratio
    return (l1 + 0.05) / (l2 + 0.05)
end

--[[
Check if contrast ratio meets WCAG AA standards for normal text
Minimum ratio: 4.5:1

@param ratio number Contrast ratio
@return boolean True if meets AA normal text standards
]]
function Contrast.meetsAA(ratio)
    if type(ratio) ~= "number" then
        error("Contrast ratio must be a number")
    end
    return ratio >= 4.5
end

--[[
Check if contrast ratio meets WCAG AA standards for large text
Minimum ratio: 3:1

@param ratio number Contrast ratio
@return boolean True if meets AA large text standards
]]
function Contrast.meetsAALarge(ratio)
    if type(ratio) ~= "number" then
        error("Contrast ratio must be a number")
    end
    return ratio >= 3.0
end

--[[
Check if contrast ratio meets WCAG AAA standards for normal text
Minimum ratio: 7:1

@param ratio number Contrast ratio
@return boolean True if meets AAA normal text standards
]]
function Contrast.meetsAAA(ratio)
    if type(ratio) ~= "number" then
        error("Contrast ratio must be a number")
    end
    return ratio >= 7.0
end

--[[
Check if contrast ratio meets WCAG AAA standards for large text
Minimum ratio: 4.5:1

@param ratio number Contrast ratio
@return boolean True if meets AAA large text standards
]]
function Contrast.meetsAAALarge(ratio)
    if type(ratio) ~= "number" then
        error("Contrast ratio must be a number")
    end
    return ratio >= 4.5
end

--[[
Get the WCAG compliance level for a contrast ratio
@param ratio number Contrast ratio
@param isLargeText boolean Optional, whether text is large (default: false)
@return string Compliance level: "AAA", "AA", "Fail", or "N/A"
]]
function Contrast.getComplianceLevel(ratio, isLargeText)
    if type(ratio) ~= "number" then
        error("Contrast ratio must be a number")
    end

    isLargeText = isLargeText or false

    if isLargeText then
        if Contrast.meetsAAA(ratio) then
            return "AAA"
        elseif Contrast.meetsAALarge(ratio) then
            return "AA"
        else
            return "Fail"
        end
    else
        if Contrast.meetsAAA(ratio) then
            return "AAA"
        elseif Contrast.meetsAA(ratio) then
            return "AA"
        else
            return "Fail"
        end
    end
end

--[[
Calculate minimum contrast ratio for readability
Returns 4.5 for standard UI elements (AA normal text)

@return number Minimum recommended contrast ratio
]]
function Contrast.getMinimumRatio()
    return 4.5
end

--[[
Check if two colors have sufficient contrast for UI visibility
Uses AA normal text standard (4.5:1) as minimum

@param r1 number First color red component [0-1]
@param g1 number First color green component [0-1]
@param b1 number First color blue component [0-1]
@param r2 number Second color red component [0-1]
@param g2 number Second color green component [0-1]
@param b2 number Second color blue component [0-1]
@return boolean True if contrast is sufficient
@return number The actual contrast ratio
]]
function Contrast.hasSufficientContrast(r1, g1, b1, r2, g2, b2)
    local ratio = Contrast.getContrastRatio(r1, g1, b1, r2, g2, b2)
    return ratio >= Contrast.getMinimumRatio(), ratio
end

--[[
Find the closest color to target that meets minimum contrast with background
Adjusts lightness in LAB color space to achieve target contrast

@param targetR number Target color red [0-1]
@param targetG number Target color green [0-1]
@param targetB number Target color blue [0-1]
@param bgR number Background color red [0-1]
@param bgG number Background color green [0-1]
@param bgB number Background color blue [0-1]
@param minRatio number Minimum contrast ratio (default: 4.5)
@return table {r, g, b} Adjusted color in [0-1] range
@return number Achieved contrast ratio
]]
function Contrast.adjustForContrast(targetR, targetG, targetB, bgR, bgG, bgB, minRatio)
    minRatio = minRatio or 4.5

    -- Check if already meets contrast
    local currentRatio = Contrast.getContrastRatio(targetR, targetG, targetB, bgR, bgG, bgB)
    if currentRatio >= minRatio then
        return {r = targetR, g = targetG, b = targetB}, currentRatio
    end

    -- Need to require conversion module for LAB adjustments
    local conversion = require("modules.colors.conversion")

    -- Convert to LAB for perceptually uniform adjustments
    local lab = conversion.rgbToLab(targetR, targetG, targetB)
    local bgLuminance = Contrast.getLuminance(bgR, bgG, bgB)

    -- Determine if we need to lighten or darken
    local targetLuminance = Contrast.getLuminance(targetR, targetG, targetB)
    local shouldLighten = bgLuminance > targetLuminance

    -- Adjust lightness iteratively
    local step = shouldLighten and 5 or -5
    local maxIterations = 20
    local bestColor = {r = targetR, g = targetG, b = targetB}
    local bestRatio = currentRatio

    for i = 1, maxIterations do
        lab.l = clamp(lab.l + step, 0, 100)

        -- Convert back to RGB
        local rgb = conversion.labToRgb(lab.l, lab.a, lab.b)

        -- Check contrast
        local ratio = Contrast.getContrastRatio(rgb.r, rgb.g, rgb.b, bgR, bgG, bgB)

        if ratio >= minRatio then
            return rgb, ratio
        end

        -- Track best attempt
        if ratio > bestRatio then
            bestColor = rgb
            bestRatio = ratio
        end

        -- Stop if we've maxed out lightness
        if lab.l <= 0 or lab.l >= 100 then
            break
        end
    end

    -- Return best attempt even if it didn't meet target
    return bestColor, bestRatio
end

-- ============================================================================
-- API Wrapper Functions (matching requested interface)
-- ============================================================================

--[[
  Calculate the relative luminance of an RGB color according to WCAG 2.0.
  This is a wrapper function that matches the requested API.

  @param r number Red component in [0-1] range (normalized)
  @param g number Green component in [0-1] range (normalized)
  @param b number Blue component in [0-1] range (normalized)
  @return number Relative luminance value in [0-1] range
]]
function Contrast.calculateLuminance(r, g, b)
    return Contrast.getLuminance(r, g, b)
end

--[[
  Calculate the contrast ratio between two RGB colors according to WCAG 2.0.
  This is a wrapper function that matches the requested API using table parameters.

  @param rgb1 table First color as {r, g, b} with values in [0-1] range
  @param rgb2 table Second color as {r, g, b} with values in [0-1] range
  @return number Contrast ratio (e.g., 5.8 for a 5.8:1 ratio)
]]
function Contrast.calculateContrastRatio(rgb1, rgb2)
    return Contrast.getContrastRatio(rgb1.r, rgb1.g, rgb1.b, rgb2.r, rgb2.g, rgb2.b)
end

--[[
  Check if a contrast ratio meets WCAG 2.0 accessibility requirements.
  This is a wrapper function that matches the requested API.

  @param ratio number The calculated contrast ratio
  @param level string The WCAG level: "AA" or "AAA"
  @return boolean True if the ratio meets or exceeds the requirement
]]
function Contrast.meetsContrastRequirement(ratio, level)
    if level == "AA" then
        return Contrast.meetsAA(ratio)
    elseif level == "AAA" then
        return Contrast.meetsAAA(ratio)
    else
        error(string.format("Invalid WCAG level '%s'. Must be 'AA' or 'AAA'.", tostring(level)))
    end
end

return Contrast
