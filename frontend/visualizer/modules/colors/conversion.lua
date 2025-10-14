--[[
Color Space Conversion Module

Provides utilities for converting between RGB, HSL, and LAB color spaces.
All conversions use standard formulas and maintain color accuracy.

RGB values are expected in [0-1] range (LOVE2D format)
HSL: H in [0-360], S and L in [0-1]
LAB: L in [0-100], A and B in [-128 to 127]
]]

local conversion = {}

-- Helper: Clamp value to range
local function clamp(value, min, max)
    return math.max(min, math.min(max, value))
end

-- Helper: Convert sRGB to linear RGB
local function srgbToLinear(c)
    if c <= 0.04045 then
        return c / 12.92
    else
        return math.pow((c + 0.055) / 1.055, 2.4)
    end
end

-- Helper: Convert linear RGB to sRGB
local function linearToSrgb(c)
    if c <= 0.0031308 then
        return c * 12.92
    else
        return 1.055 * math.pow(c, 1/2.4) - 0.055
    end
end

--[[
Convert RGB color to HSL color space
@param r number Red component [0-1]
@param g number Green component [0-1]
@param b number Blue component [0-1]
@return table {h, s, l} where h is [0-360], s and l are [0-1]
]]
function conversion.rgbToHsl(r, g, b)
    -- Validate inputs
    if type(r) ~= "number" or type(g) ~= "number" or type(b) ~= "number" then
        error("RGB values must be numbers")
    end

    r = clamp(r, 0, 1)
    g = clamp(g, 0, 1)
    b = clamp(b, 0, 1)

    -- Find max and min components
    local max = math.max(r, g, b)
    local min = math.min(r, g, b)
    local delta = max - min

    -- Initialize HSL values
    local h, s, l

    -- Calculate lightness
    l = (max + min) / 2

    -- Handle grayscale (no saturation)
    if delta == 0 then
        h = 0  -- Hue is undefined for grayscale, use 0 by convention
        s = 0  -- No saturation for grayscale
    else
        -- Calculate saturation
        -- S = delta / (1 - |2L - 1|)
        s = delta / (1 - math.abs(2 * l - 1))

        -- Calculate hue based on which component is maximum
        if max == r then
            -- Red is max: H = 60 * ((G-B)/delta mod 6)
            h = 60 * (((g - b) / delta) % 6)
        elseif max == g then
            -- Green is max: H = 60 * ((B-R)/delta + 2)
            h = 60 * ((b - r) / delta + 2)
        else
            -- Blue is max: H = 60 * ((R-G)/delta + 4)
            h = 60 * ((r - g) / delta + 4)
        end

        -- Ensure hue is in [0-360] range
        if h < 0 then
            h = h + 360
        end
    end

    return {h = h, s = s, l = l}
end

--[[
Convert HSL color to RGB color space
@param h number Hue [0-360]
@param s number Saturation [0-1]
@param l number Lightness [0-1]
@return table {r, g, b} where all values are [0-1]
]]
function conversion.hslToRgb(h, s, l)
    -- Validate inputs
    if type(h) ~= "number" or type(s) ~= "number" or type(l) ~= "number" then
        error("HSL values must be numbers")
    end

    -- Normalize hue to [0-360] range
    h = h % 360
    if h < 0 then
        h = h + 360
    end

    s = clamp(s, 0, 1)
    l = clamp(l, 0, 1)

    -- Calculate chroma: C = (1 - |2L - 1|) * S
    local c = (1 - math.abs(2 * l - 1)) * s

    -- Calculate X = C * (1 - |(h/60) mod 2 - 1|)
    local h_prime = h / 60
    local x = c * (1 - math.abs(h_prime % 2 - 1))

    -- Calculate m = L - C/2
    local m = l - c / 2

    -- Determine RGB' values based on hue sector
    local r_prime, g_prime, b_prime

    if h_prime >= 0 and h_prime < 1 then
        -- Red to Yellow sector
        r_prime, g_prime, b_prime = c, x, 0
    elseif h_prime >= 1 and h_prime < 2 then
        -- Yellow to Green sector
        r_prime, g_prime, b_prime = x, c, 0
    elseif h_prime >= 2 and h_prime < 3 then
        -- Green to Cyan sector
        r_prime, g_prime, b_prime = 0, c, x
    elseif h_prime >= 3 and h_prime < 4 then
        -- Cyan to Blue sector
        r_prime, g_prime, b_prime = 0, x, c
    elseif h_prime >= 4 and h_prime < 5 then
        -- Blue to Magenta sector
        r_prime, g_prime, b_prime = x, 0, c
    else
        -- Magenta to Red sector (5-6)
        r_prime, g_prime, b_prime = c, 0, x
    end

    -- Add m to get final RGB values
    return {
        r = r_prime + m,
        g = g_prime + m,
        b = b_prime + m
    }
end

--- Get the complementary hue for a given hue
-- @param hue Hue in [0-360] degrees
-- @return number Complementary hue in [0-360] degrees
function conversion.getComplementaryHue(hue)
    return (hue + 180) % 360
end

--- Get the complementary color for a given RGB color
-- @param r Red component in [0-1] range
-- @param g Green component in [0-1] range
-- @param b Blue component in [0-1] range
-- @return table {r, g, b} Complementary color in RGB
function conversion.getComplementaryColor(r, g, b)
    -- Convert to HSL
    local hsl = conversion.rgbToHsl(r, g, b)

    -- Get complementary hue
    local comp_hue = conversion.getComplementaryHue(hsl.h)

    -- Convert back to RGB with complementary hue
    return conversion.hslToRgb(comp_hue, hsl.s, hsl.l)
end

--[[
Convert RGB to XYZ color space (intermediate for LAB conversion)
Uses D65 illuminant and sRGB color space
@param r number Red component [0-1]
@param g number Green component [0-1]
@param b number Blue component [0-1]
@return table {x, y, z} in XYZ color space
]]
function conversion.rgbToXyz(r, g, b)
    -- Validate inputs
    if type(r) ~= "number" or type(g) ~= "number" or type(b) ~= "number" then
        error("RGB values must be numbers")
    end

    r = clamp(r, 0, 1)
    g = clamp(g, 0, 1)
    b = clamp(b, 0, 1)

    -- Convert to linear RGB
    r = srgbToLinear(r)
    g = srgbToLinear(g)
    b = srgbToLinear(b)

    -- Apply sRGB to XYZ transformation matrix (D65)
    local x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    local y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    local z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    return {x = x * 100, y = y * 100, z = z * 100}
end

--[[
Convert XYZ to RGB color space
Uses D65 illuminant and sRGB color space
@param x number X component
@param y number Y component
@param z number Z component
@return table {r, g, b} where all values are [0-1]
]]
function conversion.xyzToRgb(x, y, z)
    -- Validate inputs
    if type(x) ~= "number" or type(y) ~= "number" or type(z) ~= "number" then
        error("XYZ values must be numbers")
    end

    -- Normalize from [0-100] to [0-1]
    x = x / 100
    y = y / 100
    z = z / 100

    -- Apply XYZ to sRGB transformation matrix (D65)
    local r = x *  3.2404542 + y * -1.5371385 + z * -0.4985314
    local g = x * -0.9692660 + y *  1.8760108 + z *  0.0415560
    local b = x *  0.0556434 + y * -0.2040259 + z *  1.0572252

    -- Convert from linear to sRGB
    r = linearToSrgb(r)
    g = linearToSrgb(g)
    b = linearToSrgb(b)

    -- Clamp to valid range
    r = clamp(r, 0, 1)
    g = clamp(g, 0, 1)
    b = clamp(b, 0, 1)

    return {r = r, g = g, b = b}
end

--[[
Convert XYZ to LAB color space
Uses D65 illuminant (X=95.047, Y=100.0, Z=108.883)
@param x number X component
@param y number Y component
@param z number Z component
@return table {l, a, b} in LAB color space
]]
function conversion.xyzToLab(x, y, z)
    -- Validate inputs
    if type(x) ~= "number" or type(y) ~= "number" or type(z) ~= "number" then
        error("XYZ values must be numbers")
    end

    -- D65 illuminant reference values
    local refX = 95.047
    local refY = 100.0
    local refZ = 108.883

    x = x / refX
    y = y / refY
    z = z / refZ

    local function labF(t)
        if t > 0.008856 then
            return math.pow(t, 1/3)
        else
            return 7.787 * t + 16/116
        end
    end

    x = labF(x)
    y = labF(y)
    z = labF(z)

    local l = 116 * y - 16
    local a = 500 * (x - y)
    local b = 200 * (y - z)

    return {l = l, a = a, b = b}
end

--[[
Convert LAB to XYZ color space
Uses D65 illuminant (X=95.047, Y=100.0, Z=108.883)
@param l number Lightness [0-100]
@param a number A component [-128 to 127]
@param b number B component [-128 to 127]
@return table {x, y, z} in XYZ color space
]]
function conversion.labToXyz(l, a, b)
    -- Validate inputs
    if type(l) ~= "number" or type(a) ~= "number" or type(b) ~= "number" then
        error("LAB values must be numbers")
    end

    local y = (l + 16) / 116
    local x = a / 500 + y
    local z = y - b / 200

    local function labInvF(t)
        if t > 0.206897 then
            return math.pow(t, 3)
        else
            return (t - 16/116) / 7.787
        end
    end

    x = labInvF(x)
    y = labInvF(y)
    z = labInvF(z)

    -- D65 illuminant reference values
    local refX = 95.047
    local refY = 100.0
    local refZ = 108.883

    return {x = x * refX, y = y * refY, z = z * refZ}
end

--[[
Convert RGB to LAB color space (convenience function)
@param r number Red component [0-1]
@param g number Green component [0-1]
@param b number Blue component [0-1]
@return table {l, a, b} in LAB color space
]]
function conversion.rgbToLab(r, g, b)
    local xyz = conversion.rgbToXyz(r, g, b)
    return conversion.xyzToLab(xyz.x, xyz.y, xyz.z)
end

--[[
Convert LAB to RGB color space (convenience function)
@param l number Lightness [0-100]
@param a number A component [-128 to 127]
@param b number B component [-128 to 127]
@return table {r, g, b} where all values are [0-1]
]]
function conversion.labToRgb(l, a, b)
    local xyz = conversion.labToXyz(l, a, b)
    return conversion.xyzToRgb(xyz.x, xyz.y, xyz.z)
end

--[[
Convert RGB from [0-255] to [0-1] range
@param r number Red component [0-255]
@param g number Green component [0-255]
@param b number Blue component [0-255]
@return table {r, g, b} where all values are [0-1]
]]
function conversion.normalize255(r, g, b)
    if type(r) ~= "number" or type(g) ~= "number" or type(b) ~= "number" then
        error("RGB values must be numbers")
    end

    return {
        r = clamp(r / 255, 0, 1),
        g = clamp(g / 255, 0, 1),
        b = clamp(b / 255, 0, 1)
    }
end

--[[
Convert RGB from [0-1] to [0-255] range
@param r number Red component [0-1]
@param g number Green component [0-1]
@param b number Blue component [0-1]
@return table {r, g, b} where all values are [0-255]
]]
function conversion.denormalize255(r, g, b)
    if type(r) ~= "number" or type(g) ~= "number" or type(b) ~= "number" then
        error("RGB values must be numbers")
    end

    return {
        r = math.floor(clamp(r, 0, 1) * 255 + 0.5),
        g = math.floor(clamp(g, 0, 1) * 255 + 0.5),
        b = math.floor(clamp(b, 0, 1) * 255 + 0.5)
    }
end

return conversion
