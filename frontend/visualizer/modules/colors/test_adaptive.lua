#!/usr/bin/env lua

--[[
Test script for adaptive color generation

Tests the adaptive.lua module with green felt RGB(34, 139, 34) / (0.13, 0.55, 0.13)
Expected: Cyan-like colors with 4.5:1 contrast ratio
]]

-- Add module path
package.path = package.path .. ";../../?.lua;../?.lua;?.lua"

-- Load modules
local adaptive = require("modules.colors.adaptive")
local contrast = require("modules.colors.contrast")

print("\n" .. string.rep("=", 70))
print("Adaptive Color Generation Test")
print(string.rep("=", 70))

-- Test with standard green felt
local greenFelt = {r = 0.13, g = 0.55, b = 0.13}

print(string.format("\nTable Felt Color: RGB(%.2f, %.2f, %.2f)",
    greenFelt.r, greenFelt.g, greenFelt.b))

local luminance = contrast.getLuminance(greenFelt.r, greenFelt.g, greenFelt.b)
print(string.format("Table Felt Luminance: %.4f", luminance))

-- Test 1: generateHighContrastColor
print("\n" .. string.rep("-", 70))
print("Test 1: generateHighContrastColor()")
print(string.rep("-", 70))

local highContrast = adaptive.generateHighContrastColor(greenFelt)
print(string.format("Generated Color: RGB(%.3f, %.3f, %.3f, %.1f)",
    highContrast.r, highContrast.g, highContrast.b, highContrast.a))

local ratio1 = contrast.getContrastRatio(
    highContrast.r, highContrast.g, highContrast.b,
    greenFelt.r, greenFelt.g, greenFelt.b
)
print(string.format("Contrast Ratio: %.2f:1", ratio1))
print(string.format("Meets WCAG AA (4.5:1): %s",
    contrast.meetsAA(ratio1) and "YES" or "NO"))

-- Test 2: generateComplementaryColor
print("\n" .. string.rep("-", 70))
print("Test 2: generateComplementaryColor()")
print(string.rep("-", 70))

local complementary = adaptive.generateComplementaryColor(greenFelt)
print(string.format("Generated Color: RGB(%.3f, %.3f, %.3f, %.1f)",
    complementary.r, complementary.g, complementary.b, complementary.a))

local ratio2 = contrast.getContrastRatio(
    complementary.r, complementary.g, complementary.b,
    greenFelt.r, greenFelt.g, greenFelt.b
)
print(string.format("Contrast Ratio: %.2f:1", ratio2))
print(string.format("Meets WCAG AA (4.5:1): %s",
    contrast.meetsAA(ratio2) and "YES" or "NO"))

-- Test 3: generatePalette
print("\n" .. string.rep("-", 70))
print("Test 3: generatePalette()")
print(string.rep("-", 70))

local palette = adaptive.generatePalette(greenFelt.r, greenFelt.g, greenFelt.b)

print(string.format("\nGenerated Palette for Green Felt:"))
print(string.rep("-", 70))

local colorNames = {"primary", "secondary", "collision", "ghost", "aimLine"}
local minRatios = {
    primary = 4.5,
    secondary = 4.0,
    collision = 5.0,
    ghost = 3.5,
    aimLine = 3.0
}

local allPass = true
for _, name in ipairs(colorNames) do
    local color = palette[name]
    local ratio = contrast.getContrastRatio(
        color.r, color.g, color.b,
        greenFelt.r, greenFelt.g, greenFelt.b
    )
    local minRatio = minRatios[name]
    local meets = ratio >= minRatio

    print(string.format("\n%s:", name:upper()))
    print(string.format("  RGB: (%.3f, %.3f, %.3f)", color.r, color.g, color.b))
    print(string.format("  Contrast Ratio: %.2f:1", ratio))
    print(string.format("  Minimum Required: %.1f:1", minRatio))
    print(string.format("  Status: %s", meets and "PASS" or "FAIL"))

    allPass = allPass and meets
end

-- Test 4: Validate palette
print("\n" .. string.rep("-", 70))
print("Test 4: validatePalette()")
print(string.rep("-", 70))

local isValid, messages = adaptive.validatePalette(palette)

print(string.format("\nValidation Result: %s", isValid and "VALID" or "INVALID"))
print("\nValidation Messages:")
for i, msg in ipairs(messages) do
    print(string.format("  %d. %s", i, msg))
end

-- Summary
print("\n" .. string.rep("=", 70))
print(string.format("Overall Result: %s",
    (allPass and isValid) and "ALL TESTS PASSED" or "SOME TESTS FAILED"))
print(string.rep("=", 70) .. "\n")

-- Test with different preset felt colors
print("\n" .. string.rep("=", 70))
print("Bonus: Test with Different Felt Colors")
print(string.rep("=", 70))

local presets = {"green", "blue", "red", "burgundy", "black", "purple"}

for _, presetName in ipairs(presets) do
    local felt = adaptive.getPresetFelt(presetName)
    local pal = adaptive.generatePalette(felt.r, felt.g, felt.b)
    local valid, msgs = adaptive.validatePalette(pal)

    print(string.format("\n%s Felt: RGB(%.2f, %.2f, %.2f) - %s",
        presetName:upper(),
        felt.r, felt.g, felt.b,
        valid and "VALID" or "INVALID"
    ))
end

print("")
