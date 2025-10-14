-- Verification script for WCAG 2.0 contrast calculations
-- Tests the requested API functions

local contrast = require('modules.colors.contrast')

print("\n" .. string.rep("=", 60))
print("WCAG 2.0 Contrast Ratio Implementation Verification")
print(string.rep("=", 60) .. "\n")

-- Test 1: Black vs White (maximum contrast)
print("Test 1: Black (0,0,0) vs White (1,1,1)")
print(string.rep("-", 60))

local black = {r = 0, g = 0, b = 0}
local white = {r = 1, g = 1, b = 1}

-- Calculate luminance
local blackLum = contrast.calculateLuminance(black.r, black.g, black.b)
local whiteLum = contrast.calculateLuminance(white.r, white.g, white.b)

print(string.format("  Black luminance: %.8f", blackLum))
print(string.format("  White luminance: %.8f", whiteLum))

-- Calculate contrast ratio
local ratio1 = contrast.calculateContrastRatio(black, white)
print(string.format("\n  Contrast ratio:  %.2f:1", ratio1))
print(string.format("  Expected:        21.00:1"))
print(string.format("  Difference:      %.4f", math.abs(ratio1 - 21.0)))
print(string.format("  Test Result:     %s", math.abs(ratio1 - 21.0) < 0.01 and "PASS" or "FAIL"))

-- Check WCAG compliance
local meetsAA = contrast.meetsContrastRequirement(ratio1, "AA")
local meetsAAA = contrast.meetsContrastRequirement(ratio1, "AAA")
print(string.format("\n  Meets AA (4.5:1):  %s", meetsAA and "YES" or "NO"))
print(string.format("  Meets AAA (7:1):   %s", meetsAAA and "YES" or "NO"))

-- Test 2: Cyan vs Green (moderate contrast)
print("\n" .. string.rep("=", 60))
print("Test 2: Cyan (0,1,1) vs Green (0.13,0.55,0.13)")
print(string.rep("-", 60))

local cyan = {r = 0, g = 1, b = 1}
local green = {r = 0.13, g = 0.55, b = 0.13}

-- Calculate luminance
local cyanLum = contrast.calculateLuminance(cyan.r, cyan.g, cyan.b)
local greenLum = contrast.calculateLuminance(green.r, green.g, green.b)

print(string.format("  Cyan luminance:  %.8f", cyanLum))
print(string.format("  Green luminance: %.8f", greenLum))

-- Calculate contrast ratio
local ratio2 = contrast.calculateContrastRatio(cyan, green)
print(string.format("\n  Contrast ratio:  %.2f:1", ratio2))
print(string.format("  Expected:        ~5.80:1"))
print(string.format("  Difference:      %.4f", math.abs(ratio2 - 5.8)))
print(string.format("  Test Result:     %s", math.abs(ratio2 - 5.8) < 0.2 and "PASS" or "FAIL"))

-- Check WCAG compliance
local meetsAA2 = contrast.meetsContrastRequirement(ratio2, "AA")
local meetsAAA2 = contrast.meetsContrastRequirement(ratio2, "AAA")
print(string.format("\n  Meets AA (4.5:1):  %s", meetsAA2 and "YES" or "NO"))
print(string.format("  Meets AAA (7:1):   %s", meetsAAA2 and "YES" or "NO"))

-- Summary
print("\n" .. string.rep("=", 60))
print("Summary")
print(string.rep("=", 60))

local test1Pass = math.abs(ratio1 - 21.0) < 0.01 and meetsAA and meetsAAA
local test2Pass = math.abs(ratio2 - 5.8) < 0.2 and meetsAA2 and not meetsAAA2

print(string.format("\nTest 1 (Black vs White):     %s", test1Pass and "PASS" or "FAIL"))
print(string.format("Test 2 (Cyan vs Green):      %s", test2Pass and "PASS" or "FAIL"))
print(string.format("\nOverall Result:              %s",
    (test1Pass and test2Pass) and "ALL TESTS PASSED" or "SOME TESTS FAILED"))

print("\n" .. string.rep("=", 60) .. "\n")

-- Additional API verification
print("API Function Verification")
print(string.rep("=", 60))
print("\nVerifying all requested functions are available:")
print(string.format("  ✓ calculateLuminance():        %s",
    type(contrast.calculateLuminance) == "function" and "Available" or "Missing"))
print(string.format("  ✓ calculateContrastRatio():    %s",
    type(contrast.calculateContrastRatio) == "function" and "Available" or "Missing"))
print(string.format("  ✓ meetsContrastRequirement():  %s",
    type(contrast.meetsContrastRequirement) == "function" and "Available" or "Missing"))

print("\n" .. string.rep("=", 60) .. "\n")
