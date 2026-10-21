-- Coordinate contract shared by the projector's WebSocket consumers.
--
-- The backend sends image geometry as raw x/y plus per-axis factors that map
-- it onto its canonical 4K image plane. These are coordinate-space constants,
-- not camera input scales. Physical dimensions tagged in meters are kept in
-- meters and must be mapped using table dimensions when a renderer needs them.
local Coordinates = {
    CANONICAL_WIDTH = 3840,
    CANONICAL_HEIGHT = 2160,
    CANONICAL_SPACE = "4k_canonical",
    -- Match the backend's camera-to-table homography: physical coordinates
    -- begin at the calibrated playing area's top-left corner, with +X toward
    -- top-right and +Y toward bottom-left.
    TABLE_METER_ORIGIN = "playing_area_top_left"
}

local function finite_number(value)
    return type(value) == "number" and value == value
        and value ~= math.huge and value ~= -math.huge
end

local function canonical_space(space)
    return space == "4k_canonical"
end

local function valid_scale(scale)
    return type(scale) == "table"
        and finite_number(scale[1]) and scale[1] > 0
        and finite_number(scale[2]) and scale[2] > 0
end

local function image_space(space)
    return space == nil or canonical_space(space) or space == "pixel"
end

local image_units = {
    pixels = true,
    canonical_pixels = true,
    pixels_per_second = true,
    canonical_pixels_per_second = true
}

local physical_units = {
    meter = true,
    meters = true,
    meter_per_second = true,
    meters_per_second = true
}

Coordinates.validScale = valid_scale
Coordinates.isFiniteNumber = finite_number

local function xy_values(point)
    if type(point) ~= "table" then return nil, nil end
    if finite_number(point.x) and finite_number(point.y) then
        return point.x, point.y
    end
    return nil, nil
end

--- Convert a tagged image point to canonical pixels without mutating its input.
-- API points must use named x/y fields and explicit own or inherited scale.
function Coordinates.toCanonicalPoint(point, fallbackScale, fallbackSpace)
    local x, y = xy_values(point)
    if not x or not y then return nil, "point must contain finite x and y" end

    if point.coordinate_scale ~= nil or point.units ~= nil then
        return nil, "point uses unsupported legacy coordinate metadata"
    end
    if type(point) == "table" and (point.width_px ~= nil or point.height_px ~= nil) then
        return nil, "pixel extents must use width and height fields"
    end

    local pointScale = type(point) == "table" and point.scale or nil
    local scale = pointScale or fallbackScale
    local space = type(point) == "table" and point.coordinate_space or fallbackSpace
    local unit = type(point) == "table" and point.unit or nil

    -- Image points may have an optional explicit space tag, but a projector,
    -- normalized-table, or physical point must never be treated as camera
    -- pixels merely because it also happens to carry a scale pair.
    if not image_space(space) then
        return nil, "image point coordinate space is incompatible with canonical pixels"
    end
    if unit ~= nil and not image_units[unit] then
        return nil, "image point unit is incompatible with canonical pixels"
    end

    if not valid_scale(scale) then
        return nil, "image point requires two finite positive scale factors"
    end

    local canonical = {}
    if type(point) == "table" then
        for key, value in pairs(point) do
            if key ~= "scale" and key ~= "coordinate_space" then
                canonical[key] = value
            end
        end
    end
    canonical.x = x * scale[1]
    canonical.y = y * scale[2]
    -- Rect-like records carry extents as coordinates too. Preserve their shape
    -- while applying the matching axis factor to each span.
    if canonical.width ~= nil then
        if not finite_number(canonical.width) then return nil, "point width must be finite" end
        canonical.width = canonical.width * scale[1]
    end
    if canonical.height ~= nil then
        if not finite_number(canonical.height) then return nil, "point height must be finite" end
        canonical.height = canonical.height * scale[2]
    end
    canonical.scale = {1, 1}
    canonical.coordinate_space = Coordinates.CANONICAL_SPACE
    return canonical
end

local function clone_table(value)
    local copy = {}
    for key, item in pairs(value) do
        if type(item) == "table" then
            copy[key] = clone_table(item)
        else
            copy[key] = item
        end
    end
    return copy
end

local point_fields = {
    position = true,
    tip_position = true,
    butt_position = true,
    predicted_position = true,
    velocity = true,
    acceleration = true,
    direction = true,
    center = true,
    point = true,
    start = true,
    finish = true,
    ["end"] = true,
    position_history = true,
    history = true,
    playing_area_corners = true,
    corner_points = true,
    corners = true,
    pocket_positions = true,
    pockets = true,
    rails = true,
    rail_positions = true,
    boundaries = true,
    boundary = true
}

local scale_fields = {
    scale = true,
    dimensions_scale = true,
    radius_scale = true,
    pocket_radius_scale = true,
    length_scale = true,
    width_scale = true,
    height_scale = true
}

local function scale_bounds(bounds, inheritedScale, inheritedSpace)
    if type(bounds) ~= "table" then return nil, "bounds must be an object or array" end
    if bounds.coordinate_scale ~= nil or bounds.units ~= nil then
        return nil, "bounds use unsupported legacy coordinate metadata"
    end
    local scale = bounds.scale or inheritedScale
    local space = bounds.coordinate_space or inheritedSpace
    local unit = bounds.unit
    if space == nil and (unit == "normalized_projector" or unit == "normalized_table") then
        space = unit
    end
    if bounds.unit == "meter" or bounds.unit == "meters" or space == "world_meters" then
        if space ~= nil and space ~= "world_meters" then
            return nil, "physical bounds cannot declare an image coordinate space"
        end
        return clone_table(bounds)
    end
    local xKeys = {"left", "right", "min_x", "max_x", "x_min", "x_max"}
    local yKeys = {"top", "bottom", "min_y", "max_y", "y_min", "y_max"}
    if space == "normalized_projector" or space == "normalized_table" then
        if not valid_scale(scale) or scale[1] ~= 1 or scale[2] ~= 1 then
            return nil, "normalized table/projector bounds require identity scale"
        end
        local result = clone_table(bounds)
        local hasNamedBounds = false
        if result.x ~= nil and result.y ~= nil
            and result.width ~= nil and result.height ~= nil then
            if not finite_number(result.x) or not finite_number(result.y)
                or not finite_number(result.width) or not finite_number(result.height) then
                return nil, "normalized rectangle bounds must be finite"
            end
            hasNamedBounds = true
        end
        for _, key in ipairs(xKeys) do
            if result[key] ~= nil then
                if not finite_number(result[key]) then
                    return nil, "normalized bounds x values must be finite"
                end
                hasNamedBounds = true
            end
        end
        for _, key in ipairs(yKeys) do
            if result[key] ~= nil then
                if not finite_number(result[key]) then
                    return nil, "normalized bounds y values must be finite"
                end
                hasNamedBounds = true
            end
        end
        if not hasNamedBounds then
            return nil, "normalized bounds must contain a supported rectangle shape"
        end
        result.scale = {1, 1}
        result.coordinate_space = space
        return result
    end
    if not image_space(space) then
        return nil, "image bounds coordinate space is incompatible with canonical pixels"
    end
    if not valid_scale(scale) then
        return nil, "image bounds require two finite positive scale factors"
    end

    local result = clone_table(bounds)
    local hasNamedBounds = false
    if result.x ~= nil and result.y ~= nil
        and result.width ~= nil and result.height ~= nil then
        if not finite_number(result.x) or not finite_number(result.y)
            or not finite_number(result.width) or not finite_number(result.height) then
            return nil, "rectangle bounds must be finite"
        end
        result.x = result.x * scale[1]
        result.y = result.y * scale[2]
        result.width = result.width * scale[1]
        result.height = result.height * scale[2]
        hasNamedBounds = true
    end
    for _, key in ipairs(xKeys) do
        if result[key] ~= nil then
            if not finite_number(result[key]) then return nil, "bounds x values must be finite" end
            result[key] = result[key] * scale[1]
            hasNamedBounds = true
        end
    end
    for _, key in ipairs(yKeys) do
        if result[key] ~= nil then
            if not finite_number(result[key]) then return nil, "bounds y values must be finite" end
            result[key] = result[key] * scale[2]
            hasNamedBounds = true
        end
    end
    if not hasNamedBounds then
        return nil, "bounds must contain named coordinates or rectangle extents"
    end
    result.scale = {1, 1}
    result.coordinate_space = Coordinates.CANONICAL_SPACE
    return result
end

local function normalize_pixel_dimensions(value, inheritedSpace)
    local unit = value.dimensions_unit
    if value.width_px ~= nil or value.height_px ~= nil then
        return nil, "pixel table extents must use width and height fields"
    end
    if unit == nil then
        return clone_table(value)
    end
    if unit == "meters" or unit == "meter" then
        return clone_table(value)
    end
    if unit ~= "pixels" and unit ~= "canonical_pixels" then
        return nil, "table dimensions unit must be meters or pixels"
    end
    local space = value.coordinate_space or inheritedSpace
    if not image_space(space) then
        return nil, "pixel table dimensions cannot declare a non-image coordinate space"
    end

    local dimensionsScale = value.dimensions_scale
    if dimensionsScale == nil then
        return nil, "pixel table dimensions require an explicit scale"
    end
    if not valid_scale(dimensionsScale) then
        return nil, "table dimensions scale must contain two positive finite factors"
    end

    local result = clone_table(value)
    for _, extent in ipairs({
        {key = "width", factor = dimensionsScale[1]},
        {key = "height", factor = dimensionsScale[2]}
    }) do
        if result[extent.key] ~= nil then
            if not finite_number(result[extent.key]) or result[extent.key] <= 0 then
                return nil, "pixel table extents must be positive and finite"
            end
            result[extent.key] = result[extent.key] * extent.factor
        end
    end
    result.dimensions_scale = {1, 1}
    result.dimensions_unit = "canonical_pixels"
    return result
end

local function normalize_pixel_radii(value)
    local result = clone_table(value)
    local fields = {
        {value = "radius", unit = "radius_unit", scale = "radius_scale"},
        {
            value = "pocket_radius",
            unit = "pocket_radius_unit",
            scale = "pocket_radius_scale"
        }
    }
    for _, fieldsForRadius in ipairs(fields) do
        local amount = value[fieldsForRadius.value]
        local unit = value[fieldsForRadius.unit]
        if unit == "meter" or unit == "meters" then
            -- Physical radii remain in their declared physical units.
        elseif unit == "pixels" or unit == "canonical_pixels" then
            local scale = value[fieldsForRadius.scale]
            if not valid_scale(scale) then
                return nil, "pixel radii require two positive finite scale factors"
            end
            if scale[1] ~= scale[2] then
                return nil, "circular pixel radii cannot use anisotropic scaling"
            end
            if not finite_number(amount) or amount <= 0 then
                return nil, "pixel radii must be positive and finite"
            end
            result[fieldsForRadius.value] = amount * scale[1]
            result[fieldsForRadius.unit] = "canonical_pixels"
            result[fieldsForRadius.scale] = {1, 1}
        elseif amount ~= nil then
            return nil, "radius values must declare meter or pixel units"
        end
    end
    return result
end

local function normalized_plane_point(value, inheritedScale, space)
    local scale = value.scale or inheritedScale
    if not valid_scale(scale) or scale[1] ~= 1 or scale[2] ~= 1 then
        return nil, "normalized table/projector points require identity scale"
    end
    local x, y = xy_values(value)
    if not x or not y then
        return nil, "normalized table/projector point must contain finite coordinates"
    end
    local result = clone_table(value)
    result.x = x
    result.y = y
    result.scale = {1, 1}
    result.coordinate_space = space
    return result
end

--- Normalize known spatial fields in a record, leaving physical measurements
-- (for example table.width in meters and ball.radius in meters) untouched.
function Coordinates.normalizeGeometry(value, inheritedScale, inheritedSpace, fieldName)
    if type(value) ~= "table" then
        return nil, "geometry must be an object or array"
    end
    if value.coordinate_scale ~= nil or value.units ~= nil then
        return nil, "geometry uses unsupported legacy coordinate metadata"
    end

    local unit = value.unit
    local space = value.coordinate_space or inheritedSpace
    if space == nil and (unit == "normalized_projector" or unit == "normalized_table") then
        space = unit
    end
    local x, y = xy_values(value)
    if value.x ~= nil or value.y ~= nil then
        if not finite_number(value.x) or not finite_number(value.y) then
            return nil, "point must contain finite x and y"
        end
    end
    if x and y then
        if physical_units[unit] then
            if space ~= nil and space ~= "world_meters" then
                return nil, "physical meter points cannot declare an image coordinate space"
            end
            if not finite_number(x) or not finite_number(y) then
                return nil, "world point must contain finite x and y"
            end
            return clone_table(value)
        end
        if space == "world_meters" and value.scale == nil then
            return clone_table(value)
        end
        if space == "normalized_projector" or space == "normalized_table" then
            return normalized_plane_point(value, inheritedScale, space)
        end
        return Coordinates.toCanonicalPoint(value, inheritedScale, inheritedSpace)
    end
    if point_fields[fieldName] and type(value[1]) == "number" then
        return nil, "coordinate points must use x and y object fields"
    end

    if fieldName == "bounds" or fieldName == "table_bounds" then
        return scale_bounds(value, inheritedScale, inheritedSpace)
    end

    local dimensionNormalized, dimensionsError = normalize_pixel_dimensions(
        value, inheritedSpace
    )
    if not dimensionNormalized then return nil, dimensionsError end
    local result
    result, dimensionsError = normalize_pixel_radii(dimensionNormalized)
    if not result then return nil, dimensionsError end
    local localScale = value.scale or inheritedScale
    local localSpace = value.coordinate_space or inheritedSpace
    for key, child in pairs(value) do
        if type(child) == "table" and not scale_fields[key] then
            -- Keep the parent geometry field while walking list indices so
            -- every corner or boundary point is validated as an x/y object.
            local childFieldName = type(key) == "number" and fieldName or key
            if key == "bounds" or key == "table_bounds" then
                local normalized, err = scale_bounds(child, localScale, localSpace)
                if not normalized then return nil, err end
                result[key] = normalized
            elseif point_fields[childFieldName] then
                local normalized, err = Coordinates.normalizeGeometry(
                    child, localScale, localSpace, childFieldName
                )
                if not normalized then return nil, err end
                result[key] = normalized
            else
                -- Geometry arrays can be nested (a pocket may hold `position`,
                -- a rail may hold `start`/`end`). Recursing through records
                -- catches those while leaving unrelated table dimensions alone.
                local normalized, err = Coordinates.normalizeGeometry(
                    child, localScale, localSpace, childFieldName
                )
                if normalized then
                    result[key] = normalized
                elseif point_fields[childFieldName]
                    or child.x ~= nil or child.y ~= nil then
                    return nil, err
                end
            end
        end
    end
    return result
end

--- Convert canonical image pixels into the normalized plane used by projector
-- calibration. The calibration itself maps [0,1] x [0,1] to output corners.
function Coordinates.toUnitPlane(point)
    if type(point) ~= "table" or not canonical_space(point.coordinate_space) then
        return nil, nil
    end
    if not valid_scale(point.scale) or point.scale[1] ~= 1 or point.scale[2] ~= 1 then
        return nil, nil
    end
    if not finite_number(point.x) or not finite_number(point.y) then return nil, nil end
    return point.x / Coordinates.CANONICAL_WIDTH,
        point.y / Coordinates.CANONICAL_HEIGHT
end

local function canonical_point(point)
    return type(point) == "table"
        and canonical_space(point.coordinate_space)
        and valid_scale(point.scale)
        and point.scale[1] == 1 and point.scale[2] == 1
        and finite_number(point.x) and finite_number(point.y)
end

local function valid_playing_area(corners)
    if type(corners) ~= "table" or #corners ~= 4 then
        return nil, "table projection requires four playing-area corners"
    end
    local normalized = {}
    for index, corner in ipairs(corners) do
        if not canonical_point(corner) then
            return nil, "playing-area corners must be canonical points with explicit scale"
        end
        if corner.x < 0 or corner.x > Coordinates.CANONICAL_WIDTH
            or corner.y < 0 or corner.y > Coordinates.CANONICAL_HEIGHT then
            return nil, "playing-area corners must lie within the canonical image"
        end
        normalized[index] = {
            x = corner.x / Coordinates.CANONICAL_WIDTH,
            y = corner.y / Coordinates.CANONICAL_HEIGHT
        }
    end

    -- A projective transform cannot represent crossed, concave, or collapsed
    -- corner orderings. Require a strictly convex quadrilateral in the
    -- documented TL, TR, BR, BL order before fitting its matrix.
    local orientation
    for index = 1, 4 do
        local first = normalized[index]
        local second = normalized[index % 4 + 1]
        local third = normalized[(index + 1) % 4 + 1]
        local cross = (second.x - first.x) * (third.y - second.y)
            - (second.y - first.y) * (third.x - second.x)
        if math.abs(cross) <= 1e-12 then
            return nil, "playing-area corners must form a non-degenerate quadrilateral"
        end
        local sign = cross > 0 and 1 or -1
        if orientation and sign ~= orientation then
            return nil, "playing-area corners must form a convex quadrilateral"
        end
        orientation = sign
    end
    return normalized
end

local function solve_homography(corners)
    -- Solve eight projective coefficients from the four canonical corners to
    -- (0,0), (1,0), (1,1), (0,1), using pivoted elimination for stability.
    local target = {{0, 0}, {1, 0}, {1, 1}, {0, 1}}
    local matrix = {}
    for index, source in ipairs(corners) do
        local x, y = source.x, source.y
        local u, v = target[index][1], target[index][2]
        matrix[#matrix + 1] = {
            x, y, 1, 0, 0, 0, -u * x, -u * y, u
        }
        matrix[#matrix + 1] = {
            0, 0, 0, x, y, 1, -v * x, -v * y, v
        }
    end

    for column = 1, 8 do
        local pivot = column
        for row = column + 1, 8 do
            if math.abs(matrix[row][column]) > math.abs(matrix[pivot][column]) then
                pivot = row
            end
        end
        if math.abs(matrix[pivot][column]) <= 1e-12 then
            return nil, "playing-area corner transform is singular"
        end
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        local divisor = matrix[column][column]
        for entry = column, 9 do
            matrix[column][entry] = matrix[column][entry] / divisor
        end
        for row = 1, 8 do
            if row ~= column then
                local factor = matrix[row][column]
                for entry = column, 9 do
                    matrix[row][entry] = matrix[row][entry]
                        - factor * matrix[column][entry]
                end
            end
        end
    end

    local coefficients = {}
    for index = 1, 8 do coefficients[index] = matrix[index][9] end
    return coefficients
end

--- Map a canonical image point through the calibrated playing-area quadrilateral.
-- The point and each corner must already carry explicit canonical scale metadata.
function Coordinates.toTablePlane(point, playingAreaCorners)
    if not canonical_point(point) then
        return nil, nil, "image point must be canonical and explicitly scaled"
    end
    local corners, cornersError = valid_playing_area(playingAreaCorners)
    if not corners then return nil, nil, cornersError end
    local coefficients, transformError = solve_homography(corners)
    if not coefficients then return nil, nil, transformError end

    local x = point.x / Coordinates.CANONICAL_WIDTH
    local y = point.y / Coordinates.CANONICAL_HEIGHT
    local a, b, c, d, e, f, g, h = coefficients[1], coefficients[2],
        coefficients[3], coefficients[4], coefficients[5], coefficients[6],
        coefficients[7], coefficients[8]
    local denominator = g * x + h * y + 1
    if not finite_number(denominator) or math.abs(denominator) <= 1e-12 then
        return nil, nil, "image point maps through a singular table transform"
    end
    local u = (a * x + b * y + c) / denominator
    local v = (d * x + e * y + f) / denominator
    if not finite_number(u) or not finite_number(v) then
        return nil, nil, "table transform produced non-finite coordinates"
    end
    return u, v
end

--- Return physical table dimensions only when their meter units are explicit.
function Coordinates.realWorldDimensions(tableState)
    if type(tableState) ~= "table" then return nil, nil end
    local dimensions = tableState.real_world_dimensions
    local width, height
    if type(dimensions) == "table" then
        if dimensions.unit ~= "meters" then return nil, nil end
        width, height = dimensions.width, dimensions.height
    end
    if not finite_number(width) or width <= 0
        or not finite_number(height) or height <= 0 then
        return nil, nil
    end
    return width, height
end

--- Map a table-relative meter position to its normalized physical rectangle.
-- Require an explicit origin tag even though Core's contract currently uses
-- TABLE_METER_ORIGIN, so an alternate frame cannot be silently misread.
function Coordinates.toPhysicalTablePlane(point, tableState)
    if type(point) ~= "table" or point.unit ~= "meters"
        or not finite_number(point.x) or not finite_number(point.y) then
        return nil, nil, "physical point must be explicitly expressed in meters"
    end
    local dimensions = tableState and tableState.real_world_dimensions
    local origin = point.origin
        or (tableState and tableState.table_meter_origin)
        or (type(dimensions) == "table" and dimensions.origin)
    if origin ~= Coordinates.TABLE_METER_ORIGIN then
        return nil, nil, "physical table-meter origin is required"
    end
    local width, height = Coordinates.realWorldDimensions(tableState)
    if not width or not height then
        return nil, nil, "physical table dimensions in meters are required"
    end
    return point.x / width, point.y / height
end

return Coordinates
