-- Coordinate validation shared by the UDP receiver and projector modules.
-- Table points are expressed in the unit table rectangle. The message-level
-- scale applies to every point, boundary, and normalized extent in that data.

local CoordinateContract = {}

local NORMALIZED_TABLE = "normalized_table"
local NORMALIZED_PROJECTOR = "normalized_projector"
local PROJECTOR_PIXELS = "projector_pixels"
local NORMALIZED_TABLE_VELOCITY = "normalized_table_per_second"
local WORLD_METERS = "world_meters"
local IDENTITY_SCALE = {1, 1}
local POINT_RECORD_KEYS = {
    position = true,
    center = true,
    start = true,
    finish = true,
    tip = true,
    origin = true,
    target = true,
    contact_point = true,
}
local RAW_SPATIAL_FIELDS = {
    x = true, y = true, x1 = true, y1 = true, x2 = true, y2 = true,
    tip_x = true, tip_y = true, position = true, center = true, start = true,
    finish = true, tip = true, origin = true, target = true,
    contact_point = true, bounds = true, boundary = true, boundaries = true,
    table_boundary = true, playing_area = true, corners = true, pockets = true,
    radius = true, width = true, height = true, velocity = true,
}

local function isFiniteNumber(value)
    return type(value) == "number"
        and value == value
        and value ~= math.huge
        and value ~= -math.huge
end

local function isIdentityScale(scale)
    return type(scale) == "table"
        and #scale == 2
        and isFiniteNumber(scale[1])
        and isFiniteNumber(scale[2])
        and scale[1] == IDENTITY_SCALE[1]
        and scale[2] == IDENTITY_SCALE[2]
end

local function isUnitCoordinate(value)
    return isFiniteNumber(value) and value >= 0 and value <= 1
end

CoordinateContract.isFiniteNumber = isFiniteNumber
CoordinateContract.isUnitCoordinate = isUnitCoordinate

local function fail(message)
    return false, message
end

function CoordinateContract.validateMetadata(space, scale, expectedSpace)
    if space ~= expectedSpace then
        return fail("coordinate_space must be " .. expectedSpace)
    end
    if not isIdentityScale(scale) then
        return fail("coordinate_scale must be the identity scale [1, 1]")
    end
    return true
end

function CoordinateContract.validatePoint(point, label)
    label = label or "point"
    if type(point) ~= "table" then
        return fail(label .. " must be an object")
    end
    if point.scale ~= nil then
        return fail(label .. ".scale is unsupported; use the message-level coordinate_scale")
    end
    if point.unit ~= nil then
        return fail(label .. ".unit must not override the normalized-table message space")
    end
    if point.coordinate_space ~= nil or point.coordinate_scale ~= nil then
        local metadataValid, metadataError = CoordinateContract.validateMetadata(
            point.coordinate_space,
            point.coordinate_scale,
            NORMALIZED_TABLE
        )
        if not metadataValid then
            return false, label .. ": " .. metadataError
        end
    end
    if not isUnitCoordinate(point.x) or not isUnitCoordinate(point.y) then
        return fail(label .. " x and y must be finite normalized coordinates in [0, 1]")
    end
    return true
end

local function validatePointList(points, label, minimumCount)
    if type(points) ~= "table" or #points < minimumCount then
        return fail(label .. " must contain at least " .. minimumCount .. " points")
    end
    for index, point in ipairs(points) do
        local valid, errorMessage = CoordinateContract.validatePoint(point, label .. "[" .. index .. "]")
        if not valid then
            return false, errorMessage
        end
    end
    return true
end

function CoordinateContract.validateBounds(bounds, label)
    label = label or "bounds"
    if type(bounds) == "table" then
        if bounds.scale ~= nil then
            return fail(label .. ".scale is unsupported; use the message-level coordinate_scale")
        end
        if bounds.unit ~= nil then
            return fail(label .. ".unit must not override normalized-table bounds")
        end
        if bounds.coordinate_space ~= nil or bounds.coordinate_scale ~= nil then
            local metadataValid, metadataError = CoordinateContract.validateMetadata(
                bounds.coordinate_space,
                bounds.coordinate_scale,
                NORMALIZED_TABLE
            )
            if not metadataValid then
                return false, label .. ": " .. metadataError
            end
        end
    end
    if type(bounds) ~= "table"
        or not isUnitCoordinate(bounds.x)
        or not isUnitCoordinate(bounds.y)
        or not isFiniteNumber(bounds.width)
        or not isFiniteNumber(bounds.height)
        or bounds.width <= 0
        or bounds.height <= 0
        or bounds.x + bounds.width > 1
        or bounds.y + bounds.height > 1 then
        return fail(label .. " must fit inside normalized table space with positive finite extents")
    end
    return true
end

function CoordinateContract.validateProjectorCorners(corners, space, scale)
    local valid, errorMessage = CoordinateContract.validateMetadata(space, scale, NORMALIZED_PROJECTOR)
    if not valid then
        return false, errorMessage
    end
    if type(corners) ~= "table" or #corners ~= 4 then
        return fail("projector calibration corners must contain exactly four points")
    end
    return validatePointList(corners, "projector calibration corners", 4)
end

local function validateOptionalTableBoundary(data)
    if data.table_boundary == nil then
        return true
    end
    if type(data.table_boundary) ~= "table" or #data.table_boundary ~= 4 then
        return fail("table_boundary must contain exactly four points")
    end
    return validatePointList(data.table_boundary, "table_boundary", 4)
end

local function validateOptionalBounds(data)
    if data.bounds == nil then
        return true
    end
    return CoordinateContract.validateBounds(data.bounds)
end

local function validateProjectorPixelExtent(owner, label)
    if not isFiniteNumber(owner.radius) or owner.radius <= 0 then
        return fail(label .. ".radius must be a positive finite projector-pixel extent")
    end
    if owner.radius_space ~= PROJECTOR_PIXELS then
        return fail(label .. ".radius_space must be projector_pixels")
    end
    if not isIdentityScale(owner.radius_scale) then
        return fail(label .. ".radius_scale must be the identity scale [1, 1]")
    end
    return true
end

local function validateOptionalRecordMetadata(owner, label)
    if owner.coordinate_space == nil and owner.coordinate_scale == nil then
        return true
    end
    if owner.coordinate_space == WORLD_METERS then
        if owner.unit ~= "meters" or owner.coordinate_scale ~= nil then
            return fail(label .. " world_meters coordinates require unit=meters and no image scale")
        end
        return true
    end
    local valid, errorMessage = CoordinateContract.validateMetadata(
        owner.coordinate_space,
        owner.coordinate_scale,
        NORMALIZED_TABLE
    )
    if not valid then
        return false, label .. ": " .. errorMessage
    end
    return true
end

local function validatePhysicalPoint(point, label)
    if type(point) ~= "table" or point.unit ~= "meters"
        or (point.coordinate_space ~= nil and point.coordinate_space ~= WORLD_METERS)
        or point.coordinate_scale ~= nil or point.scale ~= nil
        or not isFiniteNumber(point.x) or not isFiniteNumber(point.y) then
        return fail(label .. " physical coordinates require finite x/y, unit=meters, and no image scale")
    end
    return true
end

local function validatePhysicalDimensions(dimensions, label)
    if type(dimensions) ~= "table" or dimensions.unit ~= "meters"
        or not isFiniteNumber(dimensions.width) or dimensions.width <= 0
        or not isFiniteNumber(dimensions.height) or dimensions.height <= 0 then
        return fail(label .. " requires positive finite width/height with unit=meters")
    end
    if dimensions.coordinate_space ~= nil and dimensions.coordinate_space ~= WORLD_METERS then
        return fail(label .. ".coordinate_space must be world_meters when present")
    end
    if dimensions.coordinate_scale ~= nil or dimensions.extent_scale ~= nil or dimensions.scale ~= nil then
        return fail(label .. " physical dimensions must not carry an image scale")
    end
    return true
end

local function validatePhysicalRadius(owner, label)
    if owner.radius_unit ~= "meters"
        or not isFiniteNumber(owner.radius) or owner.radius <= 0
        or owner.radius_space ~= nil or owner.radius_scale ~= nil then
        return fail(label .. " physical radius requires a positive finite value with radius_unit=meters and no image scale")
    end
    return true
end

local function validateTableExtent(owner, label)
    if not isFiniteNumber(owner.width) or owner.width <= 0
        or not isFiniteNumber(owner.height) or owner.height <= 0 then
        return fail(label .. " width and height must be positive finite extents")
    end
    if owner.extent_space == NORMALIZED_TABLE then
        if owner.width > 1 or owner.height > 1 then
            return fail(label .. " normalized-table extents must be in (0, 1]")
        end
    elseif owner.extent_space ~= PROJECTOR_PIXELS then
        return fail(label .. ".extent_space must be normalized_table or projector_pixels")
    end
    if not isIdentityScale(owner.extent_scale) then
        return fail(label .. ".extent_scale must be the identity scale [1, 1]")
    end
    return true
end

local function validateTableVelocity(velocity, label)
    if type(velocity) ~= "table"
        or not isFiniteNumber(velocity.x)
        or not isFiniteNumber(velocity.y) then
        return fail(label .. " must contain finite x and y components")
    end
    if velocity.velocity_space ~= NORMALIZED_TABLE_VELOCITY then
        return fail(label .. ".velocity_space must be normalized_table_per_second")
    end
    if not isIdentityScale(velocity.velocity_scale) then
        return fail(label .. ".velocity_scale must be the identity scale [1, 1]")
    end
    return true
end

local function hasSpatialFields(value, seen)
    if type(value) ~= "table" then
        return false
    end
    seen = seen or {}
    if seen[value] then
        return false
    end
    seen[value] = true
    if value.x ~= nil or value.y ~= nil or value.x1 ~= nil or value.y1 ~= nil
        or value.x2 ~= nil or value.y2 ~= nil or value.tip_x ~= nil or value.tip_y ~= nil
        or value.position ~= nil or value.center ~= nil or value.start ~= nil
        or value.finish ~= nil or value.bounds ~= nil or value.table_boundary ~= nil
        or value.boundary ~= nil or value.boundaries ~= nil
        or value.playing_area ~= nil or value.corners ~= nil or value.pockets ~= nil
        or value.radius ~= nil or value.width ~= nil or value.height ~= nil
        or value.velocity ~= nil then
        return true
    end
    for _, child in pairs(value) do
        if hasSpatialFields(child, seen) then
            return true
        end
    end
    return false
end

local function hasUnexpectedSpatialFields(value, allowedRootFields, isRoot)
    if type(value) ~= "table" then
        return false
    end
    for key, child in pairs(value) do
        if not (isRoot and allowedRootFields and allowedRootFields[key]) then
            if RAW_SPATIAL_FIELDS[key] then
                return true
            end
            if type(child) == "table" and hasUnexpectedSpatialFields(child, nil, false) then
                return true
            end
        end
    end
    return false
end

local validateTablePoints

local function validateSpatialTree(value, label, seen)
    if type(value) ~= "table" then
        return true
    end
    seen = seen or {}
    if seen[value] then
        return fail(label .. " contains a cyclic object")
    end
    seen[value] = true

    local valid, errorMessage = validateOptionalRecordMetadata(value, label)
    if not valid then
        return false, errorMessage
    end
    if value.x ~= nil or value.y ~= nil then
        if value.unit == "meters" or value.coordinate_space == WORLD_METERS then
            valid, errorMessage = validatePhysicalPoint(value, label)
        else
            valid, errorMessage = CoordinateContract.validatePoint(value, label)
        end
        if not valid then
            return false, errorMessage
        end
    end
    if value.x1 ~= nil or value.y1 ~= nil or value.x2 ~= nil or value.y2 ~= nil then
        for _, coordinate in ipairs({"x1", "y1", "x2", "y2"}) do
            if not isUnitCoordinate(value[coordinate]) then
                return fail(label .. "." .. coordinate .. " must be a finite normalized coordinate in [0, 1]")
            end
        end
    end
    if value.tip_x ~= nil or value.tip_y ~= nil then
        if not isUnitCoordinate(value.tip_x) or not isUnitCoordinate(value.tip_y) then
            return fail(label .. " tip_x and tip_y must be finite normalized coordinates in [0, 1]")
        end
    end
    for key in pairs(POINT_RECORD_KEYS) do
        local point = value[key]
        if point ~= nil and (type(point) ~= "table" or point.x == nil or point.y == nil) then
            return fail(label .. "." .. key .. " must be an x/y point object")
        end
    end
    if value.table_boundary ~= nil then
        valid, errorMessage = validateTablePoints(value.table_boundary, label .. ".table_boundary", 4)
        if not valid then
            return false, errorMessage
        end
    end
    if value.boundary ~= nil then
        valid, errorMessage = validateTablePoints(value.boundary, label .. ".boundary", 4)
        if not valid then
            return false, errorMessage
        end
    end
    if value.corners ~= nil then
        valid, errorMessage = validateTablePoints(value.corners, label .. ".corners", 4)
        if not valid then
            return false, errorMessage
        end
    end
    if value.bounds ~= nil then
        valid, errorMessage = CoordinateContract.validateBounds(value.bounds, label .. ".bounds")
        if not valid then
            return false, errorMessage
        end
    end
    if value.playing_area ~= nil then
        valid, errorMessage = CoordinateContract.validateBounds(value.playing_area, label .. ".playing_area")
        if not valid then
            return false, errorMessage
        end
    end
    if value.radius ~= nil then
        if value.radius_unit == "meters" then
            valid, errorMessage = validatePhysicalRadius(value, label)
        else
            valid, errorMessage = validateProjectorPixelExtent(value, label)
        end
        if not valid then
            return false, errorMessage
        end
    end
    if value.unit ~= nil and value.unit ~= "meters" then
        return fail(label .. ".unit is unsupported; spatial units must be declared explicitly")
    end
    if value.real_world_dimensions ~= nil then
        valid, errorMessage = validatePhysicalDimensions(value.real_world_dimensions, label .. ".real_world_dimensions")
        if not valid then
            return false, errorMessage
        end
    end
    if label:match("%.real_world_dimensions$") then
        return validatePhysicalDimensions(value, label)
    end
    if value.width ~= nil or value.height ~= nil then
        local isBounds = label:match("%.bounds$") or label:match("%.playing_area$")
        local isPhysicalDimensions = label:match("%.real_world_dimensions$")
        if not isBounds and not isPhysicalDimensions then
            valid, errorMessage = validateTableExtent(value, label)
            if not valid then
                return false, errorMessage
            end
        end
    end
    if value.velocity ~= nil then
        valid, errorMessage = validateTableVelocity(value.velocity, label .. ".velocity")
        if not valid then
            return false, errorMessage
        end
    end

    for key, child in pairs(value) do
        if key ~= "velocity" and type(child) == "table" then
            if key == "real_world_dimensions" then
                valid, errorMessage = validatePhysicalDimensions(child, label .. ".real_world_dimensions")
                if not valid then
                    return false, errorMessage
                end
            else
                valid, errorMessage = validateSpatialTree(child, label .. "." .. tostring(key), seen)
                if not valid then
                    return false, errorMessage
                end
            end
        end
    end
    seen[value] = nil
    return true
end

validateTablePoints = function(points, label, exactCount)
    if type(points) ~= "table" or (exactCount and #points ~= exactCount) then
        local countDescription = exactCount and ("exactly " .. exactCount) or "one or more"
        return fail(label .. " must contain " .. countDescription .. " points")
    end
    for index, point in ipairs(points) do
        local valid, errorMessage = CoordinateContract.validatePoint(point, label .. "[" .. index .. "]")
        if not valid then
            return false, errorMessage
        end
    end
    return true
end

local function validateCommonTableCoordinates(data)
    if type(data) ~= "table" then
        return fail("data must be an object")
    end
    local valid, errorMessage = CoordinateContract.validateMetadata(
        data.coordinate_space,
        data.coordinate_scale,
        NORMALIZED_TABLE
    )
    if not valid then
        return false, errorMessage
    end
    valid, errorMessage = validateOptionalTableBoundary(data)
    if not valid then
        return false, errorMessage
    end
    return validateOptionalBounds(data)
end

function CoordinateContract.validateTrajectory(data)
    local valid, errorMessage = validateCommonTableCoordinates(data)
    if not valid then
        return false, errorMessage
    end

    if data.paths ~= nil then
        if type(data.paths) ~= "table" then
            return fail("paths must be an array")
        end
        for pathIndex, path in ipairs(data.paths) do
            if type(path) ~= "table" then
                return fail("paths[" .. pathIndex .. "] must be an object")
            end
            valid, errorMessage = validatePointList(path.points, "paths[" .. pathIndex .. "].points", 1)
            if not valid then
                return false, errorMessage
            end
        end
    end

    if data.collisions ~= nil then
        if type(data.collisions) ~= "table" then
            return fail("collisions must be an array")
        end
        for index, collision in ipairs(data.collisions) do
            valid, errorMessage = CoordinateContract.validatePoint(collision, "collisions[" .. index .. "]")
            if not valid then
                return false, errorMessage
            end
        end
    end

    if data.ghostBalls ~= nil then
        if type(data.ghostBalls) ~= "table" then
            return fail("ghostBalls must be an array")
        end
        for index, ghost in ipairs(data.ghostBalls) do
            local label = "ghostBalls[" .. index .. "]"
            valid, errorMessage = CoordinateContract.validatePoint(ghost, label)
            if not valid then
                return false, errorMessage
            end
            valid, errorMessage = validateProjectorPixelExtent(ghost, label)
            if not valid then
                return false, errorMessage
            end
        end
    end

    return validateSpatialTree(data, "trajectory")
end

function CoordinateContract.validateCollision(data)
    local valid, errorMessage = validateCommonTableCoordinates(data)
    if not valid then
        return false, errorMessage
    end
    valid, errorMessage = CoordinateContract.validatePoint(data, "collision")
    if not valid then
        return false, errorMessage
    end
    return validateSpatialTree(data, "collision")
end

function CoordinateContract.validateAim(data)
    local valid, errorMessage = validateCommonTableCoordinates(data)
    if not valid then
        return false, errorMessage
    end
    for _, key in ipairs({"x1", "y1", "x2", "y2"}) do
        if not isUnitCoordinate(data[key]) then
            return fail("aim " .. key .. " must be a finite normalized coordinate in [0, 1]")
        end
    end
    return validateSpatialTree(data, "aim")
end

function CoordinateContract.validateState(data)
    if type(data) ~= "table" then
        return fail("data must be an object")
    end
    local containsGeometry = data.balls ~= nil or data.cue ~= nil or data.table ~= nil
        or hasSpatialFields(data)
    if not containsGeometry then
        return true
    end

    local valid, errorMessage = validateCommonTableCoordinates(data)
    if not valid then
        return false, errorMessage
    end

    if data.balls ~= nil and type(data.balls) ~= "table" then
        return fail("balls must be an array")
    end
    if data.cue ~= nil and type(data.cue) ~= "table" then
        return fail("cue must be an object")
    end
    if data.table ~= nil and type(data.table) ~= "table" then
        return fail("table must be an object")
    end
    if data.table and data.table.corners ~= nil then
        valid, errorMessage = validateTablePoints(data.table.corners, "table.corners", 4)
        if not valid then
            return false, errorMessage
        end
    end
    if data.table and data.table.pockets ~= nil then
        if type(data.table.pockets) ~= "table" then
            return fail("table.pockets must be a list")
        end
        for index, pocket in ipairs(data.table.pockets) do
            local label = "table.pockets[" .. index .. "]"
            if type(pocket) ~= "table" then
                return fail(label .. " must be an object")
            end
            if pocket.x ~= nil or pocket.y ~= nil then
                valid, errorMessage = validateSpatialTree(pocket, label)
            elseif type(pocket.center) == "table" then
                valid, errorMessage = validateSpatialTree(pocket, label)
            else
                return fail(label .. " must contain an x/y point or center point")
            end
            if not valid then
                return false, errorMessage
            end
        end
    end
    return validateSpatialTree(data, "state")
end

function CoordinateContract.validateMotion(data)
    if type(data) ~= "table" then
        return fail("data must be an object")
    end
    local valid, errorMessage = validateCommonTableCoordinates(data)
    if not valid then
        return false, errorMessage
    end
    if type(data.position) ~= "table" then
        return fail("motion.position must be an x/y object")
    end
    valid, errorMessage = CoordinateContract.validatePoint(data.position, "motion.position")
    if not valid then
        return false, errorMessage
    end
    valid, errorMessage = validateTableVelocity(data.velocity, "motion.velocity")
    if not valid then
        return false, errorMessage
    end
    return validateSpatialTree(data, "motion")
end

function CoordinateContract.validateChessboardCommand(data, show)
    if data == nil then
        data = {}
    end
    if type(data) ~= "table" then
        return fail("data must be an object")
    end
    if not show then
        if hasUnexpectedSpatialFields(data, nil, true) then
            return fail("hide calibration command does not accept coordinate fields")
        end
        return true
    end
    if hasUnexpectedSpatialFields(data, {
        coordinate_space = true,
        coordinate_scale = true,
        rows = true,
        cols = true,
        squareSize = true,
        offsetX = true,
        offsetY = true,
        centered = true,
    }, true) then
        return fail("show calibration command contains unsupported coordinate geometry")
    end
    local valid, errorMessage = CoordinateContract.validateMetadata(
        data.coordinate_space,
        data.coordinate_scale,
        PROJECTOR_PIXELS
    )
    if not valid then
        return false, errorMessage
    end
    if data.squareSize ~= nil and (not isFiniteNumber(data.squareSize) or data.squareSize <= 0) then
        return fail("squareSize must be a positive finite projector-pixel extent")
    end
    if data.offsetX ~= nil and not isFiniteNumber(data.offsetX) then
        return fail("offsetX must be a finite projector-pixel coordinate")
    end
    if data.offsetY ~= nil and not isFiniteNumber(data.offsetY) then
        return fail("offsetY must be a finite projector-pixel coordinate")
    end
    if (data.offsetX == nil) ~= (data.offsetY == nil) then
        return fail("offsetX and offsetY must be supplied together")
    end
    return true
end

function CoordinateContract.validateFrame(data)
    if type(data) ~= "table" then
        return fail("data must be an object")
    end
    if hasUnexpectedSpatialFields(data, {
        width = true,
        height = true,
        dimensions_space = true,
        dimensions_scale = true,
        image = true,
        fps = true,
        latency = true,
    }, true) then
        return fail("frame message contains unsupported coordinate geometry")
    end
    if data.width == nil and data.height == nil then
        return true
    end
    if not isFiniteNumber(data.width) or data.width <= 0
        or not isFiniteNumber(data.height) or data.height <= 0 then
        return fail("frame width and height must be positive finite image extents")
    end
    if data.dimensions_space ~= "image_pixels" or not isIdentityScale(data.dimensions_scale) then
        return fail("frame dimensions require dimensions_space=image_pixels and identity dimensions_scale")
    end
    return true
end

function CoordinateContract.validateMessage(messageType, data)
    if messageType == "trajectory" then
        return CoordinateContract.validateTrajectory(data)
    elseif messageType == "collision" then
        return CoordinateContract.validateCollision(data)
    elseif messageType == "aim" then
        return CoordinateContract.validateAim(data)
    elseif messageType == "state" then
        return CoordinateContract.validateState(data)
    elseif messageType == "motion" then
        return CoordinateContract.validateMotion(data)
    elseif messageType == "show_calibration_chessboard" then
        return CoordinateContract.validateChessboardCommand(data, true)
    elseif messageType == "hide_calibration_chessboard" then
        return CoordinateContract.validateChessboardCommand(data, false)
    elseif messageType == "frame" then
        return CoordinateContract.validateFrame(data)
    elseif messageType == "event" then
        if type(data) ~= "table" then
            return fail("data must be an object")
        end
        if not hasSpatialFields(data) then
            return true
        end
        local valid, errorMessage = validateCommonTableCoordinates(data)
        if not valid then
            return false, errorMessage
        end
        return validateSpatialTree(data, "event")
    end
    -- Preserve non-spatial application messages, but prevent unfamiliar types
    -- from bypassing the explicit normalized-table coordinate contract.
    if hasSpatialFields(data) then
        local valid, errorMessage = validateCommonTableCoordinates(data)
        if not valid then
            return false, errorMessage
        end
        return validateSpatialTree(data, tostring(messageType))
    end
    return true
end

return CoordinateContract
