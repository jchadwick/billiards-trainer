-- Test runner for contrast module
local contrast = require('modules.colors.contrast')

function love.load()
  -- Run tests
  local success = contrast.runTests()

  -- Exit with appropriate code
  if success then
    print("\nTests completed successfully!")
    love.event.quit(0)
  else
    print("\nTests failed!")
    love.event.quit(1)
  end
end

function love.draw()
  -- No drawing needed for tests
end
