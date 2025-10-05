-- LÖVE2D Configuration for Billiards Projector
-- This file configures the LÖVE2D engine settings

function love.conf(t)
    t.identity = "billiards-projector"              -- Save directory name
    t.appendidentity = false                        -- Search files in source directory before save directory
    t.version = "11.5"                              -- LÖVE2D version this game was made for
    t.console = false                               -- Attach a console (Windows only)
    t.accelerometerjoystick = false                 -- Enable accelerometer on iOS and Android
    t.externalstorage = false                       -- Android only
    t.gammacorrect = false                          -- Enable gamma correction

    t.audio.mic = false                             -- Request microphone permission
    t.audio.mixwithsystem = true                    -- Keep background music playing

    -- Check for windowed mode flag (from env var or command line arg)
    local windowed = os.getenv("PROJECTOR_WINDOWED") == "true"

    t.window.title = "Billiards Projector"          -- Window title
    t.window.icon = nil                             -- Filepath to an image to use as the window's icon

    -- Scale down to 75% when in windowed mode for easier viewing on dev machine
    if windowed then
        t.window.width = 1440                       -- 1920 * 0.75
        t.window.height = 810                       -- 1080 * 0.75
    else
        t.window.width = 1920                       -- Full resolution for projector
        t.window.height = 1080
    end
    t.window.borderless = false                     -- Remove window border
    t.window.resizable = false                      -- Let the window be user-resizable
    t.window.minwidth = 1                          -- Minimum window width
    t.window.minheight = 1                          -- Minimum window height
    t.window.fullscreen = not windowed              -- Enable fullscreen unless windowed flag is set
    t.window.fullscreentype = "exclusive"           -- Choose between "desktop" fullscreen or "exclusive" fullscreen mode
    t.window.vsync = 1                              -- Vertical sync mode (1 = on, 0 = off, -1 = adaptive)
    t.window.msaa = 4                               -- Number of samples to use with multi-sampled antialiasing
    t.window.depth = nil                            -- Number of bits per sample in the depth buffer
    t.window.stencil = nil                          -- Number of bits per sample in the stencil buffer
    t.window.display = 1                            -- Index of the monitor to show the window in
    t.window.highdpi = false                        -- Enable high-dpi mode (Retina display on Mac)
    t.window.usedpiscale = true                     -- Enable automatic DPI scaling
    t.window.x = nil                                -- X coordinate of window position
    t.window.y = nil                                -- Y coordinate of window position

    t.modules.audio = false                         -- Disable audio (not needed for projector)
    t.modules.data = true                           -- Enable data module
    t.modules.event = true                          -- Enable event module
    t.modules.font = true                           -- Enable font module
    t.modules.graphics = true                       -- Enable graphics module
    t.modules.image = true                          -- Enable image module
    t.modules.joystick = false                      -- Disable joystick
    t.modules.keyboard = true                       -- Enable keyboard
    t.modules.math = true                           -- Enable math module
    t.modules.mouse = true                          -- Enable mouse
    t.modules.physics = false                       -- Disable physics (using custom physics)
    t.modules.sound = false                         -- Disable sound
    t.modules.system = true                         -- Enable system module
    t.modules.thread = true                         -- Enable thread module
    t.modules.timer = true                          -- Enable timer module
    t.modules.touch = false                         -- Disable touch
    t.modules.video = false                         -- Disable video
    t.modules.window = true                         -- Enable window module
end
