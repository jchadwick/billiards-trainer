The source code of the backend modules is in backend/<module>

I am using `python tools/video_debugger.py demo2.mkv --background empty_table.jpg` to refine the ball detection and cue detection/trajectory algorithms, so focus on using that to test changes, but ALL changes should be made in the backend/vision/ folder so that they are part of the production code, not just the debugging tool.

There will only ever be one cue on the table at a time.
There will only ever be a maximum of 15 colored balls + 1 cue ball (16 total balls) on the table at a time.
There will only ever be a maximum of 6 pockets on the table at a time.
The table will always be a standard rectangular billiards table with 6 pockets.
The table will always be viewed from a fixed overhead camera angle.
The table will always be fully visible in the camera frame.
