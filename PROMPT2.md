The source code of the backend modules is in backend/<module>
The source code of the frontend is in frontend/<module>

study /backend/<module>/SPECS.md to learn about the project specifications and /PLAN.md to understand the plan so far.

IMPORTANT: IGNORE THE FRONTEND FOR NOW. RIGHT NOW WE ONLY WANT TO GET THE BACKEND WORKING, SPECIFICALLY THE CAMERA CALIBRATION AND BALL/CUE DETECTION. Once that is done, we can move on to the frontend.

STEP 1: Determine a plan for achieving the "ULTIMATE GOAL" below. Use up to 10 subagents to research and outline the necessary steps in /PLAN.md. Leverage the oracle to ensure the plan is comprehensive and feasible. Consider searching for TODOs, minimal implementations, and placeholders in the existing codebase. Update /PLAN.md with a prioritized bullet point list of tasks that need to be implemented.

STEP 2: Implement the plan in order of priority. Use up to 50 subagents to execute the tasks outlined in /PLAN.md. Each subagent should be assigned a specific task from the plan and report back upon completion. The main agent is responsible for tracking progress and updating /PLAN.md as tasks are completed.

When you're all done your changes, commit all changes with a detailed commit message explaining what was done and why.  If this doesn't work, use up to 50 subagents to fix any issues.  Repeat until everything works.

ULTIMATE GOAL: we want to achieve a working version of a billiards trainer system which views a billiards table and the balls and cues on it, detects the position of the balls on the table, track their movement.
