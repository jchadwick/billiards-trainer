study /(backend|frontend)/<module>/SPECS.md to learn about the project specifications and /PLAN.md to understand plan so far.

The source code of the backend modules is in backend/<module>
The source code of the frontend is in frontend/<module>

Use up to 10 subagents to study existing source code in /(backend|frontend)/<module> and compare it against the relevant specifications for that module. From that update @PLAN.md which is a bullet point list sorted in priority of the items which have yet to be implemeneted. Think extra hard and use the oracle to plan. Consider searching for TODO, minimal implementations and placeholders. Study @PLAN.md to determine starting point for research and keep it up to date with items considered complete/incomplete using subagents.

Second task, once the plan is complete, is to implement the plan in order of priority. Use up to 50 subagents to implement the plan. Each subagent should be assigned a specific task from the plan and should report back when complete. The main agent keeps track of progress and must update @PLAN.md as tasks are completed.

When you're all done your changes, commit all changes with a detailed commit message explaining what was done and why.  If this doesn't work, use up to 50 subagents to fix any issues.  Repeat until everything works.

ULTIMATE GOAL we want to achieve a working version of a billiards trainer system which views a billiards table and the balls and cues on it, detects the position of the balls on the table, track their movement, and provide feedback to the user through the projector augmented reality and configuration via the web interface.
