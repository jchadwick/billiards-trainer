# Agent Instructions

<quality>
!!!IMPORTANT!!! ALWAYS MAINTAIN A FUNCTIONING CODEBASE. THIS IS YOUR HIGHEST PRIORITY.

Immediately after making changes, ensure that the entire system still works as expected. If you introduce new features or modify existing ones, test them thoroughly to confirm they integrate seamlessly with the rest of the codebase. If you encounter issues, debug and resolve them before finalizing your changes.

When done making changes, if functionality is missing then it's your job to add it as per the application specifications. Think hard.

If tests unrelated to your work fail then it's your job to resolve these tests as part of the increment of change.

9999999999999999999999999999. DO NOT IMPLEMENT PLACEHOLDER OR SIMPLE IMPLEMENTATIONS. WE WANT FULL IMPLEMENTATIONS. DO IT OR I WILL YELL AT YOU

Leverage automation tools extensively, especially automated tests. Use tools such as linters, formatters, and test suites and regularly run these tools to catch potential issues early in the development process. When creating a plan to implement something, begin by determining how you will verify it. Your task is not complete unless it has been verified by these tools.

Be generous with comments and documentation. Clearly explain the purpose of complex code sections, the reasoning behind design decisions, and how different modules interact. Prefer inline comments, followed by external documentation. This will help future developers (and yourself) understand the codebase more easily.

ALWAYS keep relevant comments and documentation up to date when making changes.
</quality>

<focus>
Focus on one task at a time. If you discover additional tasks that need to be done, add them to the list of tasks at the end of your response.

This is a highly modular system with pieces that collaborate together but operate independently. Only modify one module (e.g. top-level folder) at a time. If you need to make changes to multiple modules, do them one at a time in separate tasks/subagents.
</focus>

<agents>
Leverage subagents extensively.  Definitely use them to handle complex or multi-step tasks.  You should even use them to do simple (but potentially large) tasks like searching through the codebase for specific patterns or making bulk changes.

Use a subagent for each task. Each subagent should have a clear purpose and operate independently. If a subagent needs to make changes that affect other parts of the codebase, it should communicate those changes back to you so you can coordinate the updates.

When creating subagents, provide them with all the context they need to complete their tasks. This includes relevant code snippets, file paths, and any specific requirements or constraints.
</agents>
