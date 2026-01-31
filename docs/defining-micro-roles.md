# Defining "Micro-Roles" for an MCP Server in MAKER

Based on the "Massively Decomposed Agentic Processes" (MDAP) framework described in the paper, a developer would not define "micro-roles" for an MCP server by creating personas (e.g., "You are a Senior Python Engineer"). Instead, the developer would define roles strictly by **the specific atomic subtask the agent must perform**.

Here is how a developer would define these micro-roles using the principles from the paper:

### 1. Define Roles by Granularity (The $m=1$ Rule)
The developer must define the role such that the agent is responsible for the "smallest possible subtask" ($m=1$), effectively treating the agent as a mechanical function rather than a reasoned thinker,.
*   **The Definition:** A micro-role is defined as a function that executes exactly **one step**.
*   **Coding Example:** Instead of a role defined as "Write a Function," the developer would define a micro-role as "Write the Function Signature" or "Write the Next Line of Logic."
*   **Paper Logic:** This "extreme decomposition" allows the agent to focus entirely on a single step, which is essential for scaling to millions of steps without error.

### 2. Define Roles by Minimal Context
The developer must design the "micro-role" to receive only the specific slice of information required to complete its single step, shielding it from the full history which causes confusion.
*   **The Definition:** The role is defined by a templating function ($\phi$) that maps the current state ($x_i$) to a prompt containing *only* the necessary context.
*   **Coding Example:** If the micro-role is "Write Line 15," the MCP server should not provide the entire project directory. It should provide only the relevant variable definitions and the immediately preceding lines (Lines 10–14).
*   **Paper Logic:** "An agent’s context is limited to an amount of information sufficient to execute its single assigned step... to avoid confusion that can creep in from irrelevant context".

### 3. Define Roles by Input/Output Requirements (State Passing)
In the MAKER system, a role is not just about outputting text; it is about outputting the **new state** of the system.
*   **The Definition:** The developer must define the role to output both the **Action** (the code to be written) and the **Next State** (the updated file content or AST).
*   **Coding Example:** A micro-role's definition would require it to return: `{"action": "added_variable_x", "current_code_state": "def func():\n    x = 1"}`.
*   **Paper Logic:** Unlike single-agent setups that produce a list of moves, "each agent must produce the resulting state, since this is critical information to be fed to the next agent".

### 4. Separate "Insight" Roles from "Execution" Roles
The developer should distinguish between agents that plan the structure and agents that write the code.
*   **Decomposition Agents (Insight):** Defined to break a complex request (e.g., "Create a login page") into a sequence of atomic subtasks (e.g., "1. Create HTML file," "2. Add head tag," "3. Add body tag"),.
*   **Solver Agents (Execution):** Defined to execute one of those atomic subtasks without further decomposition.
*   **Paper Logic:** This separation isolates the "execution" capability (which can be done by smaller, cheaper models) from the "insight" capability (which requires reasoning models),.

### 5. Define "Red-Flagging" Constraints
The developer must define the micro-role with strict constraints that act as "red flags" to trigger automatic rejection.
*   **The Definition:** The role definition includes strict formatting and token limits. If the output violates these (e.g., it is too long or uses the wrong syntax), the system assumes the agent is confused and discards the result.
*   **Coding Example:** If a micro-role is defined to write *one line of code*, the developer sets a red flag for any response longer than 50 tokens. If the agent writes a whole paragraph explaining the code, it is flagged and discarded.

### Summary of an MCP Micro-Role Definition
To implement this in an MCP server, a developer would define a micro-role not as a system prompt, but as a rigid **Input/Output contract**:
1.  **Input:** The global strategy + The immediate previous state (e.g., current code block).
2.  **Task:** Perform exactly one operation (e.g., add one closing brace).
3.  **Output:** The strictly formatted action + The updated state.
4.  **Failure Condition:** Output > $N$ tokens or invalid syntax (Red Flag).
