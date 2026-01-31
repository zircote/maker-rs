
# Coordinating "Insight" and "Execution" Agents in MAKER

In the framework described by the paper, "insight" and "execution" agents coordinate through a recursive pipeline where high-level planning (insight) is treated as a distinct step that prepares the context for deterministic action (execution).

**1. Distinguishing Roles**
The paper defines the two behaviors as follows:
*   **Insight Agents:** Responsible for "creatively generating ideas, plans, and strategies". In a fully automated system, these are the **Decomposition Agents**, which receive a task and propose a way to break it into "simpler sub-tasks and a composition function".
*   **Execution Agents:** Responsible for "following through" with established plans. These are the **Problem Solver Agents**, which are assigned to "solve minimal subtasks without decomposing them" once the insight agents have broken the problem down sufficiently.

**2. Coordination via Recursive Decomposition**
Coordination is achieved by treating the creation of a subtask (insight) as a "step" in the overall decomposition process, rather than a separate phase.
*   **The Handoff:** The "insight" agents do not simply pass instructions to "execution" agents. Instead, they produce a structural breakdown (e.g., splitting a math problem into two smaller multiplications) which is then recursively processed until the tasks are atomic.
*   **The Validation Layer:** To ensure the insight agents provide valid instructions to the execution agents, the system employs **Decomposition Discriminator Agents**. These agents vote on the proposed decomposition plans (using the "first-to-ahead-by-$k$" method) to select the best strategy before any execution takes place.

**3. Coordination via Prompt Engineering (Fixed Strategy)**
In the primary experiment (Towers of Hanoi), the authors manually coordinated the two roles to isolate the performance of the execution agents.
*   **Pre-loaded Insight:** The "insight" (the optimal strategy for moving disks) was embedded directly into the prompt of every execution agent.
*   **Isolated Execution:** Because the strategy was provided a priori, the agents did not need to derive the plan; they only needed to *execute* the next step based on that plan and the current state. This design choice effectively separates the ability to have insights from the ability to execute, preventing "entangled" failures where a bad plan ruins good execution.

**4. Aggregation of Results**
Once execution agents complete their atomic tasks, coordination resumes to combine the results. **Solution Discriminator Agents** vote on the composition of the executed subtasks to ensure the final output aligns with the original strategic breakdown.