---
name: User prefers concise responses and direct action
description: User gets frustrated when assistant over-explains or recommends alternatives instead of following instructions
type: feedback
---

Do what the user asks first, explain briefly after if needed. Do not suggest alternatives when the user has made a decision. If the user says "make a plan as I SAID", they want their exact request executed, not a debate about whether it's the best approach.

**Why:** User explicitly said "NO! I first want to test a DQN" after assistant recommended PPO instead of planning the DQN as asked.

**How to apply:** Follow the user's stated approach. Save architectural opinions for when they ask "what do you think?" not when they give a direct instruction.
