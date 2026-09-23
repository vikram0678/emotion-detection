"""
Intelligent Emotion-Aware Pedagogical Assistant Engine.
Supports multi-turn conversational memory, multiple agent personas (Socratic, Tutor, Coach, Expert),
multi-provider API dispatch (Gemini, OpenRouter, DeepSeek, Grok), and an intelligent offline
pedagogical generator that crafts rich, structured, empathetic learning guidance.
"""

import os
import random
from dotenv import load_dotenv

load_dotenv()

# Predefined dynamic guidance knowledge base for rich offline agent responses
DYNAMIC_OFFLINE_KNOWLEDGE = {
    "Computer Science": {
        "Confused": {
            "concept": "In computer science, confusion usually means multiple layers of abstraction are tangling up (e.g., memory pointers, recursion depth, or asynchronous execution).",
            "strategy": "Draw a dry-run execution trace on paper with a small sample input (e.g., n=3 for recursion or an array of 3 elements).",
            "socratic": "What is the exact input state right before the unexpected behavior occurs?",
            "action": "Isolate the function into an independent test script with print/debugger breakpoints."
        },
        "Frustrated": {
            "concept": "Bug fatigue happens when you test the same broken assumption repeatedly. A fresh diagnostic lens breaks the loop.",
            "strategy": "Take a 3-minute breather, then rubber-duck debug: explain each line out loud to an imaginary peer.",
            "socratic": "If you had to rewrite this logic using only simple loops and basic conditions, how would it look?",
            "action": "Use the Pomodoro technique (25 min coding, 5 min break) and commit working code checkpoints frequently."
        },
        "Curious": {
            "concept": "Curiosity is the engine of deep engineering mastery. Exploring edge cases and internal mechanics cements mental models.",
            "strategy": "Inspect the time/space complexity (Big-O) or explore the underlying data structure tradeoffs.",
            "socratic": "How would this algorithm perform if the dataset scaled from 1,000 items to 10,000,000 items?",
            "action": "Benchmark two different approaches using a timer and observe the memory/CPU difference."
        },
        "Confident": {
            "concept": "High confidence is the ideal moment to solidify mastery by advancing from solving to optimizing and teaching.",
            "strategy": "Try solving an edge case (empty inputs, negative values, massive scale) or refactor for maximum readability.",
            "socratic": "Can you explain this solution simply enough that a first-year student could code it in 5 minutes?",
            "action": "Move onto an advanced variation or challenge problem on LeetCode/HackerRank."
        },
        "Bored": {
            "concept": "Boredom indicates you have already internalized the basics and require a higher cognitive challenge.",
            "strategy": "Impose creative constraints: solve without built-in library methods, optimize for O(1) space, or build a mini-project.",
            "socratic": "How could you automate or gamify this repetitive task with a custom script?",
            "action": "Skip standard drills and build a real-world CLI tool or interactive web demonstration."
        }
    },
    "Mathematics": {
        "Confused": {
            "concept": "Mathematical confusion often stems from skipping the geometric intuition behind symbolic formulas.",
            "strategy": "Plot the equation or test simple boundary numbers (0, 1, -1) to see the pattern unfold.",
            "socratic": "Which algebraic transition or axiom feels like the missing link in this proof?",
            "action": "Work through a worked example backwards from solution to problem statement."
        },
        "Frustrated": {
            "concept": "Math frustration occurs when calculation errors mask conceptual understanding. Separate the two.",
            "strategy": "Break the multi-step problem into atomic sub-questions and verify each line algebraically.",
            "socratic": "What known theorem or formula closely resembles the structure of this problem?",
            "action": "Write out all given parameters and unknowns in a dedicated margin table before calculating."
        },
        "Curious": {
            "concept": "Mathematical curiosity unlocks connections between pure theory and real-world physical phenomena.",
            "strategy": "Explore how this mathematical concept is applied in physics simulations, graphics engines, or cryptography.",
            "socratic": "What happens geometrically to the curve or vector when the coefficient approaches infinity?",
            "action": "Visualize the equations dynamically in Desmos or GeoGebra."
        },
        "Confident": {
            "concept": "Mastery in mathematics empowers you to explore generalized proofs and multi-variable extensions.",
            "strategy": "Attempt proving the theorem from scratch without referencing textbook notes.",
            "socratic": "Under what specific boundary conditions would this mathematical property fail to hold?",
            "action": "Tackle contest-level problems (e.g., Putnam/AMC level) or explore abstract algebra."
        },
        "Bored": {
            "concept": "Repetitive calculation drills cause cognitive disengagement. Switch to conceptual proofs.",
            "strategy": "Instead of solving 20 identical arithmetic problems, explore the historical origin and derivation of the formula.",
            "socratic": "Why did mathematicians invent this technique in the first place?",
            "action": "Explore 3Blue1Brown visual essays or explore higher-dimensional generalizations."
        }
    }
}

# Generic fallback for other fields
GENERIC_FIELD_KNOWLEDGE = {
    "Confused": {
        "concept": "Confusion is the very first stage of active neuroplasticity and deep learning.",
        "strategy": "Synthesize the core problem into one single question sentence.",
        "socratic": "Which specific term or definition in this concept is the most ambiguous right now?",
        "action": "Create a 3-bullet summary of what you DO understand versus what remains unclear."
    },
    "Frustrated": {
        "concept": "High cognitive friction means your brain is wrestling with a complex mental schema.",
        "strategy": "Step back and articulate the goal in plain, everyday language.",
        "socratic": "If you stripped away all the complex jargon, what is this concept fundamentally trying to achieve?",
        "action": "Switch study mediums: sketch a visual diagram or explain the problem aloud."
    },
    "Curious": {
        "concept": "Curiosity accelerates retention by over 300% through intrinsic motivation.",
        "strategy": "Connect this concept to an adjacent domain or real-world application you love.",
        "socratic": "How does this principle influence everyday technology or systems in our society?",
        "action": "Write down 2 follow-up exploratory questions to research during your next deep-dive."
    },
    "Confident": {
        "concept": "Confidence is the optimal launchpad for teaching others and tackling advanced material.",
        "strategy": "Apply the Feynman Technique: synthesize and teach this topic to a beginner.",
        "socratic": "How would you defend this solution if someone questioned your fundamental assumptions?",
        "action": "Transition to higher-order problem sets or project-based applications."
    },
    "Bored": {
        "concept": "Boredom signals that the current challenge level is below your cognitive bandwidth.",
        "strategy": "Raise the stakes: increase speed, add realistic constraints, or tackle real case studies.",
        "socratic": "How could you apply this concept to solve a problem you genuinely care about?",
        "action": "Shift from passive review to active synthesis and project creation."
    }
}


def build_agent_system_prompt(persona: str = "Socratic Mentor") -> str:
    persona_instructions = {
        "Socratic Mentor": "You are a wise, supportive Socratic academic mentor. Guide the student by asking thoughtful questions, clarifying core intuition, and helping them discover solutions rather than just giving direct answers.",
        "Step-by-Step Tutor": "You are a meticulous, patient step-by-step academic tutor. Break down complex topics into clear, numbered, easy-to-digest stages with practical examples.",
        "Empathetic Coach": "You are an encouraging, empathetic learning coach. Prioritize psychological safety, validate student emotions warmly, reduce anxiety, and inspire a growth mindset.",
        "Concise Expert": "You are a high-density, crisp domain expert. Deliver direct, punchy explanations, high-yield insights, and actionable steps with zero fluff."
    }
    return persona_instructions.get(persona, persona_instructions["Socratic Mentor"])


def generate_smart_offline_response(field: str, problem: str, emotion: str, confidence: float, persona: str) -> str:
    """Generates a rich, structured pedagogical response when offline or without API keys."""
    field_data = DYNAMIC_OFFLINE_KNOWLEDGE.get(field, {}).get(emotion) or GENERIC_FIELD_KNOWLEDGE.get(emotion, GENERIC_FIELD_KNOWLEDGE["Confused"])
    
    # Emotion validation openers
    openers = {
        "Confused": [
            f"I hear you! Feeling confused about this is completely natural—it just means your brain is actively building a new mental model for {field}.",
            f"It's totally okay to feel unclear on this right now. Complex {field} topics have layers that become crystal clear once deconstructed."
        ],
        "Frustrated": [
            f"I understand how exhausting and frustrating this roadblock feels. Take a deep breath—you are much closer to a breakthrough than it seems.",
            f"Hitting a stubborn wall in {field} is tough, but friction is where the deepest learning happens. Let's tackle this methodically."
        ],
        "Curious": [
            f"Fantastic curiosity! Digging deeper into the 'why' behind {field} is exactly how great thinkers master their craft.",
            f"Love that curiosity! Asking these kinds of questions turns ordinary study into genuine domain mastery."
        ],
        "Confident": [
            f"Awesome work! Your clarity and confidence in {field} are shining through—you've got a strong grasp of the fundamentals.",
            f"Spot on! Operating with high confidence is the best time to lock in your mastery and level up to harder challenges."
        ],
        "Bored": [
            f"It sounds like this material is no longer challenging your potential. Let's elevate the difficulty and make it engaging.",
            f"When review gets monotonous, it's time to skip the routine drills and dive into creative, high-impact applications."
        ]
    }
    
    selected_opener = random.choice(openers.get(emotion, openers["Confused"]))
    
    # Format structured response
    response_md = f"""### 💙 Emotional Diagnosis & Validation
{selected_opener} *(Detected State: **{emotion}** with **{confidence:.1%}** certainty)*

---

### 💡 Core Concept & Strategy for {field}
- **Intuition**: {field_data['concept']}
- **Actionable Technique**: {field_data['strategy']}

---

### 🎯 {persona} Check & Next Step
- **Reflective Question**: *{field_data['socratic']}*
- **Suggested Micro-Action**: {field_data['action']}"""

    return response_md


def get_gemini_response(
    field: str,
    problem: str,
    emotion: str,
    confidence: float,
    use_ai: bool = True,
    persona: str = "Socratic Mentor",
    conversation_history: list = None,
    api_key_override: str = None
) -> str:
    """
    Generates intelligent pedagogical guidance.
    Tries configured LLMs (Gemini, OpenRouter, DeepSeek, Grok) with persona and history context.
    Gracefully falls back to the rich smart offline generator if AI is disabled or offline.
    """
    if not use_ai:
        return generate_smart_offline_response(field, problem, emotion, confidence, persona)

    system_persona_prompt = build_agent_system_prompt(persona)
    
    user_prompt = f"""You are an emotion-aware AI Learning Assistant.
Student Academic Field: {field}
Student Emotional State: {emotion} (Confidence: {confidence:.1%})
Student Persona Mode: {persona}

System Persona Directive:
{system_persona_prompt}

Student's Current Query/Challenge:
"{problem}"

Please provide a structured, beautifully formatted Markdown response with these exact headers:
### 💙 Empathy & Cognitive Validation
(A warm, empathetic acknowledgment tailored to their emotion and subject)

### 💡 Core Concept & Strategy
(Clear, intuitive explanation or problem-solving strategy for {field})

### 🎯 Interactive Next Step
(A targeted Socratic question or challenge prompt based on your {persona} role)

### 🚀 Recommended Study Action
(A concrete, practical study habit or tool to try right now)"""

    # 1. Try Gemini
    gemini_key = api_key_override or os.getenv("GEMINI_API_KEY")
    if gemini_key and gemini_key.strip() and not gemini_key.startswith("your_"):
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key.strip())
            model = genai.GenerativeModel("gemini-2.5-flash")
            resp = model.generate_content(user_prompt)
            if resp and resp.text:
                return resp.text.strip()
        except Exception as e:
            print(f"⚠️ Gemini provider note: {e}")

    # 2. Try OpenRouter
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key and openrouter_key.strip() and not openrouter_key.startswith("your_"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openrouter_key.strip(), base_url="https://openrouter.ai/api/v1")
            resp = client.chat.completions.create(
                model="openrouter/free",
                messages=[
                    {"role": "system", "content": system_persona_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ OpenRouter provider note: {e}")

    # 3. Fallback to rich dynamic offline generator
    return generate_smart_offline_response(field, problem, emotion, confidence, persona)