"""
Student Assistant plugin for Mark LI (JARVIS).

Adds Rumi-style study features as ONE voice-callable tool: "student_assistant".
Drop this file into plugins/ and restart JARVIS — no other changes needed.

Covers: quizzes, revision notes, flashcards, formula lookups, study plans,
simplified/advanced explanations, and step-by-step problem solving — all
delivered through the normal Gemini Live voice/text session, using live
Gemini text generation (NOT offline).

Follows the same conventions main.py already uses elsewhere in this repo:
  - config/api_keys.json for the Gemini API key
  - google.genai.Client(...).models.generate_content(...) for one-off text gen
  - player.request_say(...) as the mid-task speech channel, since run()
    blocks the plugin's executor thread until it returns.
"""

import json
from pathlib import Path

from google import genai

# ── Tool declaration (Gemini function-calling schema) ───────────────────────
# Mirrors the shape of TOOL_DECLARATIONS entries in main.py.
PLUGIN = {
    "name": "student_assistant",
    "description": (
        "Helps the user study: generates quizzes, revision notes, flashcards, "
        "formula reference sheets, study plans, simplified or advanced "
        "explanations, and step-by-step problem solving. Use this whenever "
        "the user is studying, revising, doing homework, or preparing for an "
        "exam — including commands like 'quiz me', 'revise topic X', "
        "'flashcards for X', 'formula for X', 'study plan', 'explain simply', "
        "or 'hard mode'."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {
                "type": "STRING",
                "description": (
                    "One of: quiz, revise, flashcards, formula, study_plan, "
                    "explain_simply, hard_mode, solve"
                ),
            },
            "topic": {
                "type": "STRING",
                "description": "The topic or question, e.g. 'photosynthesis', 'quadratic equations'",
            },
            "subject": {
                "type": "STRING",
                "description": (
                    "Mathematics, Physics, Chemistry, Biology, ICT, English, "
                    "or another academic subject"
                ),
            },
            "difficulty": {
                "type": "STRING",
                "description": "easy | medium | hard (used for quiz and solve)",
            },
            "num_questions": {
                "type": "NUMBER",
                "description": "How many quiz questions to generate (default 5)",
            },
            "details": {
                "type": "STRING",
                "description": (
                    "Extra context for study_plan (exam date, subjects, hours "
                    "per day) or the exact question text for 'solve'"
                ),
            },
        },
        "required": ["action"],
    },
}

# ── API key / model plumbing (self-contained — no import from main.py) ──────
_BASE_DIR = Path(__file__).resolve().parent.parent
_API_CONFIG_PATH = _BASE_DIR / "config" / "api_keys.json"
_MODEL = "gemini-flash-latest"


def _get_api_key() -> str:
    with open(_API_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["gemini_api_key"]


def _generate(prompt: str) -> str:
    client = genai.Client(api_key=_get_api_key())
    resp = client.models.generate_content(model=_MODEL, contents=prompt)
    return (resp.text or "").strip()


# ── Prompt builders per action ───────────────────────────────────────────────

def _prompt_quiz(topic, subject, difficulty, num_questions):
    n = int(num_questions) if num_questions else 5
    difficulty = difficulty or "medium"
    return (
        f"You are a study tutor. Create a {difficulty}-difficulty quiz with "
        f"{n} questions on '{topic}'" + (f" ({subject})" if subject else "") + ". "
        "Mix multiple-choice, true/false, and short-answer questions. "
        "Number each question. For multiple-choice, label options A-D. "
        "After ALL questions, add an 'Answer Key' section with the correct "
        "answers and a one-line explanation for each. Keep it concise and "
        "voice-friendly — avoid heavy markdown tables."
    )


def _prompt_revise(topic, subject):
    return (
        f"You are a study tutor. Create concise revision notes on '{topic}'"
        + (f" ({subject})" if subject else "") + ". Include: a short summary "
        "paragraph, 5-8 key points as bullets, and 2-3 common mistakes "
        "students make. Keep it exam-focused and easy to skim."
    )


def _prompt_flashcards(topic, subject):
    return (
        f"You are a study tutor. Create 8 flashcards on '{topic}'"
        + (f" ({subject})" if subject else "") + ". Format each as "
        "'Q: ...' then 'A: ...' on the next line, separated by a blank line. "
        "Keep answers short (1-2 sentences)."
    )


def _prompt_formula(topic, subject):
    return (
        f"You are a study tutor. Give the key formula(s) for '{topic}'"
        + (f" ({subject})" if subject else "") + ". For each formula state: "
        "the formula itself, what each variable means, the units involved, "
        "when to use it, and one worked example calculation."
    )


def _prompt_study_plan(topic, subject, details):
    return (
        "You are a study coach. Create a study plan and timetable"
        + (f" for {subject}" if subject else "")
        + (f" on '{topic}'" if topic else "")
        + (f". Details from the student: {details}" if details else "")
        + ". If exam date, available hours, or subject list weren't given, "
        "make reasonable assumptions and state them briefly at the top. "
        "Structure it by day/week, include short goals and revision "
        "checkpoints, and keep it realistic."
    )


def _prompt_explain(topic, subject, hard):
    level = "advanced, in depth" if hard else "as simply and clearly as possible"
    return (
        f"You are a study tutor. Explain '{topic}'"
        + (f" ({subject})" if subject else "") + f", {level}. "
        "Use an example. Encourage understanding over memorization."
    )


def _prompt_solve(topic, details, subject, difficulty):
    question = details or topic
    return (
        "You are a study tutor. Solve this problem step by step: "
        f"{question}" + (f" ({subject})" if subject else "") + ". "
        "Structure your answer as: 1) Topic identified, 2) Relevant theory "
        "explained briefly, 3) Step-by-step solution, 4) Final answer clearly "
        "stated, 5) A common mistake students make on this type of problem."
    )


_BUILDERS = {
    "quiz": lambda a: _prompt_quiz(a.get("topic", ""), a.get("subject", ""),
                                    a.get("difficulty", ""), a.get("num_questions")),
    "revise": lambda a: _prompt_revise(a.get("topic", ""), a.get("subject", "")),
    "flashcards": lambda a: _prompt_flashcards(a.get("topic", ""), a.get("subject", "")),
    "formula": lambda a: _prompt_formula(a.get("topic", ""), a.get("subject", "")),
    "study_plan": lambda a: _prompt_study_plan(a.get("topic", ""), a.get("subject", ""), a.get("details", "")),
    "explain_simply": lambda a: _prompt_explain(a.get("topic", ""), a.get("subject", ""), hard=False),
    "hard_mode": lambda a: _prompt_explain(a.get("topic", ""), a.get("subject", ""), hard=True),
    "solve": lambda a: _prompt_solve(a.get("topic", ""), a.get("details", ""),
                                      a.get("subject", ""), a.get("difficulty", "")),
}

_INTERIM_LINES = {
    "quiz": "Putting a quiz together, one second.",
    "revise": "Pulling together revision notes for you.",
    "flashcards": "Making a set of flashcards for you.",
    "formula": "Looking up that formula.",
    "study_plan": "Building out a study plan.",
    "explain_simply": "Let me put that simply.",
    "hard_mode": "Let me go into more depth on that.",
    "solve": "Let me work through that step by step.",
}


def run(args: dict, player=None, session_memory=None) -> str:
    """
    Entry point called by core/plugin_loader.py.
    args: the function-call arguments Gemini extracted (matches PLUGIN["parameters"]).
    player: main.ui — use player.request_say(...) for a mid-task voice line,
            since this function blocks the plugin's executor thread.
    session_memory: reserved for future use (e.g. tracking weak topics).
    """
    action = (args.get("action") or "").strip().lower()

    if action not in _BUILDERS:
        return (
            f"Unknown study action '{action}'. Try: "
            "quiz, revise, flashcards, formula, study_plan, explain_simply, hard_mode, or solve."
        )

    if player is not None and hasattr(player, "request_say"):
        try:
            player.request_say(_INTERIM_LINES.get(action, "Working on it."))
        except Exception:
            pass  # interim speech is best-effort; never fail the tool over it

    prompt = _BUILDERS[action](args)

    try:
        return _generate(prompt) or "I couldn't generate that — try again in a moment."
    except Exception as e:
        return f"Study assistant couldn't reach Gemini: {e}"