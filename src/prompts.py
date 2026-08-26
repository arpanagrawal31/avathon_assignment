"""
Prompt templates
"""

TEACHER_SYSTEM = """You are an expert email triage analyst for enterprise operations. \
You extract structured information from corporate emails with high precision. \
When a field is genuinely absent from the email, use null — never fabricate."""

TEACHER_PROMPT = """Analyze the following corporate email and extract operational triage information as JSON.

STRICT RULES:
1. "intent" — exactly one of: request, status_update, scheduling, approval_request, escalation, fyi, other
2. "urgency" — exactly one of: low, medium, high
3. "requires_response" — boolean: does this email expect a reply?
4. "action_items" — list of action items. Use EMPTY LIST [] if no actions are needed.
   - "owner": the person responsible. Must be a VERBATIM substring from the email, or null if not stated.
   - "task": what needs to be done. May be paraphrased.
   - "deadline": when it's due. Must be a VERBATIM substring from the email, or null if not stated.
5. "escalation_flag" — boolean: does this require management attention or cross-team escalation?

CRITICAL: "owner" and "deadline" values must be EXACT substrings copied from the email text. Do not rephrase names or dates. If you cannot find an explicit owner or deadline in the email, use null.

EMAIL:
{email_body}"""

# ── Baseline prompts (for zero-shot / few-shot evaluation) ──

BASELINE_SYSTEM = (
    "You are an expert email triage analyst for enterprise operations. "
    "You extract structured information from corporate emails with high precision. "
    "When a field is genuinely absent from the email, use null — never fabricate. "
    "Output ONLY valid JSON — no markdown fences, no explanation, no extra text."
)

BASELINE_PROMPT = """Analyze the following corporate email and extract operational triage information.

Output ONLY a JSON object with exactly these fields:
- "intent": one of: request, status_update, scheduling, approval_request, escalation, fyi, other
- "urgency": one of: low, medium, high
- "requires_response": boolean
- "action_items": list of {{"owner": string|null, "task": string, "deadline": string|null}}. Use [] if none.
- "escalation_flag": boolean

RULES:
- "owner" and "deadline" must be EXACT substrings from the email, or null
- "task" may be paraphrased
- If there are no action items, use an empty list []
- Output ONLY the JSON object, nothing else

EMAIL:
{email_body}"""
