"""Authorization guardrails (design: "SashiAI Chatbot — Authorization Guardrails").

    access_policy.yaml  capabilities, modules, gate words, messages and the open product decisions (one place)
    policy.py           loads the policy
    principal.py        the caller of one question: role, readable modules, resident scope
    identity.py         the signed-in user's data (cName, facilityId, scope, accessPermissions) → Principal;
                        AUTH_MODE=file reads it from dummy_user_context.json until the app integration
    gate.py             E: deny from the question's words before any model call (0 tokens)
    guard.py            C: every plan's capabilities, checked right before execution
    app/data/scope.py   D: the caller's residents only, inside every MongoDB pipeline (row scope)
"""

from app.auth.guard import AccessDenied, check_plan, current_principal, use_principal
from app.auth.principal import SYSTEM, Principal

__all__ = ["SYSTEM", "AccessDenied", "Principal", "check_plan", "current_principal", "use_principal"]
