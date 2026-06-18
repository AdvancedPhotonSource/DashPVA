"""Scratchpad tools that externalize the agent's reasoning.

These let the model record its investigation *plan*, the *findings* it has
verified (with the evidence each came from), and the *hypotheses* it has not
yet confirmed (with the test that would settle them). The results are surfaced
to the scientist as first-class, inspectable, exportable artifacts — see
:class:`~dashpva.analysis.chat_controller.ChatController`, which maps these tool
names to typed ``ControllerEvent`` kinds (``plan``/``finding``/``hypothesis``).

Keeping these as ordinary tools (rather than special-casing the loop) means the
model decides when to externalize its reasoning, the JSON schema is generated
automatically, and the registry/executor stay domain-agnostic.
"""

from __future__ import annotations

from dashpva.analysis.tools.base import BaseTool, tool


class ReasoningTools(BaseTool):
    """Pure scratchpad — no external state. Each tool just echoes a structured
    record back; the controller turns that record into a typed event."""

    @tool(description="Record your investigation plan as 1-3 short, concrete "
                      "steps. Call this ONCE at the very start of a turn, before "
                      "any other tools.")
    def record_plan(self, steps: list[str]) -> dict:
        """Record the plan for this investigation.

        Args:
            steps: 1-3 short imperative steps you intend to carry out.
        """
        clean = [str(s).strip() for s in (steps or []) if str(s).strip()]
        if not clean:
            return {'error': 'record_plan needs a non-empty list of step strings.'}
        return {'kind': 'plan', 'steps': clean}

    @tool(description="Record a finding you have VERIFIED from a tool result, "
                      "with the exact evidence (tool name + numbers + frame/PV) "
                      "it came from. Use exact numbers, not approximations.")
    def record_finding(self, statement: str, evidence: str,
                       confidence: str = 'medium') -> dict:
        """Record a verified finding.

        Args:
            statement: The fact you are asserting.
            evidence: The tool result / numbers / frame id this is based on.
            confidence: One of 'high', 'medium', 'low'.
        """
        statement = str(statement).strip()
        evidence = str(evidence).strip()
        if not statement:
            return {'error': 'record_finding needs a non-empty statement.'}
        conf = str(confidence).strip().lower()
        if conf not in ('high', 'medium', 'low'):
            conf = 'medium'
        return {'kind': 'finding', 'statement': statement,
                'evidence': evidence, 'confidence': conf}

    @tool(description="Note a hypothesis you have NOT yet verified, together "
                      "with the specific tool call that would confirm or refute "
                      "it. Use this to track open questions.")
    def note_hypothesis(self, hypothesis: str, test: str) -> dict:
        """Note an unverified hypothesis and how to test it.

        Args:
            hypothesis: The unconfirmed idea.
            test: The concrete tool call / measurement that would settle it.
        """
        hypothesis = str(hypothesis).strip()
        if not hypothesis:
            return {'error': 'note_hypothesis needs a non-empty hypothesis.'}
        return {'kind': 'hypothesis', 'hypothesis': hypothesis,
                'test': str(test).strip()}
