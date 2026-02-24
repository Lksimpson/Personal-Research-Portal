"""
Trust behavior tests: citations present and suggested next retrieval step when evidence is missing.
Skips if Ollama is not available.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_trust_behavior_checks():
    """Run trust behavior checks; skip if Ollama not available."""
    from src.eval.trust_checks import run_trust_behavior_checks

    result = run_trust_behavior_checks()
    if result.get("skipped"):
        import pytest
        pytest.skip(result.get("message", "Ollama not available"))
    assert result.get("citations_ok"), result.get("message", "Citations check failed")
    assert result.get("suggested_next_step_ok"), result.get("message", "Suggested next step check failed")
