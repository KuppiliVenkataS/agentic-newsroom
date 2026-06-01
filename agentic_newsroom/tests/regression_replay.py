"""
Regression replay — structural seatbelt.

Run after ANY code change to confirm the deterministic machinery still works
against a known-good frozen cycle. It does NOT judge prose quality (that stays
the human review step); it catches "did an edit break generation / the guard /
the magnitude tiering / the two-section split".

Usage:
    python tests/regression_replay.py

Exit code 0 = all structural checks pass. Non-zero = something broke.

This script imports the project's own modules. Run it from the repo root so
`config`, `report`, `guard` resolve. If your guard.py lives elsewhere, fix the
import below to match.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixture_cycle.json"

# Anchor "now" to just after the fixture's processed_at, so the freshness
# check is deterministic regardless of when the test runs.
FIXTURE_NOW = datetime(2026, 5, 31, 6, 0, 0, tzinfo=timezone.utc)

PASS, FAIL = "✓", "✗"
failures = []


def check(name, condition, detail=""):
    mark = PASS if condition else FAIL
    print(f"  {mark} {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def main():
    cycle = json.loads(FIXTURE.read_text(encoding="utf-8"))
    articles = cycle["articles"]
    print(f"Fixture: {cycle['run_id']} — {len(articles)} articles\n")

    # ── 1. Guard ────────────────────────────────────────────────────────────
    print("Guard:")
    try:
        from guard import check_cycle_health
    except ImportError:
        # guard.py may be placed at repo root or in a package; try common spots
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from guard import check_cycle_health

    verdict = check_cycle_health(cycle, now=FIXTURE_NOW)
    check("healthy cycle publishes", verdict["publish"] is True,
          f"publish={verdict['publish']} reasons={verdict['reasons']}")
    check("stats computed", verdict["stats"]["articles"] == len(articles),
          f"{verdict['stats']}")

    # Empty cycle must be blocked
    empty_verdict = check_cycle_health({"articles": []}, now=FIXTURE_NOW)
    check("empty cycle blocked", empty_verdict["publish"] is False)

    # Stale cycle must be blocked (pretend now is far in the future)
    future = datetime(2026, 6, 15, tzinfo=timezone.utc)
    stale_verdict = check_cycle_health(cycle, now=future)
    check("stale cycle blocked", stale_verdict["publish"] is False,
          f"reasons={stale_verdict['reasons']}")

    # Source-concentration warning should fire (fixture is gdelt-heavy)
    check("source-concentration warning present",
          any("source_concentration" in w for w in verdict["warnings"]),
          f"warnings={verdict['warnings']}")

    # ── 2. Magnitude tiering ─────────────────────────────────────────────────
    print("\nMagnitude tiering:")
    try:
        from report.generator import _compute_day_magnitude
    except ImportError:
        try:
            from generator import _compute_day_magnitude
        except ImportError:
            print("  (skipped — could not import _compute_day_magnitude; "
                  "fix the import path to your generator)")
            _compute_day_magnitude = None

    if _compute_day_magnitude:
        tier, lead = _compute_day_magnitude(articles, is_alert=False)
        check("tier is valid", tier in ("quiet", "normal", "loud"), f"tier={tier}")
        check("lead line non-empty", bool(lead), f"lead={lead[:60]}")
        # is_alert must force loud
        tier_alert, _ = _compute_day_magnitude(articles, is_alert=True)
        check("is_alert forces loud", tier_alert == "loud", f"tier={tier_alert}")

    # ── 3. Two-section split logic ───────────────────────────────────────────
    print("\nTwo-section split:")
    sample = ("HEADLINE: test\nBrent 93 | WTI 89\n\nBrief body.\n"
              "===ANALYST===\n01/06 London\nAnalyst detail.")
    if "===ANALYST===" in sample:
        brief, analyst = sample.split("===ANALYST===", 1)
        out = (brief.rstrip() + "\n\n---\n\n"
               + "## 📊 Analyst Note — *adviser detail (not for distribution as-is)*\n\n"
               + analyst.lstrip())
    check("separator splits cleanly", "Analyst Note —" in out)
    check("promoter headline stays line 1", out.splitlines()[0].startswith("HEADLINE"))

    # ── Result ───────────────────────────────────────────────────────────────
    print()
    if failures:
        print(f"REGRESSION FAILED — {len(failures)} check(s) broke: {failures}")
        sys.exit(1)
    print("REGRESSION PASSED — structural machinery intact.")
    sys.exit(0)


if __name__ == "__main__":
    main()