"""Frame-level ball possession for More than a Hack 2027 (team Matrix AE).

Two-stage decoder over the provided automatic tracking:

  1. alive / dead   -- a 2-state Viterbi over dead-ball evidence (collective player intensity,
                       ball out of bounds, ball at rest, ball moving fast), anchored on the
                       restarts that Smart Tagging marks.
  2. home / away    -- inside every alive stretch, a 2-state Viterbi over ball-control evidence
                       (which team's nearest player is close enough to control the ball), with a
                       switching cost so that a touch or a deflection does not flip possession but
                       sustained control does.

Run it with:

    python -m possession.run --name matrix_ae

Every tunable number is in possession/config.py. None of them were fitted to the public ground
truth; see TECHNICAL.md.
"""
