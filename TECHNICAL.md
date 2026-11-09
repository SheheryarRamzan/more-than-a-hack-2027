# Matrix AE — frame-level possession (technical document, draft)

## 1. Summary

Possession is decoded as **segments**, because the ranking metric (`spellF1`) scores whole
possessions. Two questions are answered in turn, each by a two-state Viterbi decoder whose
switching cost turns noisy per-frame evidence into intervals: **is the ball alive?**, then, inside
every alive stretch, **which team has it?** The per-frame evidence for both comes from models
**trained on seven other professional matches** (open data), degraded to look like the provided
automatic tracking. Every design choice and every decoder setting was selected by leave-one-match-
out validation on those seven matches, scored with the organisers' own metric code. Nothing was
chosen on the competition match's ground truth.

Leave-one-match-out on the seven external matches (mean over three readings of "controlled
possession", see §4):

| | spellF1 | macroF1 |
|---|---|---|
| organisers' baseline (nearest player, ±2 s vote) | 0.2255 | 0.5552 |
| logistic per-frame models, default decoder | 0.4596 | 0.8315 |
| gradient-boosted per-frame models, default decoder | 0.4747 | 0.8364 |
| gradient-boosted, tuned decoder, model's own dead-ball threshold (v3) | 0.4897 | 0.8441 |
| + context features (§5), re-tuned decoder, larger trees (v4) | 0.5350 | 0.8614 |
| + second-stage models and team-change snapping (§6), re-tuned decoder (v5) | 0.5558 | 0.8728 |
| **submitted configuration** (v5, model's own dead-ball threshold, §6) | **0.5558** | **0.8728** |

The organisers' baseline scores 0.2433 / 0.6724 on the competition match itself, so the degraded
validation matches are about as hard as the real one (slightly harder).

## 2. Data used

| asset | used for |
|---|---|
| tactical ATD (players + ball) | all per-frame evidence |
| ATD metadata | track slot → team (DataFormatSpecification order, cross-checked with Player order) |
| Smart Tagging — set pieces | restart anchors: the kick is located in the tracking inside each tag window → "dead before, alive after, tagged team restarts" |
| Smart Tagging — goals | stoppage from the ball in the goal mouth to the following kick-off |
| tactical homography | tried as a coarse ball position (camera aim point) and **dropped** — see §6 |
| **IDSSE open data** (other games) | training and validation of the per-frame models and the decoder settings |
| tactical video | tried for re-detecting the ball and **dropped** — see §6 |
| broadcast video/ATD/homography | not used |

**IDSSE** (Bassek, Weber, Rein & Memmert, *An integrated dataset of synchronized spatiotemporal
and event data in elite soccer*, Scientific Data 2025; CC BY 4.0): seven full German Bundesliga /
2. Bundesliga matches (2022/23) with official DFL tracking. Every ball frame carries `BallStatus`
(alive/dead) and `BallPossession` (home/away) from corrected official tracking — the same kind of
per-frame label the challenge's ground truth is derived from. RULES.md allows external data and
labels about other games; this data is public, so the organisers can access it. No data about the
competition match other than the provided bundle is used anywhere.

## 3. Making the validation honest: degradation

Clean official tracking would flatter any method, so each external match is degraded to look like
the provided automatic tracking, using statistics of the provided tracking's *detection process*
only (never a label):

- **Ball loss.** The ATD does not lose the ball in flickers: it loses it in long stretches (median
  gap 10.7 s; two thirds of missing time is in gaps over 20 s). External balls are removed in
  alternating seen/lost stretches whose lengths are drawn from the ATD's own run-length
  distribution. The ATD loses the ball more when it is dead: measured against Smart Tagging
  pseudo-labels, 18.5 % missing in dynamic live phases, 34–36 % before restarts, 40.7 % overall,
  reproduced by 28 % (alive) / 55 % (dead).
- **Players** drop out and return (mean ~20.4 tracked of 22, as in the ATD), and jitter.
- **Referees** tracked in two of the DFL matches are removed.
- **Smart Tagging** is emulated at the true restarts with the provided file's coverage pattern (every
  corner, ~60 % of throw-ins with no team, ~20 % of free kicks, team named except on throw-ins and
  kick-offs after goals), as a 5 s window around the restart.

**Realism check.** The organisers' baseline scores 0.2255 spellF1 / 0.5552 macroF1 on the degraded
matches, against 0.2433 / 0.6724 on the competition match: the validation matches are about as
hard as the real one.

## 4. The label definition

DFL's possession changes on contact (median spell 3.8 s, 29 % under 2 s). The challenge counts
**controlled** possession — "a simple touch without control" and "a deflection that does not
intercept" do not change it (CHALLENGE.md) — and the public five minutes show shorter possessions
do exist but are rarer (2 of 17 under 2 s; median 7.6 s). External labels are therefore
harmonised: an interior spell shorter than a control threshold, with the same opposing team on both
sides inside one alive stretch, is given back to that team. Spells next to a stoppage are kept (a
restart always has a clear possessor).

The threshold cannot be pinned down from 17 possessions, so it is not chosen: **every validation
number is the mean over thresholds of 0.5, 1 and 2 s**, and a setting is only preferred if it wins
across these readings. Models are trained on the 1 s reading.

## 5. Evidence

**Control (team).** Distance from the ball to each team's nearest tracked player: full evidence
within 1 m, none beyond 3 m; when both teams are close, `tanh(distance gap / 1 m)`; down-weighted
when the ball moves faster than 15 m/s. Ball cleaning removes isolated jumps faster than 40 m/s and
bridges gaps up to 0.6 s.

**Collective (team) — works while the ball is unseen.** Attack direction per half from the
goalkeepers (median of each team's rearmost x). Forward speed of each team's centroid (home minus
away; the team with the ball pushes up), share of each team's players goal-side of the last-seen
ball (the defending team gets behind it), difference in team width and length (the team with the
ball stretches the pitch), relative position of the two blocks, and 2 s versions of the strongest.

**Dead ball.** Collective intensity (mean player speed, ±1 s, standardised within the match so a
model transfers between tracking systems), share of players running, a two-component mixture fitted
to the match's own intensity, ball beyond a line, ball at rest ≥ 1 s, ball moving inside the pitch,
ball missing and missing for long.

**Context (v4).** Frame-level evidence is blind inside the ATD's long ball gaps, so both models also
see context computed over the whole match (offline, so looking ahead is legitimate). Team model:
which team last had clear control of the ball and which team next has it, how long ago / how soon
(raw and with a 5 s decay), control over the past and next 2 s, time since / until the ball is seen,
and 5–10 s versions of the goal-side, forward-speed, centroid and width signals. Dead model: time
since / until the ball is seen, players speeding up over the next 2 s (a restart), 5 s intensity and
running share, the ball going out in the last / next 5 s, and the share of missing ball over ±5 s.
On validation this was the largest single gain: 0.4897 → 0.5231 at unchanged decoder settings, and
every one of the seven held-out matches improved, by 0.03–0.06.

## 6. Models and decoding

- **Per-frame models**: gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier`, 200
  trees of up to 63 leaves, ≥ 200 samples per leaf, learning rate 0.05, fixed seed, no early
  stopping) — chosen over logistic regression on validation (0.4747 vs 0.4596); tree settings by
  leave-one-match-out search (63 leaves 0.5348 vs 31 leaves 0.5326). Dead model on all graded frames with the training games' base rate; team model on alive
  frames with balanced classes. Trained on all seven matches for the submission.
- **Second stage (v5)**: a second pair of boosted models (100 trees of up to 31 leaves) reads the
  first stage's log-odds as time series — the value, its means over the past and next 1 s and 5 s
  and ±15 s, and how often its sign flips within ±2 s — for both the dead and team signals, plus
  the control evidence and whether the ball is seen. That lets it learn patterns one frame cannot
  show (evidence that flips and flips back is a deflection, not a turnover). It is trained on
  **out-of-fold** first-stage predictions: each training match is predicted by a first stage
  fitted without it, so the second stage never learns from over-confident in-sample output.
  Validation: 0.5350 → 0.5485 at unchanged decoder settings; six of seven matches improved, one
  unchanged.
- **Stage 1, alive/dead**: Viterbi on the dead log-odds plus restart anchors; minimum stoppage and
  live-play durations; edges snapped to the ball leaving the pitch or the restart kick.
- **Stage 2, home/away**: Viterbi on the team log-odds, decoded from one second before each restart
  so the player standing over the dead ball (the taker) sets the starting team; sustained bias to the
  team Smart Tagging names on a located restart. Each home↔away change is then moved to the first
  frame within ±0.5 s where the team gaining the ball has clear control (v5): the decoder places a
  change where the smoothed evidence tips over, which can lag the moment the ball is won, and the
  ranking metric scores boundary placement (snapping alone: 0.5350 → 0.5432).
- **Settings**: chosen by coordinate descent on mean leave-one-match-out spellF1 (macroF1 breaking
  ties), two passes, then the two settings that ended on a grid edge re-searched with a wider grid:
  team switching cost 4 (v5; 8 in v4, 16 in v3 — each model generation is sharper and needs less
  smoothing; with the second stage 4 / 6 / 8 / 12 / 16 score 0.5558 / 0.5541 / 0.5516 / 0.5480 /
  0.5463), alive/dead switching cost 48 (48 / 96 / 192: 0.5558 / 0.5539 / 0.5430), snapping window
  ±12 frames at control evidence ≥ 0.5, minimum stoppage 1 s, minimum live stretch 1.5 s, tag bias
  0.5 per frame over ±1 s, restart look-back 1 s. The two switching costs sit at the low edge of
  their grids, but the score is nearly flat there (0.0017 between 4 and 6), so going lower would
  chase noise.
- **Dead-ball share.** A rule fixed *before* looking at the competition match: keep the trained
  model's own alive/dead threshold if the dead share it gives there is plausible (0.43–0.53),
  otherwise shift the threshold to the league prior of 0.48 (LaLiga effective playing time over the
  graded 105.9 minutes). The v3 and v4 models gave 0.416 / 0.417 and were shifted; **v5 gives
  0.434, so its own threshold is kept** — which is also what scores best on validation (0.5558,
  against 0.5480 with the share forced to 0.48).

**Tried and dropped on validation**

| idea | held-out spellF1 | why dropped |
|---|---|---|
| restart team from the Laws of the Game (corner → attackers of that end, goal kick → defenders, throw-in / kick-off → opponents of the last toucher) | 0.4881 / 0.4878 / 0.4853 at strength 0.5 / 1 / 2, vs 0.4883 without | "last toucher" read from noisy tracking is less reliable than who stands over the ball |
| camera aim from the homography as a ball position when the ball is lost | 0.4608 vs 0.4596 (logistic) | +0.001 is noise, and validating it needs a synthetic camera for the open matches |
| re-detecting the ball in the tactical video with FootAndBall (Komorowski et al. 2020, MIT; built for long-shot football) on the GPU, to fill the ATD's ball gaps | see below | too little gain for the cost and a reproducibility risk |

*Ball detector, in detail.* The ball is ~4.4 px across in the tactical video (median over tracked
frames; p10 3.2, p90 6.4). Pilot on 5,000 frames at native resolution: a candidate within 1.5 m of
the tracked ball on 1.3 % of frames — the model does not see a ball that small. Upscaled 2×
(1,000 frames, 4.4 frames/s on an RTX 3060 laptop GPU): median distance to the tracked ball 0.9 m,
but only ~70 % of confident detections agree with it, and linking candidates into moving tracklets
does not raise that (62–72 % at every setting — the false hits are persistent objects, not
flicker); it fills ~20 % of the frames where the tracking has no ball. On validation, *perfectly*
filling half of the live-play gaps would add +0.025 spellF1 and most of them +0.048; filling ~20 %
at ~70 % reliability is worth roughly +0.01 at best, a quarter of the 0.04 noise band, and wrong
balls can flip possession. Against that: 10–12 GPU-hours per run, and GPU inference that may differ
in the last bits on another machine, which would break the rule that the repository must
regenerate the submitted file exactly. The code is kept in `detector/` and `possession/ball_fusion.py`
for reference; the submission does not use it.

## 7. Possession rules implemented

- Possession changes only on sustained control by the other team; loose balls, deflections and
  duels without control keep the previous team.
- Every restart starts a possession for the restarting team.
- Everything graded before a half's kick-off is dead; after a goal, dead until the kick-off.
- No `X` on graded frames: the scorer fills `X` with the previous label, which would carry `D` into
  a new possession.

## 8. Use of the public ground truth

Used twice, both as CHALLENGE.md invites ("use it to understand what is expected"), never to choose
a setting: (1) the shape of its possession durations informed the range of control thresholds
averaged over in §4; (2) the final file is scored on it once, as a correctness check.

## 9. Results on the competition match (aggregates only)

v5 (submitted): 353 possessions of mean 10.2 s, dead share 0.434. Agreement with Smart Tagging's
team phases is 0.750 (v4 at the same threshold: 0.749).

Agreement with Smart Tagging's team phases (which the models never see) was 0.774 for v4 (dead
share forced to 0.48), against 0.806 for v3. Checked before adopting v4: the drop comes from v4 predicting more short possessions (15 % of
alive frames in spells under 5 s, against 7 %), which coarse phase intervals cannot represent; at
comparable spell lengths v4 agrees as well or better (5–15 s: 0.735 vs 0.677; ≥ 15 s: 0.881 vs
0.890). On the held-out matches, where real labels exist, those extra possessions are what raises
spellF1 (truth 375 possessions per match; v4 predicts 284, v3 243).

Public five minutes, scored once per version as a correctness check, each version adopted on
validation *before* its check:

| version | macroF1 | spellF1 | alive/dead F1 | possessions (truth 17) |
|---|---|---|---|---|
| v3 | 0.8121 | 0.3867 | 0.942 | 13 |
| v4 | 0.8543 | 0.5250 | 0.961 | 15 |
| v5 (submitted) | 0.8422 | 0.4706 | 0.960 | 17 |

v5 scores below v4 here and above it on validation (0.5558 vs 0.5350 over ~2,600 held-out
possessions, six of seven matches better). With 17 possessions one possession matched or missed
moves spellF1 by ~0.06 and the bootstrap band is ~0.33 wide, so these five minutes confirm that
every version is correct and cannot rank them; choosing between versions on them would also be
tuning on the public ground truth, which the rules forbid.

## 10. Limitations

- **Domain shift.** The models are trained on official DFL tracking made to look like the automatic
  ATD; the degradation matches the ATD's ball-loss statistics, player dropout and tagging coverage,
  but not everything (e.g. tracker identity switches, the ATD's own speed estimates).
- **Label definition.** DFL possession is converted to controlled possession by a rule, and the
  rule's threshold is uncertain; results are averaged over three readings to stay robust to it.
- **Under-segmentation.** On validation the tuned decoder predicts ~25 % fewer possessions than the
  truth — the official metric prefers fewer, well-placed possessions to more, misplaced ones — so
  short real possessions (quick turnovers) are often missed.
- **Long ball gaps.** Two thirds of the ATD's missing ball time is in gaps over 20 s; inside them the
  team call rests on context and collective shape (team accuracy 0.879 on validation).
- **Unused assets.** The videos and the broadcast feed; a ball detector on the tactical video is the
  most promising next step for the long gaps.

## 11. Reproduction

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-solution.txt  # Windows; bin/python elsewhere
# put the data bundle in data/ (see DATA.md)
.venv/Scripts/python -m possession.run --name matrix_ae            # uses the committed models
.venv/Scripts/python evaluation/validate_submission.py evaluation/results/matrix_ae.csv

# optional: regenerate the models and the validation numbers (downloads ~2.5 GB of open data)
.venv/Scripts/python -m possession.train
.venv/Scripts/python -m possession.validate
.venv/Scripts/python -m possession.tune
```

`possession.run` reads `possession/models/idsse_models.json`, which holds only model parameters
learned from the seven open matches — nothing from the competition match — so it is committed.
Deterministic: two runs produce byte-identical files (Python 3.13.13, pinned versions in
`requirements-solution.txt`; verified byte-identical from a fresh clone in a fresh environment;
`requirements-lock.txt` is the full Windows development freeze). Derived data (caches under `data/cache/`, the open data under
`data/external/`) is gitignored with the rest of `data/`.
