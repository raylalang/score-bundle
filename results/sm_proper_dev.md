# The spectral mixture method applied as intended (dev, exploratory)

2026-10-07. Follow-up to the supervisor's naming correction
(`docs/sm_kernel_verification.md`): our estimator-v2 was a
hand-structured two-component kernel, not the Wilson & Adams METHOD.
This study runs the method proper on the same within-note cents curves.
**Development data only; exploratory; no claims.**

- Code: `scripts/eval_sm_proper.py`. Q = 5 free components, means free
  on [0, 15] Hz (physical cap only), no role pinning, no coherence
  bound; least-squares-periodogram initialization (uneven-sampling
  analogue of the paper's empirical-spectrum init), greedy component
  addition, joint Nelder-Mead polish, flat-prior-c evidence. The one
  kept bound is mathematical and uniform: v_q >= 0.025/span^2 (a
  component flat over the note is exactly redundant with the flat-prior
  constant).
- Comparator: the hand-structured 2-component kernel
  (`phase2/sm_estimator.fit_sm_note`), both fit on the tracked curve,
  both scored on the ground-truth curve's frames (the estimand-free
  curve-level protocol of `results/sm_estimator_dev.md` M4).
- Committed question, before the run: does the learned spectrum
  (a) describe the curves better, and (b) rediscover the vibrato band
  (a component in 4-9 Hz with >= 5% of mixture power) without being
  told?

## Pilot (4 dev tracks, 253 notes of >= 20 frames, 195 vibrato-identifiable; 8.1 min)

| measure | result |
|---|---|
| evidence, learned minus hand | median −0.41; learned higher on 40% of notes |
| GT-frame RMSE delta | median +0.035 cents (tie) |
| GT-frame NLL delta | median −0.030 (tie) |
| coverage@90 | learned 0.880 vs hand 0.881 |
| vibrato band rediscovered | **27%** of identifiable notes |
| wall per note | 0.9 s vs 0.5 s |

Diagnostic (one-note inspection before the batch,
`.cache/sm_proper_inspect.png`): these curves' spectra are dominated by
the drift peak at DC; on a quantized note the vibrato bump sits ~3.5
orders of magnitude below it, and the evidence prefers attributing it
to noise. The greedy mixture spends its components approximating the
drift's spectral shape (several near-DC Gaussians), which is faithful
W&A behaviour, not an optimizer failure.

## Reading (pilot-scale, pending the full run)

The method proper neither improves curve description (tie on every
curve-level measure) nor discovers the vibrato structure on most notes
(27%). Together with the earlier head-to-head, the picture is coherent:
on short, drift-dominated, quantization-noised cents curves, the
hand-imposed two-component structure costs nothing in fit quality, and
its value is interpretive, i.e. it is what DEFINES the vibrato channels.
"Pattern discovery" in the W&A sense has little to discover here at the
single-note scale.

Honest caveats: pilot scale (4 tracks); the greedy-NM optimizer is our
adaptation (the paper uses conjugate gradients, typically on longer
series); a per-track (pooled) spectrum rather than per-note might give
the method a fairer discovery setting — noted as a possible follow-up,
not scheduled.

## Full run (all dev tracks, 9,804 notes of >= 20 frames, 5,523 vibrato-identifiable; 4 shards, 2026-10-08)

| measure | result |
|---|---|
| evidence, learned minus hand | median −0.08; learned higher on 43% of notes |
| GT-frame RMSE delta | median −0.001 cents (exact tie) |
| GT-frame NLL delta | median −0.020 (tie) |
| coverage@90 | 0.800 both |
| vibrato band rediscovered | **18%** of identifiable notes |
| wall per note | 0.6 s vs 0.3 s |

The pilot's reading survives at 39x the sample and sharpens: curve-level
description is an exact tie, the evidence slightly prefers the smaller
hand-structured kernel (Occam, at matched description), and unaided
rediscovery of the vibrato band FALLS at scale (27% -> 18%; the pilot's
four tracks were vibrato-richer than the corpus). Verdict: on this
corpus the spectral mixture method has nothing to discover at the
single-note level that the hand structure does not already encode, and
the hand structure costs nothing. The channel-defining structure must
be imposed, which is the design question's point.
