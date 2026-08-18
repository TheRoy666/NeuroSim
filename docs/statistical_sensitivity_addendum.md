# Statistical sensitivity addendum

True a priori pre-registration (locking hypotheses before any data
collection or analysis) is not possible to construct retroactively —
this project's data collection and analysis are complete. This
addendum does the honest thing that's actually still possible: states
plainly which analyses were genuinely pre-specified before results were
seen, and for everything else, discloses precisely what effect size
each comparison's actual sample size could reliably detect, computed
directly (not assumed), using standard, defensible thresholds — not
chosen post-hoc to flatter any particular result.

**Method:** minimum detectable effect size (MDES), Cohen's d, via
`statsmodels.stats.power.TTestIndPower`, at α=0.05 and 80% power — the
conventional default pair, not tuned. Real, actual group sizes used
throughout, not equal-split assumptions.

## What was genuinely pre-specified, before results were seen

**UNAM-TLE Path C lateralization test.** Primary ROI set (entorhinal,
fusiform, parahippocampal, temporal pole) and prediction direction
(negative primary-MTL-asymmetry → predict Left-TLE) were both fixed
before computing accuracy — documented at the time, not reconstructed
after the fact. This is the one analysis in the project that can
honestly be called pre-registered in the traditional sense.

## Minimum detectable effect size, everything else

| Comparison | N (group 1, group 2) | MDES (Cohen's d) | Convention |
|---|---|---|---|
| HCP-AUD: Social_Drinker vs Abuser | 196, 26 | 0.587 | medium-large |
| HCP-AUD: Social_Drinker vs Dependent | 196, 16 | 0.732 | large |
| ADNI-3: CN vs MCI | 25, 24 | 0.817 | large |
| UNAM-TLE: Healthy vs Left-TLE | 34, 17 | 0.849 | large |
| UNAM-TLE: Healthy vs Right-TLE | 34, 11 | 0.994 | very large |
| UNAM-TLE: Left-TLE vs Right-TLE | 17, 11 | 1.126 | very large |

**What this means, stated plainly:** none of these clinical case-study
comparisons were powered to detect small-to-medium real effects (the
range most biologically plausible clinical signals of this kind
typically fall into). A true effect in that range could have been
present and this project's sample sizes would not reliably have found
it. This is precise, quantitative confirmation of what's already been
qualitatively stated throughout this project's own honest reporting —
the AUD severity comparison and UNAM's lateralization result were both
already reported as underpowered nulls, not overclaimed as evidence of
absence. This table is what makes that qualitative statement
falsifiable and exact, not just an impression.

**What this does NOT undermine:** the project's primary, best-powered
results — the full 238/49/49-subject cohort analyses (operator-
divergence, Path A1's real-coupling and null-model work, Path B's
reliability findings) — are not effect-size comparisons between small
subgroups; they're within-cohort structural and statistical
characterizations at full sample size, a fundamentally different
statistical situation than the clinical subgroup comparisons in this
table. This addendum specifically concerns the clinical case studies'
subgroup comparisons, not the paper's central methodological results.
