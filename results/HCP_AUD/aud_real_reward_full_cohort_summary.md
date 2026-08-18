# Phase 7 real-reward-network full-cohort result (238/238, 0 failed, 92.6 min)

## Group medians, E_craving_to_rest — real reward-network definition

- Social_Drinker (N=196): median=0.2571, IQR=[0.1985, 0.3098]
- Abuser (N=26): median=0.2090, IQR=[0.1858, 0.2610]
- Dependent (N=16): median=0.2563, IQR=[0.1894, 0.3307]

Non-monotonic: Abuser sits distinctly below both Social_Drinker and
Dependent, which are nearly identical to each other.

## Discordant MZ pairs (twin-controlled) — real reward-network definition

- Sev0_vs_Sev1 (N=12): mean_dE=-0.0017, 6/12 positive (chance-level direction)
- Sev0_vs_Sev2 (N=8): mean_dE=-0.0624, 3/8 positive
- Sev1_vs_Sev2 (N=1): uninterpretable at N=1

Weak, inconsistent signal once genetic/shared-environment confounds
are controlled for — worth reporting honestly as weak, not smoothed
over.

## Teleportation ratio — real reward-network definition

Overall median=3.38x, IQR=[2.83, 4.24] — close to the already-reported
operator-divergence AUD median (3.22x), a reassuring sign the core
operator-divergence finding is robust to this state-definition change.

---

## The complete comparison: real reward-network vs. the original proxy

The original proxy-based reward-network run (`run_hcp_aud_batch.py`,
same 238 subjects) was already available in this project's result
files (`aud_group_comparison_public.csv`, `aud_discordant_pairs_public.csv`,
`aud_teleportation_public.csv`) — not a separate run that needed to be
requested. This comparison should have been completed at the time this
summary was first written; it's completed here instead.

### E_craving_to_rest group medians, side by side

| Group | Proxy median | Real median | Both show the dip? |
|---|---|---|---|
| Social_Drinker (N=196) | 0.2820 | 0.2571 | — |
| Abuser (N=26) | 0.2350 | 0.2090 | **Yes — lowest in both** |
| Dependent (N=16) | 0.2843 | 0.2563 | — |

**The non-monotonic pattern replicates exactly**: Abuser sits distinctly
below both other groups under both reward-network definitions. This is
not a state-definition artifact — it holds regardless of whether the
reward network is defined by proxy variance or by real anatomical
reward-circuit parcellation.

### Discordant MZ pairs, side by side

| Contrast | Proxy mean_dE | Real mean_dE | Proxy N+ | Real N+ |
|---|---|---|---|---|
| Sev0_vs_Sev1 | -0.0027 | -0.0017 | 7/12 | 6/12 |
| Sev0_vs_Sev2 | -0.0616 | -0.0624 | 3/8 | 3/8 |
| Sev1_vs_Sev2 | -0.0708 | uninterpretable (N=1) | 0/1 | — |

**Sev0_vs_Sev2's signal is nearly numerically identical between the
two state definitions** (-0.0616 vs. -0.0624, same 3/8 direction split)
— about as clean a confirmation as a twin-controlled result this small
can offer. Sev0_vs_Sev1 stays at chance level in both.

### Teleportation ratio, side by side

Proxy: median=3.22x, IQR=[2.71, 3.96]. Real: median=3.38x,
IQR=[2.83, 4.24]. Consistent.

### Status: complete, not one half

**The real reward-network results confirm the original proxy-based
findings rather than revealing a different picture.** Both the
group-level non-monotonic Abuser dip and the weak, twin-controlled
discordant-pair signal replicate closely under an entirely different
reward-network state definition — real anatomical parcellation instead
of proxy variance-based selection. This is a genuine methodological
validation: the AUD severity findings reported throughout this project
are not an artifact of how the reward network was operationally
defined.
