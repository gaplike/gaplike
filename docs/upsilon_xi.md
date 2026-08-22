# The Υ and Ξ diagnostics

Every Bayesian analysis quotes a posterior width. Two separate things can go
wrong with it, and the field's standard diagnostic — the PP plot — sees only
one of them. This page defines the two numbers the paper uses instead,
$\Upsilon$ and $\Xi$, shows why together they are strictly more informative
than a PP plot, and how to read them.

## Three widths, two ratios

For one parameter $\theta$, three different widths are in play:

$\sigma_{\rm quoted}$
: the width the analysis *reports* — the posterior standard deviation your
  chain gives you.

$\sigma_{\rm scatter}$
: the width the point estimate *actually has*: repeat the experiment over
  noise realizations and measure how the recovered $\hat\theta$ fluctuates
  about the truth (systematic offsets included).

$\sigma_{\rm exact}$
: the width of the *exact* analysis of the same data — the information the
  data actually hold. No honest analysis can beat it.

The two diagnostics are the two independent ratios:

$$
\Upsilon \;=\; \frac{\sigma_{\rm scatter}}{\sigma_{\rm quoted}}
\qquad\text{(calibration: is the error bar honest?)}
$$

$$
\Xi \;=\; \frac{\sigma_{\rm quoted}}{\sigma_{\rm exact}}
\qquad\text{(efficiency: was all the information used?)}
$$

$\Upsilon = 1$ means the quoted interval means what it says; $\Upsilon > 1$
is overconfidence (the estimates scatter beyond the bars), $\Upsilon < 1$ is
conservatism. $\Xi = 1$ means the quoted width is the width the data allow;
$\Xi > 1$ means information was thrown away, and $\Xi < 1$ means the
analysis claims information the data do not hold.

Their product is constrained: $\Upsilon\,\Xi =
\sigma_{\rm scatter}/\sigma_{\rm exact} \ge 1$, with equality only for the
exact analysis. That is the information bound — an analysis in the
"forbidden" corner $\Upsilon < 1,\ \Xi < 1$ would outperform the exact
likelihood.

The four realizable regimes, each anchored to a **measured value from the
paper** (repeated experiments, one per row; the grey band is
$\pm\sigma_{\rm exact}$):

<img class="gl-figure" src="_static/upsilon_xi_quadrants.gif" alt="four analyses as repeated experiments, one per quadrant of the Upsilon-Xi plane"/>

## The mathematics

Everything is computable analytically, at Fisher level — no injection
campaigns and no sampling. Let the analysis use a model covariance
$\Sigma'$ while the data are truly drawn with covariance $Q$, with
independent channels $c$ of identical covariance ($(\cdot)^{+}$ denotes the
pseudo-inverse on the observed subspace — the gated covariance is singular,
exactly as in the paper). For the signal parameters, three matrices decide
everything:

$$
F'_{ab} = \sum_c 2\,{\rm Re}\!\left[\partial_a h_c^\dagger\, \Sigma'^{+}\,
          \partial_b h_c\right],
\qquad
M_{ab} = \sum_c 2\,{\rm Re}\!\left[\partial_a h_c^\dagger\, \Sigma'^{+} Q\,
         \Sigma'^{+}\, \partial_b h_c\right],
$$

$$
F_{ab} = \sum_c 2\,{\rm Re}\!\left[\partial_a h_c^\dagger\, Q^{+}\,
         \partial_b h_c\right].
$$

(These are the complex frequency-domain forms the pipeline evaluates; the
paper's real time-domain expressions, Eqs. 76–82 there, are the same
quantities under the DFT.)

$F'$ is the Fisher matrix *the model believes*: the quoted covariance is
$F'^{-1}$. But the maximum-likelihood estimate is a linear functional of the
data, and the data carry $Q$, not $\Sigma'$ — so the estimate actually
scatters with the sandwich covariance $F'^{-1} M F'^{-1}$ (Godambe–White;
the paper's Eq. 80). And $F$ is the Fisher matrix of the exact analysis.
Per parameter $a$:

$$
\Upsilon_a = \sqrt{\frac{\left[F'^{-1} M F'^{-1}\right]_{aa}}
                        {\left[F'^{-1}\right]_{aa}}},
\qquad
\Xi_a = \sqrt{\frac{\left[F'^{-1}\right]_{aa}}
                   {\left[F^{-1}\right]_{aa}}}.
$$

When the model is correct ($\Sigma' = Q$) all three matrices coincide and
$\Upsilon_a = \Xi_a = 1$ identically. For the noise parameters the same
construction runs through the score of the Gaussian likelihood, with one
addition: a mis-specified noise model is also *biased*, and the systematic
offset $b_a$ is folded into the scatter,

$$
\Upsilon_a^2 = \frac{{\rm Var}[\hat\lambda_a] + b_a^2}
                    {\left[F'^{-1}\right]_{aa}},
$$

so noise-sector overconfidence from bias and from underquoted widths are
caught by the same number. This is exactly the paper's Eq. (78): its first
term is the score-covariance sandwich ($\mathrm{Var}[\hat\lambda]$), and its
second — the product of two mean-score traces — is the outer product of the
leading-order bias, $b_a b_b$.

In this repository the implementation is
[`paper/diagnostics.py`](https://github.com/FedericoPozzoli/gaplike/blob/main/paper/diagnostics.py)
(`ApproxModel.upsilon_s`, `.xi_s`, `.upsilon_n`, `.xi_n`, plus the
Godambe–White sandwich and KL pseudo-true bias for the nonlinear noise
sector); `paper/dump_upsxi.py` tabulates every scenario and model tier into
`paper/results/upsxi_dump.json`.

## Why not just a PP plot?

A PP plot answers one question: over many injections, does the true value
fall inside the quoted $p$-credible interval a fraction $p$ of the time? For
Gaussian posteriors of quoted width $\sigma_{\rm quoted}$ and true scatter
$\sigma_{\rm scatter}$, the expected curve is, with
$z_p = \Phi^{-1}\!\big(\tfrac{1+p}{2}\big)$ (unbiased case),

$$
C(p) \;=\; 2\,\Phi\!\left(\frac{z_p}{\Upsilon}\right) - 1 .
$$

The right-hand side contains $\Upsilon$ and nothing else. **A PP plot
measures calibration only: $\Xi$ does not appear.** An analysis quoting
intervals ten times wider than the data allow, but calibrated, produces a
diagonal PP plot indistinguishable from the exact analysis — as the second
panel below shows with the paper's own convolved-diagonal case
($\bar\Upsilon_s = 0.92$, $\bar\Xi_s = 9.3$ in scenario C):

<img class="gl-figure" src="_static/upsilon_xi_pp.gif" alt="the same four analyses through a PP plot: only Upsilon is visible"/>

Three practical advantages over the PP plot follow:

1. **The blind spot is closed.** $\Xi$ measures precisely the thing a PP
   plot cannot see: how much information the analysis wastes. The paper's
   drastic-comb scenario is the cautionary tale — the convolved diagonal
   passes any PP test while quoting intervals 8.6–10.2× wider than the exact
   analysis of the *same data*.
2. **No injection campaign.** A PP plot needs hundreds of end-to-end
   analyses before its staircase settles (watch the early frames above);
   $\Upsilon$ and $\Xi$ are closed-form Fisher-level quantities, evaluated
   once, exactly — and small miscalibrations that finite-injection noise
   would bury are visible immediately.
3. **Per parameter, and signed.** A sagging PP curve says "something is
   off"; $(\Upsilon_a, \Xi_a)$ says *which* parameter, in *which* direction
   (overconfident vs conservative, inflated vs too narrow), and by *how
   much*.

## How to read them

| $\Upsilon$ | $\Xi$ | verdict | paper anchor (Table III) |
|---|---|---|---|
| $\simeq 1$ | $\simeq 1$ | honest **and** optimal | exact time domain (1 by construction) |
| $< 1$ | $\gg 1$ | calibrated but inflated: right answer, loose bars | convolved diagonal, scenario C, signal sector: $\bar\Upsilon_s = 0.92$, $\bar\Xi_s = 9.3$ |
| $> 1$ | $< 1$ | overconfident: bars too small to be true | $W_c$-scaled Whittle, scenario B, signal sector: $\bar\Upsilon_s = 1.47$, $\bar\Xi_s = 0.82$ |
| $> 1$ | $> 1$ | the worst of both: wide *and* wrong | convolved diagonal, scenario C, $\lambda_{\rm tm}$: 1.23, 13.7 |

Signal-sector entries are the paper's averages over the 11 signal
parameters; per-parameter values (Fig. 7 of the paper) regenerate with
`paper/dump_upsxi.py`.

Rules of thumb:

- **$\Upsilon$ is the honesty factor of the error bar.** Multiplying the
  quoted width by $\Upsilon$ restores (approximate) calibration — so
  $\Upsilon = 1.5$ means every quoted interval should really be 1.5× wider.
- **$\Xi$ is the price in effective sensitivity.** Because both the
  curvature and the score variance are quadratic in
  $\partial h/\partial\theta$, an analysis with $\Xi \gg 1$ behaves, at
  leading order, as though the SNR were smaller by that factor — an
  approximation with $\Xi \approx 10$ turns an SNR-300 event into an
  effectively SNR-30 one, at every loudness.
- **The product is the total price.** $\Upsilon\,\Xi =
  \sigma_{\rm scatter}/\sigma_{\rm exact}$ compares your estimator head-on
  with the exact analysis; 1 is attainable only by being both calibrated and
  optimal.
- **Failure modes differ in kind.** The convolved diagonal fails by
  *inflation* ($\Upsilon \lesssim 1$, $\Xi \gg 1$): not a confidently wrong
  answer, but a correct answer with error bars an order of magnitude too
  loose. The Whittle tiers fail by *overconfidence* ($\Upsilon > 1$,
  $\Xi < 1$) — and in the noise sector, catastrophically so (sandwich
  $\Upsilon$ up to $\sim 10^2$, driven by a pseudo-true bias).

## Explore

The whole picture is interactive:
<a href="_static/upsilon_xi_explorer.html">**the Υ–Ξ explorer**</a> has
sliders for $\Upsilon$ and $\Xi$ (they stop at the information bound
$\Upsilon\,\Xi \ge 1$), a draggable point on the $(\Xi, \Upsilon)$ plane
with the forbidden region hatched out, the paper anchors click-to-load, a
systematic-offset dial, and live repeated-experiment and PP views. It is a
single self-contained HTML file
([`assets/upsilon_xi_explorer.html`](https://github.com/FedericoPozzoli/gaplike/blob/main/assets/upsilon_xi_explorer.html))
— open it in any browser, mid-talk included.

Both animations are rendered by `notebooks/anim_upsilon_xi.py` (numpy /
scipy / matplotlib only, deterministic); the anchor values regenerate with
`paper/dump_upsxi.py`. See [Reproducing the paper](reproducing.md).
