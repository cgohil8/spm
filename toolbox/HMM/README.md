# HMM toolbox

Variational Bayesian hidden Markov models (HMMs) with multivariate Gaussian
observations, for M/EEG. One group-level model is fitted to all sessions;
the data are read from SPM datasets, and the toolbox covers the steps from
sensor data to state spectra, power maps and networks. `Contents.m` lists
the functions and `examples/` has complete pipelines (see
`examples/README.md`).

## Model

Data $\mathbf y_t \in \mathbb R^D$ are split into segments (sessions or
trials), each a separate Markov chain over states $s_t \in \{1,\dots,K\}$:

$$
\begin{aligned}
\boldsymbol\pi &\sim \mathrm{Dir}(\boldsymbol\alpha_0), & s_\text{first} &\sim \mathrm{Cat}(\boldsymbol\pi),\\
\mathbf A_{i\cdot} &\sim \mathrm{Dir}(\boldsymbol\beta_{0,i\cdot}), & s_t \mid s_{t-1}=i &\sim \mathrm{Cat}(\mathbf A_{i\cdot}),\\
\boldsymbol\Lambda_k &\sim \mathcal W(\mathbf W_0,\nu_0), & \boldsymbol\mu_k \mid \boldsymbol\Lambda_k &\sim \mathcal N(\mathbf m_0,(\kappa_0\boldsymbol\Lambda_k)^{-1}),\\
& & \mathbf y_t \mid s_t = k &\sim \mathcal N(\boldsymbol\mu_k,\boldsymbol\Lambda_k^{-1}).
\end{aligned}
$$

$\mathbf W$ is the Wishart scale matrix ($\mathbb E[\boldsymbol\Lambda]=\nu\mathbf W$).
With `zeromean`, $\boldsymbol\mu_k = \mathbf 0$ and the states differ only
in their covariance; this is the default with time-delay embedding
(TDE-HMM).

## Inference

The posterior is approximated by
$q(s_{1:T})\,q(\boldsymbol\pi)\,q(\mathbf A)\prod_k q(\boldsymbol\mu_k,\boldsymbol\Lambda_k)$,
with Dirichlet and Normal-Wishart factors; $q(s_{1:T})$ is not factorised
over time.

- **E-step:** forward-backward with $\exp\mathbb E_q[\log\boldsymbol\pi]$,
  $\exp\mathbb E_q[\log\mathbf A]$ (rows sum to less than one and are not
  renormalised) and the expected log-likelihoods $\mathbb E_q[\log\mathcal N(\mathbf y_t\mid\boldsymbol\mu_k,\boldsymbol\Lambda_k^{-1})]$.
  The Viterbi path on the same terms is the mode of $q(s_{1:T})$.
- **M-step:** conjugate updates from the state-weighted counts, transition
  counts and first and second moments of the data.
- **Free energy:** $F = \sum_\text{segments}\log\tilde Z - \mathrm{KL}[q(\theta)\,\|\,p(\theta)]$
  after each E-step; with batch inference it never decreases. $F + \log K!$ approximates the
  log evidence (`HMM.Fperm`) and is used to compare different $K$.
- **Initialisation:** random state paths followed by an M-step; the best
  of `ninit` short fits is continued. `bestof` runs several complete fits.
  Restart $r$ uses substream $r$ of a local random stream seeded with
  `seed`, so results are reproducible and do not depend on parallel
  workers.
- **Missing samples** (bad samples of M/EEG data) contribute no
  observations, but the chain runs through them; they are not segment
  breaks.
- **Stochastic inference** (`inference = 'stochastic'`): segments are cut
  into chunks with buffers; natural parameters are updated with
  mini-batches of chunks, $\boldsymbol\eta \leftarrow (1-\rho_i)\boldsymbol\eta + \rho_i(\boldsymbol\eta_0 + \tfrac{M}{|\mathcal B|}\mathbf t_\mathcal B)$,
  $\rho_i = (i+\text{delay})^{-\text{forget}}$. The final $F$ and state
  probabilities come from one exact E-step over the whole segments.

## Priors

Defaults (`spm_hmm_set_priors`), for data z-scored within each session:

| Parameter | Default |
|---|---|
| $\nu_0$ | $D+2$ |
| $\mathbf W_0$ | such that $\mathbb E_\text{prior}[\boldsymbol\Sigma_k] = \mathrm{diag}$(channel variances) |
| $\mathbf m_0$, $\kappa_0$ | mean of the data, 1 |
| $\boldsymbol\alpha_0$ | 1 |
| $\boldsymbol\beta_0$ | 1 off the diagonal, $1+\delta$ on it, with $\delta = K(L-1)-L$ for an expected lifetime of $L$ samples ($\mathbb E_\text{prior}[A_{ii}] = 1-1/L$) |

## M/EEG data

`spm_eeg_hmm` reads SPM datasets: each good trial is a segment (one per
dataset for continuous data), samples marked bad are missing, channels are
z-scored within each dataset, and the lifetime prior is given in seconds.
Optional PCA (`pca`) and time-delay embedding with PCA (`tde`) are applied
as the data are read; the prepared data are held in memory when they fit
(`inmemory`). State probabilities are written to new datasets, and
`spm_hmm_params` returns the parameters, also mapped back to the channels.

Speed: `spm_hmm_compile` builds an optional C version of the
forward-backward recursion (same results). With an open parallel pool,
`parallel` runs restarts and fits on the workers and shares the segments
of each E-step between them.

## References

- Beal (2003). Variational algorithms for approximate Bayesian inference. PhD thesis, UCL, ch. 3.
- Bishop (2006). Pattern recognition and machine learning, §10.2.
- Foti, Xu, Laird, Fox (2014). Stochastic variational inference for hidden Markov models. NeurIPS.
- Vidaurre et al. (2018). Discovering dynamic brain networks from big data in rest and task. NeuroImage 180.
- Gohil et al. (2024). osl-dynamics, a toolbox for modeling fast dynamic brain activity. eLife 12.
