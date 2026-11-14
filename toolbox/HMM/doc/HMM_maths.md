# Variational Bayesian HMM with Gaussian observations: maths note

Design note for `toolbox/HMM`. Every update and free-energy term the code will
implement is listed here. Each one has been checked numerically by
`hmm_maths_check.m`, which sits in this folder. Section 13 lists those checks.

---

## 1. Notation and generative model

- Data: $\mathbf{y}_t \in \mathbb{R}^D$ for $t = 1,\dots,T$, split into **segments**
  (sessions, continuous recordings, or trials). Each segment is a separate Markov chain.
- Hidden states: $s_t \in \{1,\dots,K\}$.

$$
\begin{aligned}
\boldsymbol\pi &\sim \mathrm{Dir}(\boldsymbol\alpha_0), &
s_{\text{first}} \mid \boldsymbol\pi &\sim \mathrm{Cat}(\boldsymbol\pi)\quad\text{(first sample of every segment)}\\
\mathbf{A}_{i\cdot} &\sim \mathrm{Dir}(\boldsymbol\beta_{0,i\cdot}), &
s_t \mid s_{t-1}=i,\mathbf{A} &\sim \mathrm{Cat}(\mathbf{A}_{i\cdot})\\
\boldsymbol\Lambda_k &\sim \mathcal{W}(\mathbf{W}_0,\nu_0), &
\boldsymbol\mu_k \mid \boldsymbol\Lambda_k &\sim \mathcal{N}\!\left(\mathbf{m}_0,(\kappa_0\boldsymbol\Lambda_k)^{-1}\right)\\
& & \mathbf{y}_t \mid s_t=k &\sim \mathcal{N}\!\left(\boldsymbol\mu_k,\boldsymbol\Lambda_k^{-1}\right)
\end{aligned}
$$

**Zero-mean option.** Set $\boldsymbol\mu_k \equiv \mathbf{0}$ and keep only $\boldsymbol\Lambda_k \sim \mathcal{W}(\mathbf{W}_0,\nu_0)$.
This is used for time-delay-embedded (TDE) and covariance-only models.

**Wishart convention.** $\mathbf{W}$ is the *scale* matrix, so $\mathbb{E}[\boldsymbol\Lambda] = \nu\mathbf{W}$. The density is

$$
\log\mathcal{W}(\boldsymbol\Lambda\mid\mathbf{W},\nu)=\tfrac{\nu-D-1}{2}\log|\boldsymbol\Lambda|-\tfrac12\mathrm{tr}(\mathbf{W}^{-1}\boldsymbol\Lambda)
-\tfrac{\nu D}{2}\log 2-\tfrac{\nu}{2}\log|\mathbf{W}|-\log\Gamma_D(\tfrac{\nu}{2}),
$$

with $\log\Gamma_D(a)=\tfrac{D(D-1)}{4}\log\pi+\sum_{d=1}^{D}\log\Gamma\!\left(a+\tfrac{1-d}{2}\right)$ and
$\psi_D(a)=\sum_{d=1}^{D}\psi\!\left(a+\tfrac{1-d}{2}\right)$. A proper prior needs $\nu > D-1$.
(SPM's `spm_lg_gamma` gets $\log\Gamma_D$ wrong; see §14.)

## 2. Variational family

$$
q(s_{1:T},\boldsymbol\theta)=q(s_{1:T})\;q(\boldsymbol\pi)\;q(\mathbf{A})\;\prod_{k=1}^K q(\boldsymbol\mu_k,\boldsymbol\Lambda_k)
$$

- $q(s_{1:T})$ is **not** factorised over time. Its optimum is a Markov chain, so we infer it exactly
  by forward–backward.
- $q(\boldsymbol\pi)=\mathrm{Dir}(\mathbf{a})$, and $q(\mathbf{A}_{i\cdot})=\mathrm{Dir}(\mathbf{B}_{i\cdot})$ independently for each row.
- $q(\boldsymbol\mu_k,\boldsymbol\Lambda_k)=\mathcal{N}(\boldsymbol\mu_k\mid\mathbf{m}_k,(\kappa_k\boldsymbol\Lambda_k)^{-1})\,\mathcal{W}(\boldsymbol\Lambda_k\mid\mathbf{W}_k,\nu_k)$.
  This is the joint Normal–Wishart posterior, which is the exact conditional form. We do **not** factorise $\boldsymbol\mu_k$ and $\boldsymbol\Lambda_k$ apart.

Given $q(s)$, the parameter factors are conditionally independent. So one M-step that updates all of them
together is an exact coordinate-ascent step.

## 3. Expectations needed

$$
\begin{aligned}
\log\tilde\pi_k &= \psi(a_k)-\psi\!\left(\textstyle\sum_j a_j\right) &&= \mathbb{E}_q[\log\pi_k]\\
\log\tilde A_{ij} &= \psi(B_{ij})-\psi\!\left(\textstyle\sum_j B_{ij}\right) &&= \mathbb{E}_q[\log A_{ij}]\\
\mathbb{E}_q\log|\boldsymbol\Lambda_k| &= \psi_D(\nu_k/2)+D\log 2+\log|\mathbf{W}_k|\\
\mathbb{E}_q\!\left[(\mathbf{y}-\boldsymbol\mu_k)^\top\boldsymbol\Lambda_k(\mathbf{y}-\boldsymbol\mu_k)\right] &= D/\kappa_k+\nu_k(\mathbf{y}-\mathbf{m}_k)^\top\mathbf{W}_k(\mathbf{y}-\mathbf{m}_k)
\end{aligned}
$$

The **expected log emission** is then

$$
\log\tilde b_{tk}=\mathbb{E}_q\log\mathcal{N}(\mathbf{y}_t\mid\boldsymbol\mu_k,\boldsymbol\Lambda_k^{-1})
=-\tfrac D2\log 2\pi+\tfrac12\mathbb{E}_q\log|\boldsymbol\Lambda_k|-\tfrac12\!\left[\tfrac{D}{\kappa_k}+\nu_k(\mathbf{y}_t-\mathbf{m}_k)^\top\mathbf{W}_k(\mathbf{y}_t-\mathbf{m}_k)\right].
$$

In the zero-mean case, drop $D/\kappa_k$ and set $\mathbf{m}_k=\mathbf{0}$.

$\tilde{\mathbf{A}}=\exp(\log\tilde{\mathbf{A}})$ has rows summing to **less than 1**. **It must not be renormalised.**
The sub-normalised matrix is exactly what makes $q(s)$ optimal. Renormalising it is the approximation HMM-MAR makes.

## 4. Update of q(s): forward–backward

$q(s_{1:T}) \propto \prod_{\text{seg}} \tilde\pi_{s_\text{first}}\prod_t \tilde A_{s_{t-1}s_t}\prod_t \tilde b_{t s_t}$.
Per segment we use a scaled recursion that has no clamping and gives an exact log-normaliser:

$$
M_t=\max_k\log\tilde b_{tk},\qquad \hat b_{tk}=\exp(\log\tilde b_{tk}-M_t)
$$

$$
\begin{aligned}
\hat{\boldsymbol\alpha}_1 &= \tilde{\boldsymbol\pi}\odot\hat{\mathbf b}_1/c_1, &
\hat{\boldsymbol\alpha}_t &= (\hat{\boldsymbol\alpha}_{t-1}\tilde{\mathbf A})\odot\hat{\mathbf b}_t/c_t \quad(c_t\text{ normalises})\\
\hat{\boldsymbol\beta}_T &= \mathbf 1, &
\hat{\boldsymbol\beta}_t &= \tilde{\mathbf A}\,(\hat{\mathbf b}_{t+1}\odot\hat{\boldsymbol\beta}_{t+1})/c_{t+1}\\
\gamma_{tk} &= \hat\alpha_{tk}\hat\beta_{tk}, &
\xi_t(i,j) &= \hat\alpha_{t-1,i}\,\tilde A_{ij}\,\hat b_{tj}\,\hat\beta_{tj}/c_t\\
\log\tilde Z_{\text{seg}} &= \textstyle\sum_t(\log c_t+M_t)
\end{aligned}
$$

The Viterbi path computed on $(\log\tilde{\boldsymbol\pi},\log\tilde{\mathbf A},\log\tilde{\mathbf b})$ is exactly the **mode of $q(s_{1:T})$**.

## 5. Update of q(θ): sufficient statistics and conjugate updates

Let $w_t\in\{0,1\}$ mark observed samples (see §7). The sufficient statistics, summed over segments, are:

$$
\begin{aligned}
g^{(1)}_k &= \textstyle\sum_{\text{seg}}\gamma_{\text{first},k}, &
\Xi_{ij} &= \textstyle\sum_{t\,\text{not first}}\xi_t(i,j)\\
N_k &= \textstyle\sum_t w_t\gamma_{tk}, &
\mathbf S^{(1)}_k &= \textstyle\sum_t w_t\gamma_{tk}\mathbf y_t, \qquad
\mathbf S^{(2)}_k = \textstyle\sum_t w_t\gamma_{tk}\mathbf y_t\mathbf y_t^\top
\end{aligned}
$$

The updates are:

$$
\begin{aligned}
\mathbf a &= \boldsymbol\alpha_0+\mathbf g^{(1)}, \qquad \mathbf B=\boldsymbol\beta_0+\boldsymbol\Xi\\
\kappa_k &= \kappa_0+N_k,\qquad \nu_k=\nu_0+N_k,\qquad \mathbf m_k=(\kappa_0\mathbf m_0+\mathbf S^{(1)}_k)/\kappa_k\\
\mathbf W_k^{-1} &= \mathbf W_0^{-1}+N_k\bar{\boldsymbol\Sigma}_k+\tfrac{\kappa_0N_k}{\kappa_0+N_k}(\bar{\mathbf y}_k-\mathbf m_0)(\bar{\mathbf y}_k-\mathbf m_0)^\top
\end{aligned}
$$

where $\bar{\mathbf y}_k=\mathbf S^{(1)}_k/N_k$ and $\bar{\boldsymbol\Sigma}_k=\mathbf S^{(2)}_k/N_k-\bar{\mathbf y}_k\bar{\mathbf y}_k^\top$.
The last line is algebraically equal to
$\mathbf W_0^{-1}+\mathbf S^{(2)}_k+\kappa_0\mathbf m_0\mathbf m_0^\top-\kappa_k\mathbf m_k\mathbf m_k^\top$.
We use the centred form because it is numerically safer.

**Zero mean:** $\nu_k=\nu_0+N_k$ and $\mathbf W_k^{-1}=\mathbf W_0^{-1}+\mathbf S^{(2)}_k$.

**States with no data.** When $N_k\to0$, the posterior returns to the prior. This needs no special handling.

## 6. Free energy

### 6.1 General form (valid at any point)

$$
F=\underbrace{\mathbf g^{(1)}\!\cdot\log\tilde{\boldsymbol\pi}+\textstyle\sum_{ij}\Xi_{ij}\log\tilde A_{ij}+\sum_k Q_k}_{\mathbb E_q[\log p(Y,s\mid\theta)]}
\;+\;H[q(s)]\;-\;\mathrm{KL}[q(\theta)\,\|\,p(\theta)]
$$

where $Q_k=\sum_t w_t\gamma_{tk}\log\tilde b_{tk}$, computed from the sufficient statistics:

$$
Q_k=N_k\!\left(-\tfrac D2\log2\pi+\tfrac12\mathbb E\log|\boldsymbol\Lambda_k|\right)-\tfrac{DN_k}{2\kappa_k}
-\tfrac{\nu_k}{2}\!\left[\mathrm{tr}(\mathbf W_k\mathbf S^{(2)}_k)-2\mathbf m_k^\top\mathbf W_k\mathbf S^{(1)}_k+N_k\mathbf m_k^\top\mathbf W_k\mathbf m_k\right].
$$

In the zero-mean case, $Q_k=N_k(\cdot)-\tfrac{\nu_k}{2}\mathrm{tr}(\mathbf W_k\mathbf S^{(2)}_k)$.

**Entropy of q(s).** There are two equivalent ways to get it, and both have been verified:

- $H=\sum_\text{seg}\log\tilde Z_\text{seg}-\big[\mathbf g^{(1)}\!\cdot\log\tilde{\boldsymbol\pi}+\sum\Xi\log\tilde{\mathbf A}+\sum_kQ_k\big]$,
  where the bracket is evaluated with the $q(\theta)$ that **was used in the E-step**. The code stores this value at E-step time.
- Chain entropy: $H=-\sum_\text{seg}\sum_k\gamma_{\text{first},k}\log\gamma_{\text{first},k}-\sum_t\sum_{ij}\xi_t(i,j)\log\frac{\xi_t(i,j)}{\gamma_{t-1,i}}$.

### 6.2 Shortcut immediately after an E-step

When $q(s)$ is optimal for the current $q(\theta)$:

$$
F=\sum_\text{seg}\log\tilde Z_\text{seg}-\mathrm{KL}[q(\theta)\,\|\,p(\theta)] .
$$

**This shortcut is not valid after an M-step.** Use 6.1 there, for example in tests that check F after every individual update.

### 6.3 KL terms

$$
\mathrm{KL}=\mathrm{KL}_\text{Dir}(\mathbf a\|\boldsymbol\alpha_0)+\sum_i\mathrm{KL}_\text{Dir}(\mathbf B_{i\cdot}\|\boldsymbol\beta_{0,i\cdot})+\sum_k\mathrm{KL}_\text{NW}(k)
$$

$$
\mathrm{KL}_\text{Dir}(\mathbf a\|\mathbf a_0)=\log\Gamma(\textstyle\sum a)-\sum\log\Gamma(a_j)-\log\Gamma(\sum a_0)+\sum\log\Gamma(a_{0j})+\sum_j(a_j-a_{0j})\big(\psi(a_j)-\psi(\sum a)\big)
$$

$$
\mathrm{KL}_\mathcal{W}=\tfrac{\nu_0}{2}\big(\log|\mathbf W_0|-\log|\mathbf W|\big)+\tfrac{\nu}{2}\big(\mathrm{tr}(\mathbf W_0^{-1}\mathbf W)-D\big)
+\log\Gamma_D(\tfrac{\nu_0}{2})-\log\Gamma_D(\tfrac{\nu}{2})+\tfrac{\nu-\nu_0}{2}\psi_D(\tfrac{\nu}{2})
$$

$$
\mathrm{KL}_\text{NW}=\mathrm{KL}_\mathcal{W}+\tfrac12\Big[D\big(\tfrac{\kappa_0}{\kappa}-\log\tfrac{\kappa_0}{\kappa}-1\big)+\kappa_0\nu(\mathbf m-\mathbf m_0)^\top\mathbf W(\mathbf m-\mathbf m_0)\Big]
$$

In the zero-mean case use $\mathrm{KL}_\mathcal{W}$ only.

### 6.4 Exactness check for K = 1

With $K=1$, the converged F equals the exact log evidence:

$$
\log p(Y)=-\tfrac{ND}{2}\log\pi+\log\Gamma_D(\tfrac{\nu}{2})-\log\Gamma_D(\tfrac{\nu_0}{2})+\tfrac{\nu_0}{2}\log|\mathbf W_0^{-1}|-\tfrac{\nu}{2}\log|\mathbf W^{-1}|+\tfrac D2\log\tfrac{\kappa_0}{\kappa}.
$$

Drop the last term in the zero-mean case.

### 6.5 Comparing models with different K

F bounds the evidence of one labelling of the states. When the posterior modes are well separated,
$\log p(Y\mid K)\approx F+\log K!$ (Bishop 2006, §10.2.4). Add $\log K!$ whenever F is compared across K.

## 7. Segments, missing samples and padding

**Segments.** Sessions and trials are separate chains, and each starts from $\tilde{\boldsymbol\pi}$.

**Bad samples are missing data, not segment breaks.** Set $\log\tilde b_{tk}=0$ and $w_t=0$.
The chain carries on through the gap, so transitions inside the gap still add to $\Xi$, and $\log\tilde Z$ includes the gap.
This is the exact variational treatment of observations that are missing at random. Splitting the chain at bad samples
would add fake segment starts and distort $q(\boldsymbol\pi)$.

**Padding** (used to give chunks equal length) must be **neutral**. A padded step uses the **identity transition matrix** and
$\log\tilde b=0$, and contributes nothing to any statistic. Padding with $\log\tilde b=0$ and the normal $\tilde{\mathbf A}$
is *not* neutral: because the rows of $\tilde{\mathbf A}$ sum to less than 1, it shifts $\gamma$ by about $10^{-2}$ in the check (§13 E).

## 8. Stochastic variational inference

### 8.1 Natural-parameter coordinates

All global factors are written in coordinates that are **affine** in the sufficient statistics, so $\boldsymbol\eta=\boldsymbol\eta_0+\mathbf t$:

| factor | coordinates $\boldsymbol\eta$ | prior $\boldsymbol\eta_0$ | statistic $\mathbf t$ |
|---|---|---|---|
| $q(\boldsymbol\pi)$ | $\mathbf a$ | $\boldsymbol\alpha_0$ | $\mathbf g^{(1)}$ |
| $q(\mathbf A)$ | $\mathbf B$ | $\boldsymbol\beta_0$ | $\boldsymbol\Xi$ |
| $q(\boldsymbol\mu_k,\boldsymbol\Lambda_k)$ | $\nu_k,\ \kappa_k,\ \kappa_k\mathbf m_k,\ \mathbf W_k^{-1}+\kappa_k\mathbf m_k\mathbf m_k^\top$ | $\nu_0,\ \kappa_0,\ \kappa_0\mathbf m_0,\ \mathbf W_0^{-1}+\kappa_0\mathbf m_0\mathbf m_0^\top$ | $N_k,\ N_k,\ \mathbf S^{(1)}_k,\ \mathbf S^{(2)}_k$ |
| zero mean | $\nu_k,\ \mathbf W_k^{-1}$ | $\nu_0,\ \mathbf W_0^{-1}$ | $N_k,\ \mathbf S^{(2)}_k$ |

**Implementation detail.** The code accumulates $\mathbf S^{(1)}_k,\mathbf S^{(2)}_k$ about $\mathbf m_0$, i.e. using $\mathbf y_t-\mathbf m_0$.
In that shifted frame the prior coordinates become $\kappa_0\mathbf m_0\to\mathbf 0$ and $\mathbf W_0^{-1}+\kappa_0\mathbf m_0\mathbf m_0^\top\to\mathbf W_0^{-1}$.
This is the same model, but it avoids cancellation in $\mathbf W^{-1}=\mathbf P-\kappa\mathbf m\mathbf m^\top$ when the data are not centred.

Blending in any affine re-parameterisation of the natural parameters is the natural-gradient step (Hoffman et al. 2013).
Blending $\mathbf m_k$, $\mathbf W_k$ or $\mathbf W_k^{-1}$ directly is **not**.
To recover the usual parameters: $\mathbf m=(\kappa\mathbf m)/\kappa$ and $\mathbf W^{-1}=\mathbf P-\kappa\mathbf m\mathbf m^\top$, where $\mathbf P$ is the last coordinate.

### 8.2 The update

1. Cut every segment into **chunks**: intervals $[a_c,b_c]$ that tile the segment. Let $M$ be the total number of chunks.
2. Each epoch, take a random permutation of the chunks and split it into mini-batches $\mathcal B$.
3. For each mini-batch:

$$
\hat{\mathbf t}=\frac{M}{|\mathcal B|}\sum_{c\in\mathcal B}\mathbf t_c,\qquad
\boldsymbol\eta\leftarrow(1-\rho_i)\boldsymbol\eta+\rho_i(\boldsymbol\eta_0+\hat{\mathbf t}),\qquad
\rho_i=(i+\tau)^{-r},\ r\in(0.5,1],\ \tau\ge0.
$$

The scale factor $M/|\mathcal B|$ gives an **unbiased** estimate of *every* full-data statistic under uniform chunk sampling,
including $\mathbf g^{(1)}$, even though chunks differ in length. A ratio of sample counts is not unbiased. Do not weight
the sampling either: non-uniform sampling breaks unbiasedness unless importance weights are added.

### 8.3 Chunk windows, buffers and which statistics each chunk owns

- Forward–backward for chunk $c$ runs over the **window** $[a_c-L_\text{buf},\,b_c+L_\text{buf}]$, clipped to the segment.
- **Initial distribution at the window start:**
  - If the window starts at a true segment start, use $\tilde{\boldsymbol\pi}$.
  - Otherwise use the stationary distribution of the row-normalised $\tilde{\mathbf A}$. This is an approximation, and the buffer absorbs it.
- **Ownership.** Chunk $c$ contributes:
  - $\gamma_t$ for $t\in[a_c,b_c]$ (masked by $w_t$);
  - $\xi_t$, the transition $t-1\to t$, for $t\in[a_c,b_c]$ excluding the segment start;
  - $\mathbf g^{(1)}$ only if $a_c$ is a segment start.
- With these rules every quantity is counted exactly once.
- A chunk that does not start at a segment start **needs a left buffer of at least 1**, otherwise the transition into $a_c$ is lost.
- To process chunks in parallel, windows have equal length, with padding as in §7.

### 8.4 Defaults and the initialisation for stochastic inference

| option (`S.svi.`) | default | note |
|---|---|---|
| `chunklength` | 1000 | interior samples per chunk; segments are split into near-equal chunks |
| `buffer` | 100 | samples either side; must be ≥ 1 |
| `batchsize` | 32 | chunks per mini-batch |
| `nepochs` | 20 | random partition of the chunks into mini-batches every epoch |
| `delay`, `forget` | 1, 0.9 | $\rho_i=(i+\text{delay})^{-\text{forget}}$; chosen by the simulation study below |
| `initchunks` | 64 | chunks used for the batch initialisation |
| `Fevery` | 0 | exact F every so many epochs (costs one full E-step) |

**Simulation study of the defaults** (not in CI). Each problem was fitted with 10 epochs, 1000-sample chunks and 32 chunks per batch. The table gives the F gap to batch VB per sample, $\times10^{-6}$; lower is better.

| problem | forget 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|
| $K=6$, $D=10$, zero mean, lifetime 20, $N=2\times10^5$ | 102 | 60 | 39 | 29 | 26 |
| $K=8$, $D=40$, zero mean, lifetime 25, $N=2\times10^5$ | 2171 | 1222 | 783 | 560 | 457 |
| $K=4$, $D=5$, with means, lifetime 10, $N=10^5$ | 162 | 102 | 71 | 56 | 49 |

- With delay 1, a higher forgetting rate was better on every problem.
- Delay 10 helped only at low rates and was worse at 1.0.
- We use 0.9, which is HMM-MAR's default too:
  - it is within 10–20% of 1.0 on every problem;
  - at 1.0 the step size $1/(i+1)$ is a plain running average, which forgets a poor initialisation slowly, and these simulations, which start from good initialisations, would not show that.
- The buffer length (10, 25, 50, 100 or 200 samples) made no difference, to within 0.1 of the values above, on any of the problems.
  - The default stays at 100 to be safe for longer state lifetimes in real data, at about 10% extra cost.
- All of these defaults are provisional until they have been checked on real M/EEG data.

**Initialisation.**
1. Run batch VB (§11) on `initchunks` random chunks, treating each chunk as a separate segment and using the **full-data prior**.
2. Run one E-step on that subset, then set $\boldsymbol\eta=\boldsymbol\eta_0+\frac{M}{M_\text{sub}}\mathbf t_\text{sub}$.
3. Scale $\mathbf g^{(1)}$ first by (number of segments)/$M$, because every subset chunk counts as a segment start but the data contain only that many true starts.

### 8.5 Monitoring and the final free energy

- During SVI, $\log\tilde Z$ per window includes the buffers and $q(s)$ is only approximately optimal, so the free energy reported per batch is a **diagnostic only**.
- After SVI, run **one full E-step over whole segments**, without chunking. This gives the exact $F$ (via 6.2) for the final $q(\theta)$, together with the $\gamma$ to report.
- Only this F is used to choose between restarts or values of K.
- **Sanity check:** with chunks equal to whole segments, $\mathcal B$ equal to all chunks and $\rho=1$, the SVI update reduces exactly to the batch update. This is verified in §13 H.

## 9. Decoding and summary statistics

- The state probabilities are $\gamma$ from the final full E-step.
- The Viterbi path is the mode of $q(s)$, as in §4. It is computed by max-product on $(\log\tilde{\boldsymbol\pi},\log\tilde{\mathbf A},\log\tilde{\mathbf b})$ and checked against enumeration.
- **Summary statistics** are computed per group, for example per dataset, from state sequences. They use the osl-dynamics definitions:
  - a **visit** is a run of consecutive samples in one state, including runs cut off at the start or end of a segment;
  - a **lifetime** is the length of a visit;
  - an **interval** is the number of samples between the end of one visit and the start of the next;
  - the **switching rate** counts entries into a state after the first sample, per second;
  - **fractional occupancy** is the proportion of samples in each state.
- With no missing samples, our results equal osl-dynamics' to $10^{-15}$ (checked on 3 random paths).
- **Missing samples** are not observed time. Each segment is split at missing samples, so no visit or interval spans a gap, and missing samples are left out of the occupancy and switching-rate denominators.

## 9a. M/EEG data

- **Datasets** are read on demand from SPM's memory-mapped `meeg` files.
- **Segments:** each good trial is a segment (one per dataset for continuous data). Bad trials are skipped.
- **Missing samples:** a sample is missing if any selected channel is bad there, according to artefact events, as in `meeg/badsamples`.
- **Channels** are matched by label across datasets. By default all good MEEG/LFP/SRC channels are used.
- **Standardisation** (default on): each channel is z-scored within each dataset, using the mean and SD of its good samples in the selected trials. This matches the data assumption of §10.
- **Lifetime prior:** the sticky-prior lifetime is given in seconds and converted using the sampling rate.
- **Output datasets:** state probabilities are written to new `meeg` datasets with channels `State1..K` of type 'Other'. Trials that were not modelled are filled with NaN.

## 9-pre. Parcellation and leakage correction

Both follow `osl_dynamics.meeg.parcellation`, which is identical to osl-ephys.

**Parcellation** (`spm_eeg_hmm_parcellate` = `parcellate_lcmv`; weights from `spm_hmm_parcel_weights` = `_get_parcel_weights`):
- **Inputs:** a DAiSS scalar LCMV beamformer on a grid, with sensor-space filters $F$ (source $i$: `W{i}*U'`), and an atlas sampled at the source positions by nearest neighbour.
- **Atlas:** a 3D label image becomes one-hot weights over its sorted labels; a 4D image is used as weights.
- **Source covariance:** $FCF^\top$, with $C$ the biased sensor covariance over good samples of good trials.
- **PCA weights:** for parcel $p$, with $s$ the atlas weights scaled to a maximum of 1:
  - take the leading eigenvector $u$ (eigenvalue $d$) of $\mathrm{diag}(s)\,C_p\,\mathrm{diag}(s)$;
  - the mask $m$ is the sources with $s>0.5$ and non-zero variance;
  - $V_p = \mathrm{sign}(\overline{u_m})\cdot\frac{\sum|u_m|\sigma_m}{\sum|u_m|}\cdot u\odot s/\sqrt d$.
  - So each parcel time course is the first principal component of its weighted sources, with standard deviation equal to the $|u|$-weighted mean source SD and a positive mean loading.
- **Centroid method:** the source nearest the weighted centre.
- **Applying the filters:** parcel filters $V^\top F$ are applied to every sample, minus the filtered good-sample mean.
- **Two differences from osl-dynamics:**
  1. osl-dynamics parcellates MNE's weight-normalised filters. DAiSS's LCMV has unit gain, so `S.weightnorm = 'unit-noise-gain'` (the default) scales each filter to unit norm: the unit-noise-gain beamformer for white sensor noise.
  2. osl-dynamics includes grid points with no dipole when scaling probabilistic weights, because its grid covers the whole MNI box. DAiSS grids leave those points out, so 4D probabilistic atlases can be scaled slightly differently. Label atlases are unaffected.
- **Bad samples** are left out of the covariance and mean. They are not zero-filled, and artefact events are kept.

**Symmetric orthogonalisation** (`spm_eeg_hmm_orthogonalise` = `symmetric_orthogonalization(maintain_magnitudes=True)`; Colclough et al. 2015):
- With $A$ the good samples and $D=\mathrm{diag}(\|a_j\|)$: $AD=USV^\top$, and the output is $UV^\top D = AM$ with $M=DVS^{-1}V^\top D$.
- $M$ is computed from $A^\top A$, accumulated in blocks; the eigendecomposition of $DA^\top AD$ gives $V$ and $S^2$. This agrees with the SVD route to $4\times10^{-9}$ relative.
- $M$ is then applied to every sample.
- Each channel keeps its norm, and the channels have zero inner products over the good samples.
- Rank-deficient data raise an error.

**Atlases are not shipped with SPM.** MEG parcellations, such as Glasser52 (CC BY 4.0), are in the osl-files repository, `https://github.com/OHBA-analysis/osl-files`, folder `parcellation/`, with labels and coordinates in `docs/parcellations`. They can also be fetched with `osl-dynamics-download-data`. Cite the atlas authors.

## 9b. Time-delay embedding and PCA (TDE-HMM)

`spm_hmm_tde_pca` wraps any data reader:

- **Embedding.** Row $t$ of the embedded data is $[\mathbf y_{t+o+\ell_1},\dots,\mathbf y_{t+o+\ell_E}]$, where the lags $\ell_e$ are integers and $o=-\min_e\ell_e$.
  - Each segment loses $\max\ell-\min\ell$ samples.
  - A row is missing if any sample it uses is missing.
- **PCA.** The covariance of the embedded data is pooled over all segments, using the observed rows only. The top `npca` eigenvectors are kept, or as many as reach a given fraction of the variance. Each component's sign is fixed by making its largest loading positive.
- **Standardisation.** Components are then z-scored within groups, e.g. datasets, as in osl-dynamics' `tde_pca` followed by `standardize`.
- **Never stored.** Each block is embedded, projected and standardised as it is read. Embedded data are never held in memory or written to disk.
- **Zero-mean states.** As in HMM-MAR and osl-dynamics, a TDE-HMM normally uses zero-mean states, and `spm_eeg_hmm` defaults to them when embedding is requested. States then differ in the covariance of the embedded data, i.e. in their auto- and cross-spectra.
- **Writing results.** `spm_eeg_hmm` writes the state probabilities back at the original samples. The $o$ samples at each end of a segment are NaN.
- **Caveat.** Consecutive embedded rows share most of their samples, so the model's assumption that observations are conditionally independent is violated more strongly than for raw data. As with HMM-MAR and osl-dynamics, F is then over-confident in absolute terms. Comparisons of F between models fitted to the same embedded data remain the intended use.

## 9c. Amplitude envelope (AE-HMM)

`spm_eeg_hmm_envelope` writes a new dataset holding the envelopes of the selected channels. For each trial (or the whole recording) and channel, it follows osl-dynamics' `filter` → `amplitude_envelope` → `moving_average`:

1. **Filter:** a zero-phase Butterworth band-pass of order 5, run in two passes, as `spm_eeg_filter` does. Its interior output is identical to SciPy's `butter` + `filtfilt`; the two pad the ends differently.
2. **Envelope:** $|\mathrm{hilbert}(x)|$, computed with the FFT. Like SciPy's, it spreads the transients at the ends of the record: in the test, the envelope of a constant-amplitude sinusoid is off by 1.6% at 2 s from the ends and by 0.03% at the centre.
3. **Moving average:** a centred moving average over an odd window. Only samples where the full window fits are kept. osl-dynamics trims the rest; we set them to NaN so the output stays aligned with the original samples.

**Bad samples.** As in osl-dynamics, steps 1–3 run straight through bad samples. Artefact events and bad trials are copied to the new dataset, so `spm_eeg_hmm` still treats the original bad samples, and the NaN edges, as missing.

**Fitting.** `spm_eeg_hmm` on the envelope dataset standardises each channel per dataset and fits states with learned means and full covariances, as osl-dynamics does for the AE-HMM.

## 9d. Sign flipping

Source reconstruction leaves the sign of each parcel arbitrary, so it can differ between sessions. A group TDE-HMM needs consistent signs, because sign changes alter the cross-channel covariance. The AE-HMM does not, because envelopes have no sign.

We implement osl-dynamics' algorithm (`osl_dynamics.data.sign_flipping` and `Data.align_channel_signs`) to the same definitions:

- **Embedded covariance** (`spm_hmm_embedded_cov` = `calc_cov`):
  - time-delay embed with $E$ (odd) embeddings, in channel-major order with the most recent sample first;
  - standardise the embedded columns (population SD), then take the covariance normalised by $n-1$;
  - optionally one covariance per frequency band, stacked;
  - embedding is done within segments, and rows that use a bad sample are dropped. osl-dynamics instead drops bad segments before embedding.
- **Objective:** the Pearson correlation between the entries of the flipped session covariance and of the template that lie at least $E$ places above the diagonal, pooled over bands.
  - This leaves out the within-channel entries, which do not depend on the signs.
  - For signs $\mathbf f$ it depends only on $P=\sum_{p<q}f_pf_qA_{pq}$ and $Q=\sum_{p<q}f_pf_qB_{pq}$, where $A$ and $B$ are $D\times D$ sums of $c\,t$ and $c$ over each channel pair's entries.
  - $\mathbf f$ and $-\mathbf f$ give the same value.
- **Search** (`spm_hmm_find_flips` = `find_flips`): greedy single-channel coordinate ascent.
  - It starts from all $+1$, from the signs of the two leading eigenvectors of $A$, and from 3 random sign vectors with a fixed seed. The best result is kept.
  - Our random vectors come from MATLAB's generator, not NumPy's, so they differ from osl-dynamics'.
- **Template** (`spm_hmm_sign_flip` = `align_channel_signs`):
  - **default:** the session with the median summed sign-invariant similarity to the other sessions;
  - or a given template covariance;
  - or **two-stage** (`spm_hmm_build_template` = `build_template`): start from the medoid, then twice align every session to the template and average. With an external template, the flips $\mathbf g$ aligning the dataset template to it are multiplied into every session's flips.
- **M/EEG** (`spm_eeg_hmm_sign_flip` = `sign_flip_mne_raw`):
  - each dataset is one session; band filters run through bad samples;
  - writes `sf_` copies in which only the selected channels are multiplied by their signs.
- **Pipeline:** sign flip the parcel data, then `spm_eeg_hmm` with `S.tde`.

## 9e. State spectra (multitaper)

`spm_hmm_multitaper` follows `osl_dynamics.analysis.spectral.multitaper_spectra` (with `keepdims`). For each session:

1. Standardise each channel (population SD).
2. For each state $k$, multiply the data by $\alpha_k(t)$: state probabilities, or a one-hot Viterbi path.
3. Zero-pad by $\lfloor w/2\rfloor$ at each end.
4. Take $\lfloor T/\text{step}\rfloor$ windows of $w$ samples (default $w=2f_s$, step $=w$).
5. Taper each window with DPSS tapers (time-half-bandwidth 4, 7 tapers, unit energy; `spm_dpss` equals SciPy's to $2\times10^{-15}$) and take the FFT with $n_\text{fft}=w$.
6. Average $\overline{X_m}X_n$ over tapers and windows, then scale by $2/f_s$ and by $T/\sum_t\alpha_k(t)^2$.

- **Frequencies:** those with $f\ge f_\min$ and $f\le f_\max$.
- **Coherence:** $|P_{xy}|/\sqrt{P_{xx}P_{yy}}$ (magnitude), with NaN set to 0.
- **Session weights:** each session's fraction of samples.

**Missing samples.** These are artefacts, and states missing at the edges lost to embedding. Data are standardised over observed samples, then the data and $\alpha$ are set to zero there. $T$ in the scaling includes the missing samples, which exactly compensates for averaging over the zeroed time. osl-dynamics has no missing samples.

`spm_eeg_hmm_spectra` builds the inputs from M/EEG datasets and a model fitted by `spm_eeg_hmm`:
- the session's good trials are concatenated;
- states come from the Viterbi paths or from the state-probability datasets;
- the embedding offset is taken into account.

## 9f. Spectral components (NMF)

`spm_hmm_spectral_components` follows `osl_dynamics.analysis.spectral.decompose_spectra`:
- **Input:** the coherence of every session, state and channel pair (upper triangle, without the diagonal) forms the rows of $X$ [rows × F].
- **Factorisation:** $X\approx W\,\text{comp}$, using `spm_hmm_nmf`.
- **Output:** the components, ordered by their peak frequency and not normalised. Downstream functions divide by their sums.

`spm_hmm_nmf` is sklearn's `non_negative_factorization` with its defaults (Frobenius loss, coordinate descent, no shuffling, tolerance $10^{-4}$, up to 50,000 iterations):
- **Initialisation:** NNDSVDA, filling entries below $10^{-6}$ with $\bar X$.
- **Each iteration:** update $W$, then $H$, one component at a time: $w_t\leftarrow\max(w_t-(WH H^\top_{:,t}-XH^\top_{:,t})/(HH^\top)_{tt},0)$.
- **Stop:** when the summed |projected gradient| relative to the first iteration is at most the tolerance.

**One difference from sklearn.** sklearn computes the NNDSVDA starting point from a *randomised* SVD; osl-dynamics passes `random_state=None`, so its components vary slightly between runs. We use the exact truncated SVD, which the randomised SVD approximates.

## 10. Priors (defaults and rationale)

**Data assumption.** Each channel is standardised per session (z-scored). A single streaming pass over the full dataset
computes the pooled mean $\bar{\mathbf y}$ and the per-channel variances $\mathbf v$ ($\approx\mathbf 1$ after z-scoring).
That pass is needed anyway to check the data.

| parameter | default | rationale |
|---|---|---|
| $\nu_0$ | $D+2$ | The weakest Wishart prior for which $\mathbb E[\boldsymbol\Sigma_k]$ exists. Its weight is $D+2$ pseudo-samples, compared with $N_k\sim10^4$–$10^6$. |
| $\mathbf W_0$ | $[(\nu_0-D-1)\,\boldsymbol\Sigma_0]^{-1}=\boldsymbol\Sigma_0^{-1}$, with $\boldsymbol\Sigma_0=\mathrm{diag}(\mathbf v)$ | Gives $\mathbb E_\text{prior}[\boldsymbol\Sigma_k]=\boldsymbol\Sigma_0$. This sets the scale only; the prior carries **no** correlation structure. It uses variance rather than range, so it is robust to artefacts. For zero-mean models, $\mathbf v$ is the per-channel mean square about zero, because there the state covariance is the second moment. |
| $\mathbf m_0$ | $\bar{\mathbf y}$ ($\approx\mathbf 0$) | Centre of the data. |
| $\kappa_0$ | $1$ | Prior covariance of $\boldsymbol\mu_k$ is $\boldsymbol\Sigma_k$, a unit-information prior. It has negligible effect on the posterior but sets the Occam penalty $\approx\frac D2\log(N_k/\kappa_0)$ per state in F. Check sensitivity when choosing K. |
| $\boldsymbol\alpha_0$ | $\mathbf 1$ | Uniform. |
| $\boldsymbol\beta_0$ | $\beta_0=1$ off the diagonal, $\beta_0+\delta$ on it | Sticky prior set by an expected lifetime $L$ in **samples** (lifetime in seconds × sampling rate). |

**Choosing δ from the expected lifetime.** For a prior mean self-transition of $1-1/L$:

$$
\delta=\beta_0\big(K(L-1)-L\big)\quad\Rightarrow\quad \mathbb E_\text{prior}[A_{ii}]=1-1/L,
$$

which needs $L\ge K/(K-1)$.

- Each row carries $\beta_0L(K-1)$ pseudo-counts. For example, $K=8$ and $L=25$ give 175, against $\sim T/K$ observed transitions per row.
- So with large datasets the prior mainly matters for states that receive almost no data.
- $L$ is the lifetime **at the prior mean**. By Jensen's inequality, $\mathbb E[1/(1-A_{ii})]\ge1/(1-\mathbb E[A_{ii}])$.

**Empirical Bayes.** Only pooled first and second moments are taken from the data, so the double use of the data is negligible.
The code reports prior pseudo-counts next to the data counts, so the user can see how strong the prior is.

## 11. Initialisation

**Principle.** Always initialise $q(s)$ first, as $\gamma$ and $\Xi$, then run an M-step to get $q(\theta)$. Never set $q(\theta)$ by hand.

**Ways to get the initial $q(s)$:**

1. `'random'` (the default; follows HMM-MAR's default initialisation):
   - For each segment, sample a Markov state path with transition matrix $\mathbf P\propto(d-1)\mathbf I+\mathbf 1$ (rows normalised), so $p(\text{stay})=d/(d+K-1)$.
   - $d$ is `S.initdiag`, which defaults to $\min(\operatorname{median}_i T_i/10,\,500)$, as in HMM-MAR.
   - The first state is drawn uniformly.
   - Add $10^{-4}\times$uniform noise to the one-hot path and renormalise it to get $\gamma$.
   - Approximate $\Xi$ by $\sum_t\gamma_{t-1}^\top\gamma_t$.
2. User-supplied $\gamma$, for example from an earlier fit or a group model.
3. *(Possible later additions)* k-means on samples when means are learned, or on sliding-window covariances for zero-mean models.

**Restarts** (as in HMM-MAR). Run $n_\text{init}=5$ short fits of 10 full VB cycles each, and keep the one with the highest F.
Unlike HMM-MAR, our F is the correct bound (§6), the RNG is local, and the extra option of initialising with a smaller K is not implemented.

- All restarts must compute F on the **same data**: the same subset of sessions when the dataset is large.
- *(Not yet implemented)* run `best_of` complete fits and keep the one with the highest final exact F (§8.5).

**Moving from a subset to SVI on all data.** Initialise $\boldsymbol\eta=\boldsymbol\eta_0+\frac{M}{M_\text{sub}}\mathbf t_\text{sub}$.
The subset statistics are rescaled to full-data size, so the starting posterior has the right concentration.
This avoids HMM-MAR's over-counting, where each subject in a group receives the whole group's statistics.

**Reproducibility.** Use a local `RandStream` seeded from options. The global RNG is never changed.

## 11a. Several complete fits

`S.bestof = n` runs $n$ complete fits with seeds `S.seed`, `S.seed+1`, and so on. Each fit includes its own initialisations, so the fits are independent. The fit with the highest final F is kept, and `HMM.Fbestof` records every fit's F.

## 12. Numerical implementation notes

- **Emissions:** compute $\mathbf W_k=\mathbf C_k^\top\mathbf C_k$ by Cholesky. Then $(\mathbf y-\mathbf m)^\top\mathbf W(\mathbf y-\mathbf m)=\|(\mathbf Y-\mathbf m^\top)\mathbf C_k^\top\|^2$ row-wise, at cost $O(TKD^2)$, vectorised over time.
- **Log-determinants** use Cholesky. Never call `det` or `inv` inside KL terms; use solves.
- **Forward–backward:** use the scaled recursion of §4 with subtraction of the per-time maximum. There is no clamping and $\log\tilde Z$ is exact. Run it across chunks in parallel, one $K\times n_\text{chunks}$ operation per time step.
- **Precision matrices:** symmetrise every $\mathbf W_k^{-1}$. If Cholesky fails, **raise an error** instead of adding silent jitter.
- **Dead states:** report $N_k$ and leave them alone; there is no forced pruning in version 1.
- **MATLAB compatibility:** the code must run on R2020b, the oldest release in CI.
- **Speed** (profiled on R2026b, Apple silicon, at a TDE-HMM size: $D=80$, $K=8$, zero mean, $5\times10^5$ samples):
  - one full E-step takes 0.8 s: expected log-likelihood 50%, sufficient statistics 30%, forward–backward 15%;
  - the first two are single BLAS calls per state (about 60 GFLOP/s), and the forward–backward loop costs 0.01 s per $5\times10^4$ samples under MATLAB's JIT;
  - so a MEX implementation would gain little, and none is used;
  - a stochastic epoch with 1000-sample chunks and 100-sample buffers costs about 1.5 full E-steps, because the buffers add 20% plus per-batch overhead;
  - stochastic inference pays off when it needs fewer epochs than batch VB needs cycles, and when only mini-batches are read from disk.

## 13. Verification (MATLAB R2026b)

**Formulas (`hmm_maths_check.m`, then `tests/test_spm_hmm.m`):**

| check | result |
|---|---|
| A. $\mathbb E\log|\boldsymbol\Lambda|$, $\mathrm{KL}_\mathcal W$, $\mathrm{KL}_\text{NW}$, $\mathrm{KL}_\text{Dir}$, $\mathbb E\log\pi_k$, $\mathbb E_q\log\mathcal N$ against Monte Carlo ($2\times10^5$ draws) | all within 1.4 standard errors |
| B. Forward–backward against brute-force enumeration ($K=3$, $T=6$) | $\log\tilde Z$, $\gamma$, $\Xi$ and $H$ agree to $10^{-14}$ |
| B. Identity 6.2 against explicit $\mathbb E[\text{score}]+H-\mathrm{KL}$ | agree to $4\times10^{-14}$ |
| B. $\mathbb E_{q(\theta)}[\log p(Y,s\mid\theta)]=\text{score}(s)$ by Monte Carlo | 0.9 standard errors |
| C. $K=1$: F equals exact log evidence (with mean and zero-mean) | agree to $10^{-13}$ |
| D. Batch VB: F never decreases after **every** E- and M-step (60 iterations, with mean and zero-mean) | smallest step $-2.5\times10^{-11}$ on $|F|\approx2\times10^4$, i.e. round-off |
| E. Padding with identity transitions is neutral; naive padding is not | $3\times10^{-16}$ against $10^{-2}$ |
| F. Chunked forward–backward with buffers against full forward–backward | buffer of 0: $\gamma$ off by $4\times10^{-2}$; buffer ≥ 10: $10^{-14}$ (well-separated simulated states; real data will need longer buffers, which a test will set) |
| G. $\delta$ from lifetime $L$ | prior $\mathbb E[A_{ii}]=1-1/L$ exactly |
| H. SVI with $\rho=1$, full batch, chunks equal to segments, against the batch update | $4.5\times10^{-13}$ |
| H. Bias of $M/|\mathcal B|$ scaling (4000 random batches of segments with unequal lengths) | within Monte Carlo noise |

**Stochastic inference (phase 2):**

| check | result |
|---|---|
| Chunks equal to whole segments: chunked E-step against full E-step (with missing samples) | agree to $10^{-11}$ |
| Chunks of unequal length processed together, against one at a time (padding is neutral) | agree to $10^{-12}$ |
| Owned statistics of 250-sample chunks with a 50-sample buffer, summed, against full-data statistics | agree to $10^{-15}$ (buffer 1: $10^{-2}$; buffer 5: $10^{-5}$) |
| Stochastic inference with $\rho=1$, all chunks in each batch, chunks equal to segments | F never decreases (it is batch VB) |
| Stochastic inference against batch VB (3,500 samples; 250-sample chunks; 4 chunks per batch) | relative F difference $2\times10^{-5}$; >95% of states recovered |
| Benchmark: 40 × 10,000 samples, D = 10, K = 6, zero mean (not in CI) | batch 8.8 s; stochastic 3.7 s for 10 epochs; F gap 0.9–2.4 nats ($2$–$6\times10^{-6}$ per sample), shrinking every epoch |

**Decoding, statistics and M/EEG (phase 3):**

| check | result |
|---|---|
| Viterbi path and score against enumeration ($K=3$, $T=7$, one missing sample) | identical path; score to $10^{-10}$ |
| `spm_hmm_decode` against the state probabilities from the fit | agree to $10^{-12}$ |
| Summary statistics on a hand-worked path (with and without a missing sample, and pooled) | exact |
| Summary statistics against osl-dynamics `post_hoc` (3 random paths, $K=4$; not in CI) | agree to $\le4\times10^{-15}$ |
| Synthetic continuous `meeg` (2 datasets, artefact on one channel, extra 'Other' channel) | right channels; the 51 artefact samples are missing; standardised to mean 0, SD 1; >95% of states decoded correctly; output datasets valid |
| Synthetic epoched `meeg` (bad trial, conditions) | bad trial excluded and written as NaN; condition selection |
| Stochastic inference on `meeg` read from disk | >95% of states decoded correctly |

**Embedding, complete fits and batch interface (phase 4):**

| check | result |
|---|---|
| Embedded rows rebuilt from the PCA reader with a full basis, against explicit construction (unequal lags, missing sample) | agree to $10^{-12}$; exactly the rows using the missing sample are missing |
| PCA: orthonormal components; covariance of projected data equals the eigenvalues; selection by variance fraction; standardisation within groups | to $10^{-10}$ |
| States with identical zero-lag covariance and different autocorrelation (+0.9, 0, −0.9) | zero-mean HMM 44% (chance about 33%); TDE-HMM (lags −3 to 3, 6 components) 97% |
| `bestof`: F is the maximum over fits; the first fit equals a plain fit with the same seed | exact |
| Batch job through `spm_jobman` (TDE, two `meeg` datasets) | model file saved; zero mean by default; >90% decoded correctly; embedding edges NaN; probabilities sum to 1 |

**Amplitude envelope:**

| check | result |
|---|---|
| Envelope dataset against a manual filter → `abs(spm_hilbert)` → `movmean` (valid part) | agree to $10^{-5}$ relative (data stored as float32); edge NaNs and the artefact are read as missing |
| Constant-amplitude sinusoids (amplitudes 3 and 0.5, one with a DC offset), centre of the record | envelope equal to the amplitude to 0.1% |
| Epoched data, channel selection, prefix | edges NaN per trial; bad trial kept; only the selected channels written |
| States that differ only in 10 Hz amplitude: envelope, then `spm_eeg_hmm` with learned means | >90% of samples decoded correctly |

**Sign flipping:**

| check | result |
|---|---|
| Embedded covariance against explicit construction (order, standardisation, missing rows, segments) | agree to $10^{-12}$ |
| Greedy search against exhaustive search over all $2^7$ sign vectors; reduced objective against direct correlation | equal to $10^{-12}$ |
| 5 sessions with planted signs (median-session template) | all sessions end up with the same convention |
| Two-stage and plain alignment to a template from another dataset of the same process | sessions take the other dataset's convention |
| Stacked covariances for 2 frequency bands | shape $2\times N\times N$; signs recovered |
| M/EEG datasets (extra 'Other' channel, an artefact) | only the selected channels flipped; events kept; `sf_` copies |
| Against osl-dynamics on the same 5 simulated sessions (not in CI) | covariances $8\times10^{-15}$; same template session; correlations $7\times10^{-16}$; identical flips (up to the irrelevant global sign); `build_template`: same medoid, template $6\times10^{-15}$ |

**Parcellation and orthogonalisation:**

| check | result |
|---|---|
| PCA weights: single-source parcel gives that source; identical sources give that source; parcel SD equals the $|u|$-weighted mean source SD; positive mean loading; centroid | exact (to $10^{-10}$ or better) |
| Synthetic DAiSS structure (8 sources, 2-parcel label atlas, 12 sensors, artefact) | parcels correlate >0.99 with their source signals; demeaned; unmarking the artefact changes the filters |
| Orthogonalisation of the parcels | zero correlation over good samples ($<10^{-6}$, float32 storage); norms kept; rank deficiency errors |
| Against `osl_dynamics.meeg.parcellation` (not in CI): `_get_parcel_weights` on a probabilistic atlas with overlaps, a no-variance source and a single-source parcel; `symmetric_orthogonalization` | weights $5\times10^{-15}$; orthogonalised data $4\times10^{-9}$ relative |

**State spectra:**

| check | result |
|---|---|
| Unit-variance white noise | PSD $=2/f_s$ (2%); integrates to the variance |
| Sinusoid at 12.5 Hz | peak at 12.5 Hz |
| Coherence: channel with itself; a shared signal; independent noise | 1 exactly; >0.8; <0.2 |
| States with and without a 10 Hz oscillation (one-hot states) | 10 Hz power >20× higher in the oscillating state |
| 10% of samples missing | spectrum unchanged (3%) |
| AE-HMM fit on a dataset, spectra from Viterbi paths and from probability datasets | 10 Hz power per state and channel correlates >0.95 with the simulated amplitude² |
| Against osl-dynamics `multitaper_spectra` (2 sessions, soft 3-state probabilities, overlapping windows, frequency range; static; not in CI) | PSD $3\times10^{-7}$ relative, coherence $5\times10^{-7}$, static $2\times10^{-7}$ (osl-dynamics computes in single precision); weights exact |

**Spectral components (NMF):**

| check | result |
|---|---|
| $X=W_0H_0$ with two spectral bumps | relative error $<10^{-3}$; components correlate >0.999 with $H_0$ |
| Coherence built from two spectral components | components peak at the planted frequencies, in order |
| Against sklearn's coordinate descent from the same NNDSVDA start (not in CI) | $2$–$5\times10^{-14}$ relative |
| Against osl-dynamics `decompose_spectra` as it runs (randomised SVD, single precision, 5 seeds; not in CI) | $10^{-5}$–$10^{-4}$ relative, varying with the seed |

The checks in all nine tables are in `tests/test_spm_hmm.m`, except the seven marked "not in CI".

## 14. Problems found in existing SPM helpers (do not reuse)

- **`spm_lg_gamma`** computes $\sum_\alpha\log\Gamma(b+\tfrac{\alpha-1}{2})$. The multivariate gamma is $\sum_\alpha\log\Gamma(b-\tfrac{\alpha-1}{2})$, which is what the Box & Tiao reference in its header gives.
  For example, `spm_lg_gamma(3,2.5)` returns 3.896, while $\log\Gamma_3(2.5)=1.881$.
- **`spm_kl_wishart`** inherits that error. It also uses $\mathbb E_p\log|\boldsymbol\Lambda|$ in the cross-entropy term, where $\mathbb E_q\log|\boldsymbol\Lambda|$ is needed.
  In the check it returns 15.94 against a true 18.54. Correcting both errors reproduces the true value exactly.
- **Who is affected:** `spm_mix` (free energy), `spm_mar` (free energy, used to select model order), `spm_mlm_bayes`.
- **Plan:** `toolbox/HMM` uses its own private, tested helpers. The shared helpers should be fixed in a separate pull request, because the fix changes free-energy values in the functions above.

## 15. Where each part is implemented

| section | function |
|---|---|
| §1, §10 priors | `spm_hmm_set_priors` |
| §3 expected log emission | `spm_hmm_expected_loglik` |
| §4 forward–backward | `spm_hmm_forward_backward` |
| §4–5 E-step: $q(s)$, sufficient statistics, $H$ | `spm_hmm_infer_states` |
| §5, §8.1–8.2 M-step, batch or stochastic | `spm_hmm_update_params` (with `private/natural_to_posterior`) |
| §7, §8.3 chunks, buffers, ownership, padding | `spm_hmm_make_chunks`, `spm_hmm_infer_states_chunked` |
| §8.2, §8.4 stochastic loop and initialisation | `private/fit_stochastic` |
| §6.3 KL terms | `spm_hmm_kl_divergence` |
| §6.1 general F | `spm_hmm_free_energy` |
| §6.2, §6.5, §11 fitting loop, restarts, $F+\log K!$ | `spm_hmm` |
| §9 decoding, Viterbi | `spm_hmm_decode`, `spm_hmm_viterbi` |
| §9 summary statistics | `spm_hmm_summary_stats` |
| §9e state spectra | `spm_hmm_multitaper`, `spm_eeg_hmm_spectra` |
| §9f spectral components (NMF) | `spm_hmm_nmf`, `spm_hmm_spectral_components` |
| §9a M/EEG | `spm_hmm_meeg_reader`, `spm_eeg_hmm` |
| §9-pre parcellation, orthogonalisation | `spm_eeg_hmm_parcellate`, `spm_hmm_parcel_weights`, `spm_eeg_hmm_orthogonalise` |
| §9b time-delay embedding + PCA | `spm_hmm_tde_pca` |
| §9c amplitude envelope | `spm_eeg_hmm_envelope` |
| §9d sign flipping | `spm_hmm_embedded_cov`, `spm_hmm_find_flips`, `spm_hmm_build_template`, `spm_hmm_sign_flip`, `spm_eeg_hmm_sign_flip` |
| §11a several complete fits | `spm_hmm` (`S.bestof`) |
| batch interface | `tbx_cfg_hmm` (SPM > Tools > HMM) |
| all data access (memory or disk) | `private/data_reader` |
| simulation | `spm_hmm_simulate` |
| tests | `tests/test_spm_hmm.m` |

## 16. References

- Beal, M. J. (2003). *Variational Algorithms for Approximate Bayesian Inference*. PhD thesis, UCL, ch. 3 (VB for HMMs).
- MacKay, D. J. C. (1997). Ensemble learning for hidden Markov models. Technical report, Cambridge.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, §10.2 (VB Gaussian mixtures, Normal–Wishart).
- Hoffman, M. D., Blei, D. M., Wang, C., Paisley, J. (2013). Stochastic variational inference. *JMLR* 14.
- Foti, N., Xu, J., Laird, D., Fox, E. (2014). Stochastic variational inference for hidden Markov models. *NeurIPS*.
- Johnson, M. J., Willsky, A. S. (2014). Stochastic variational inference for Bayesian time series models. *ICML*.
- Rabiner, L. R. (1989). A tutorial on hidden Markov models. *Proc. IEEE* 77.
- osl-dynamics (HMM inference and summary-statistic conventions).
