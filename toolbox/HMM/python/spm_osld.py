"""osl-dynamics tasks for SPM's HMM toolbox.

A job is a folder holding job.json, {"task": <name>, "args": {...}}, and
the arrays the arguments refer to. An array is stored as a raw binary file
in column-major (MATLAB) order and referred to as
{"ndarray": <file>, "shape": [...], "dtype": <dtype>}. The result is
written to result.json in the same folder, in the same format.

Usage: python spm_osld.py <job folder>, or run(<job folder>) in process.
"""

import gc
import importlib.metadata
import json
import math
import os
import platform
import sys
import warnings

import numpy as np

if sys.platform == "darwin":
    # NumPy with Apple's Accelerate raises floating-point warnings in matmul
    # whose results are finite and correct
    warnings.filterwarnings("ignore", message=r".* encountered in matmul", category=RuntimeWarning)


def run(job_dir):
    """Run the job in job_dir; return the path of its result.json."""
    job_dir = str(job_dir)
    with open(os.path.join(job_dir, "job.json")) as f:
        job = json.load(f)
    args = _decode(job.get("args") or {}, job_dir)
    args = {k: v for k, v in args.items() if v is not None}
    out = TASKS[job["task"]](store_dir=os.path.join(job_dir, "store"), **args)
    gc.collect()  # release memory-mapped files in the job folder
    result = _Encoder(job_dir).encode(out)
    path = os.path.join(job_dir, "result.json")
    with open(path, "w") as f:
        json.dump(result, f)
    return path


# ----------------------------------------------------------------------------
# Tasks


def versions(store_dir, check=False):
    """Versions of Python and the packages; with check, import them."""
    out = {"python": platform.python_version(), "executable": sys.executable}
    for name in ["osl-dynamics", "tensorflow", "tensorflow-probability", "tf-keras", "numpy"]:
        try:
            out[name.replace("-", "_")] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name.replace("-", "_")] = ""
    if check:
        import tensorflow  # noqa: F401
        from osl_dynamics.models import hmm  # noqa: F401
    return out


def fit_hmm(
    store_dir,
    data,
    sampling_frequency,
    n_states,
    learn_means,
    config=None,
    standardize=True,
    n_pca_components=None,
    n_embeddings=None,
    n_tde_pca_components=None,
    seed=None,
    model_dir=None,
):
    """Prepare the data, train an HMM and infer the state time courses.

    data is a list of [samples x channels] arrays, one per session. The
    state probabilities and Viterbi paths are returned for every sample of
    data: samples lost at the ends of each session by time-delay embedding
    are NaN (probabilities) or 0 (paths).
    """
    from osl_dynamics.inference import modes
    from osl_dynamics.models import hmm
    from osl_dynamics.utils.misc import set_random_seed

    if seed is not None:
        set_random_seed(int(seed))
    data = _sessions(data)
    n_in = [x.shape[0] for x in data]
    d = _data(data, sampling_frequency, store_dir)

    # Preparation
    out = {}
    if standardize:
        d.standardize()
    if n_pca_components:
        d.pca(n_pca_components=int(n_pca_components))
        out["pca_components"] = np.asarray(d.pca_components)
        if standardize and not n_embeddings:
            d.standardize()
    offset = 0
    if n_embeddings:
        d.tde_pca(
            n_embeddings=int(n_embeddings),
            n_pca_components=int(n_tde_pca_components) if n_tde_pca_components else None,
        )
        out["tde_pca_components"] = np.asarray(d.pca_components)
        offset = int(n_embeddings) // 2
        if standardize:
            d.standardize()

    # Model
    cfg = {
        "sequence_length": 200,
        "batch_size": 256,
        "learning_rate": 0.01,
        "n_epochs": 20,
        "n_init": 5,
        "n_init_epochs": 2,
    }
    cfg.update(config or {})
    cfg.update(
        n_states=int(n_states),
        n_channels=d.n_channels,
        learn_means=bool(learn_means),
        learn_covariances=True,
    )
    for k in ["sequence_length", "batch_size", "n_epochs", "n_init", "n_init_epochs", "best_of"]:
        if k in cfg:
            cfg[k] = int(cfg[k])
    model = hmm.Model(hmm.Config(**cfg))
    model.train(d)
    if model_dir:
        model.save(model_dir)

    # State time courses on every prepared sample
    prepared = _sessions(d.time_series())
    alpha, path = _infer(model, prepared, sampling_frequency, store_dir)

    # Summary statistics of the Viterbi paths
    stc = [np.eye(cfg["n_states"], dtype=int)[p] for p in path]
    G, K = len(stc), cfg["n_states"]
    out["stats"] = {
        "fo": np.reshape(modes.fractional_occupancies(stc), (G, K)),
        "mean_lifetime": np.reshape(modes.mean_lifetimes(stc, sampling_frequency), (G, K)),
        "mean_interval": np.reshape(modes.mean_intervals(stc, sampling_frequency), (G, K)),
        "switching_rate": np.reshape(modes.switching_rates(stc, sampling_frequency), (G, K)),
    }

    # Back to the samples of data
    out["alpha"], out["path"] = [], []
    for a, p, n in zip(alpha, path, n_in):
        A = np.full((n, K), np.nan)
        A[offset : offset + a.shape[0]] = a
        P = np.zeros(n)
        P[offset : offset + p.shape[0]] = p + 1
        out["alpha"].append(A)
        out["path"].append(P)

    means, covs = model.get_means_covariances()
    out.update(
        free_energy=float(model.free_energy(d)),
        initial_state_probs=np.asarray(model.get_initial_state_probs()),
        trans_prob=np.asarray(model.get_trans_prob()),
        means=np.asarray(means).T,
        covs=np.moveaxis(np.asarray(covs), 0, -1),
        offset=offset,
        config={k: v for k, v in cfg.items() if isinstance(v, (bool, int, float, str))},
    )
    return out


def sign_flip(
    store_dir,
    data,
    n_embeddings=15,
    standardize=True,
    template_cov=None,
    n_random_starts=3,
    two_stage=False,
):
    """Signs that align the channels of each session (Data.align_channel_signs)."""
    data = _sessions(data)
    d = _data(data, None, store_dir)
    d.align_channel_signs(
        template_cov=template_cov,
        n_random_starts=int(n_random_starts),
        n_embeddings=int(n_embeddings),
        standardize=bool(standardize),
        two_stage=bool(two_stage),
    )
    flips = []
    for x, y in zip(data, _sessions(d.time_series())):
        s = np.sign(np.sum(np.asarray(x, np.float64) * np.asarray(y, np.float64), axis=0))
        s[s == 0] = 1
        flips.append(s)
    return {"flips": np.array(flips)}


def envelope(store_dir, data, sampling_frequency, low_freq=None, high_freq=None, order=5, n_window=1):
    """Band-pass filter, amplitude envelope and moving average of each array.

    Each output array is shorter than its input by n_window - 1 samples,
    (n_window - 1)/2 at each end.
    """
    d = _data(_sessions(data), sampling_frequency, store_dir)
    if low_freq is not None or high_freq is not None:
        d.filter(low_freq=low_freq, high_freq=high_freq, order=int(order))
    d.amplitude_envelope()
    if int(n_window) > 1:
        d.moving_average(int(n_window))
    return {"envelope": _sessions(d.time_series()), "offset": (int(n_window) - 1) // 2}


def spectra(
    store_dir,
    data,
    sampling_frequency,
    alpha=None,
    window_length=None,
    step_size=None,
    time_half_bandwidth=4,
    n_tapers=7,
    frequency_range=None,
    standardize=True,
    calc_coh=True,
    calc_cpsd=False,
    n_jobs=1,
):
    """State multitaper spectra (osl_dynamics.analysis.spectral.multitaper_spectra)."""
    from osl_dynamics.analysis import spectral

    data = _sessions(data)
    if alpha is not None:
        alpha = _sessions(alpha)
    res = spectral.multitaper_spectra(
        data=data,
        sampling_frequency=sampling_frequency,
        alpha=alpha,
        window_length=None if window_length is None else int(window_length),
        step_size=None if step_size is None else int(step_size),
        time_half_bandwidth=time_half_bandwidth,
        n_tapers=int(n_tapers),
        frequency_range=None if frequency_range is None else list(np.ravel(frequency_range)),
        standardize=bool(standardize),
        calc_cpsd=bool(calc_cpsd),
        calc_coh=bool(calc_coh),
        keepdims=True,
        n_jobs=int(n_jobs),
    )
    out = {"f": np.asarray(res[0]), ("cpsd" if calc_cpsd else "psd"): np.asarray(res[1])}
    if calc_coh:
        out["coh"] = np.asarray(res[2])
    return out


def spectral_components(store_dir, coh, n_components):
    """Spectral components of coherence (spectral.decompose_spectra)."""
    from osl_dynamics.analysis import spectral

    comp = spectral.decompose_spectra(np.asarray(coh), int(n_components))
    return {"components": np.reshape(comp, (int(n_components), -1))}


def power_maps(store_dir, f, psd, components=None, frequency_range=None, method="mean"):
    """Power of each session, component, state and channel (power.variance_from_spectra)."""
    from osl_dynamics.analysis import power

    G, K, D = psd.shape[:3]
    C = 1 if components is None else np.atleast_2d(components).shape[0]
    p = power.variance_from_spectra(
        np.ravel(f),
        psd,
        components=None if components is None else np.atleast_2d(components),
        frequency_range=None if frequency_range is None else list(np.ravel(frequency_range)),
        method=method,
    )
    return {"power": np.reshape(p, (G, C, K, D))}


def coherence_networks(store_dir, f, coh, components=None, frequency_range=None):
    """Mean coherence of each session, component, state and channel pair."""
    from osl_dynamics.analysis import connectivity

    G, K, D = coh.shape[:3]
    C = 1 if components is None else np.atleast_2d(components).shape[0]
    c = connectivity.mean_coherence_from_spectra(
        np.ravel(f),
        coh,
        components=None if components is None else np.atleast_2d(components),
        frequency_range=None if frequency_range is None else list(np.ravel(frequency_range)),
    )
    return {"coh": np.reshape(c, (G, C, K, D, D))}


def threshold(store_dir, conn, percentile, subtract_mean=False, mean_weights=None, absolute_value=False):
    """Edges above a percentile of each [components x states] network (connectivity.threshold)."""
    from osl_dynamics.analysis import connectivity

    conn = np.asarray(conn)
    percentile = np.asarray(percentile, dtype=float)
    if percentile.size == 1:
        percentile = float(percentile)
    edges = connectivity.threshold(
        conn,
        percentile,
        subtract_mean=bool(subtract_mean),
        mean_weights=None if mean_weights is None else np.ravel(mean_weights),
        absolute_value=bool(absolute_value),
        return_edges=True,
    )
    edges = np.reshape(edges, conn.shape)
    return {"edges": edges.astype(float)}


def orthogonalise(store_dir, data):
    """Symmetric orthogonalisation weights of [samples x channels] data.

    meeg.parcellation.symmetric_orthogonalization with maintain_magnitudes:
    the orthogonalised data are data @ weights.
    """
    from osl_dynamics.meeg.parcellation import symmetric_orthogonalization

    _, weights = symmetric_orthogonalization(
        np.asarray(data, np.float64).T, maintain_magnitudes=True, compute_weights=True
    )
    return {"weights": np.asarray(weights)}


TASKS = {
    "versions": versions,
    "fit_hmm": fit_hmm,
    "sign_flip": sign_flip,
    "envelope": envelope,
    "spectra": spectra,
    "spectral_components": spectral_components,
    "power_maps": power_maps,
    "coherence_networks": coherence_networks,
    "threshold": threshold,
    "orthogonalise": orthogonalise,
}


# ----------------------------------------------------------------------------
# Helpers


def _sessions(x):
    """A list of 2D arrays, one per session."""
    if isinstance(x, np.ndarray):
        x = [x]
    return [np.asarray(a).reshape(a.shape[0], -1) if np.ndim(a) != 2 else np.asarray(a) for a in x]


def _data(arrays, sampling_frequency, store_dir):
    """An osl-dynamics Data object holding the arrays (float32)."""
    from osl_dynamics.data import Data

    os.makedirs(store_dir, exist_ok=True)
    return Data(
        [np.ascontiguousarray(a, dtype=np.float32) for a in arrays],
        sampling_frequency=sampling_frequency,
        store_dir=store_dir,
        n_jobs=1,
    )


def _infer(model, prepared, sampling_frequency, store_dir):
    """State probabilities and Viterbi paths (0-based) of every prepared sample.

    osl-dynamics infers the states of whole sequences only; the samples
    after the last whole sequence of a session take their states from the
    session's final sequence_length samples.
    """
    L = model.config.sequence_length
    for x in prepared:
        if x.shape[0] < L:
            raise ValueError(
                f"A session has {x.shape[0]} samples after preparation, "
                f"fewer than sequence_length ({L})."
            )
    d = _data(prepared, sampling_frequency, store_dir)
    alpha = _sessions(model.get_alpha(d))
    path = [np.argmax(v, axis=1) for v in _sessions(model.get_viterbi_path(d))]
    tails = [i for i, x in enumerate(prepared) if x.shape[0] % L]
    if tails:
        dt = _data([prepared[i][-L:] for i in tails], sampling_frequency, store_dir)
        at = _sessions(model.get_alpha(dt))
        vt = [np.argmax(v, axis=1) for v in _sessions(model.get_viterbi_path(dt))]
        for j, i in enumerate(tails):
            r = prepared[i].shape[0] % L
            alpha[i] = np.concatenate([alpha[i], at[j][-r:]])
            path[i] = np.concatenate([path[i], vt[j][-r:]])
    return alpha, path


def _decode(x, job_dir):
    if isinstance(x, dict):
        if "ndarray" in x:
            a = np.fromfile(os.path.join(job_dir, x["ndarray"]), dtype=x.get("dtype", "float64"))
            return a.reshape([int(n) for n in x["shape"]], order="F")
        return {k: _decode(v, job_dir) for k, v in x.items()}
    if isinstance(x, list):
        return [_decode(v, job_dir) for v in x]
    return x


class _Encoder:
    def __init__(self, job_dir):
        self.job_dir = job_dir
        self.n = 0

    def encode(self, x):
        if isinstance(x, dict):
            return {k: self.encode(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [self.encode(v) for v in x]
        if isinstance(x, np.ndarray) and x.ndim > 0:
            self.n += 1
            name = f"out{self.n}.bin"
            a = np.asarray(x, dtype=np.float64)
            a.ravel(order="F").tofile(os.path.join(self.job_dir, name))
            return {"ndarray": name, "shape": list(a.shape), "dtype": "float64"}
        if isinstance(x, np.generic):
            x = x.item()
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x


if __name__ == "__main__":
    try:
        run(sys.argv[1])
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
