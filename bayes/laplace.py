"""MAP + Laplace posterior, as a drop-in for a NUTS run.

Why this is available at all is a property of the model, not a shortcut. The
production model already marginalises the latent abilities analytically with the
Kalman filter, so what is left is a smooth, closed-form marginal posterior over
~56 hyperparameters. NUTS spends 10^6 gradient evaluations exploring that;
L-BFGS finds its mode in ~170 and the Hessian there describes its shape.

Measured on the 2023 fold of the production model: MAP 2.0s, Hessian 8.6s, 1000
constrained draws 0.6s - against tens of minutes for 4 sequential NUTS chains.
That is the difference between a backtest you run when you have a hypothesis and
one you run overnight.

**What this approximation asserts.** That the posterior is Gaussian on the
unconstrained scale. Two consequences worth holding onto:

* Scale parameters are skewed on that scale even when they are well behaved, so
  intervals come out slightly narrow. This shows up in the coverage report and
  nowhere else, which is why the coverage report is the thing to read after
  switching.
* A hierarchical scale whose posterior piles up against zero has no interior
  mode, and a Gaussian at the mode is then not an approximation of anything.
  That failure is *detectable*, which is the whole reason this module reports
  the Hessian's eigenvalues rather than quietly inverting it: a non-positive
  direction means the assumption failed, not that the answer is a bit off.

The importance diagnostic is the second line of defence. Draws come from the
fitted Gaussian q and the exact log posterior is known, so the self-normalised
weights w = p/q say how well q covers p. A weight ESS near 1.0 means the
Gaussian is doing the job; a small one means a few draws carry all the mass and
the approximation is missing where the posterior actually lives. It costs one
extra likelihood evaluation per draw and it turns "approximate and hope" into
"approximate and check".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from numpyro.infer.util import initialize_model
from scipy.optimize import minimize

# Eigenvalues below this are treated as an unidentified direction rather than a
# curvature to invert. The clip keeps the covariance finite so the caller gets a
# posterior and a loud diagnostic instead of a NaN and a traceback.
MIN_EIGENVALUE = 1e-6


@dataclass
class LaplaceFit:
    """Posterior draws with the interface the rest of the code expects.

    ``get_samples`` mirrors ``MCMC.get_samples`` so everything downstream -
    ``filtered_states``, ``predict_games``, the parameter summary in the
    backtest - consumes this without knowing which inference produced it.
    """

    samples: dict
    mode: dict
    diagnostics: dict = field(default_factory=dict)

    def get_samples(self, group_by_chain: bool = False) -> dict:
        if group_by_chain:
            return {k: v[None, ...] for k, v in self.samples.items()}
        return self.samples

    def print_summary(self, *args, **kwargs) -> None:
        d = self.diagnostics

        def num(key, fmt=".2f"):
            # `importance_check=False` leaves the weight keys absent, and
            # formatting None with :.2f raises - so a diagnostic printout used
            # to crash the run that had asked for less diagnosis.
            v = d.get(key)
            return format(v, fmt) if isinstance(v, (int, float)) else "n/a"

        print(
            f"Laplace: dim={d.get('dim')} logpost={num('logpost', '.1f')} "
            f"grad_evals={d.get('grad_evals')} "
            f"newton_step={num('newton_step', '.3f')} "
            f"eig_min={num('eig_min', '.2e')} nonpos={d.get('n_nonpositive')} "
            f"psis_khat={num('psis_khat', '.2f')} "
            f"weight_ess={num('weight_ess_count', '.1f')} "
            f"max_w={num('max_weight', '.3f')}"
        )


def fit_laplace(
    model,
    *model_args,
    num_samples: int = 1000,
    seed: int = 0,
    n_restarts: int = 3,
    importance_check: bool = True,
    **model_kwargs,
) -> LaplaceFit:
    """Fit ``model`` by MAP and return draws from the Laplace approximation.

    ``n_restarts`` optimisations run from different random initialisations and
    the best mode wins. Each costs a couple of seconds, and the alternative -
    trusting a single L-BFGS run from a uniform init on a hierarchical posterior
    - is exactly the kind of silent failure this whole module has to avoid.
    """
    rng = jax.random.PRNGKey(seed)
    init_key, draw_key = jax.random.split(rng)

    mi = initialize_model(init_key, model, model_args=model_args,
                          model_kwargs=model_kwargs)
    flat0, unflatten = ravel_pytree(mi.param_info.z)
    dim = int(flat0.size)

    pot = jax.jit(lambda x: mi.potential_fn(unflatten(x)))
    val_grad = jax.jit(jax.value_and_grad(lambda x: mi.potential_fn(unflatten(x))))
    pot(flat0).block_until_ready()

    n_eval = [0]

    def objective(x):
        n_eval[0] += 1
        v, g = val_grad(jnp.asarray(x, flat0.dtype))
        return np.asarray(v, np.float64), np.asarray(g, np.float64)

    best = None
    for r in range(max(1, n_restarts)):
        # Restart r = 0 uses numpyro's own initialisation; later ones jitter it,
        # on the unconstrained scale where a unit-ish perturbation is meaningful
        # for every parameter regardless of its support.
        key = jax.random.fold_in(init_key, r)
        start = np.asarray(flat0, np.float64)
        if r:
            start = start + 0.5 * np.asarray(
                jax.random.normal(key, (dim,)), np.float64
            )
        res = minimize(objective, start, jac=True, method="L-BFGS-B",
                       options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
        # A restart that diverges returns fun=nan, and `nan < anything` is False
        # in both directions - so a NaN result taken as `best` can never be
        # displaced and the function returns a mode made of NaNs, silently.
        # Observed live: availability restart 1 came back nan / success=False /
        # nit=0. Non-finite and unsuccessful results are discarded outright.
        if not (np.isfinite(res.fun) and np.all(np.isfinite(res.x))):
            continue
        if best is None or res.fun < best.fun:
            best = res

    if best is None:
        raise RuntimeError(
            f"every one of {max(1, n_restarts)} L-BFGS restarts returned a "
            "non-finite objective; there is no mode to expand around."
        )

    mode_flat = jnp.asarray(best.x, flat0.dtype)

    # The potential is the negative log posterior, so its Hessian is the
    # precision matrix directly.
    hess = np.asarray(jax.hessian(pot)(mode_flat), np.float64)
    hess = 0.5 * (hess + hess.T)
    eig, vecs = np.linalg.eigh(hess)
    n_nonpos = int((eig <= MIN_EIGENVALUE).sum())
    # A negative eigenvalue is a saddle, not a flat direction. Clipping it up to
    # +MIN_EIGENVALUE turns "this is not a mode" into "this is the direction of
    # maximal posterior variance", which is the most confident possible way to
    # be wrong. Refuse instead.
    if eig.min() < 0:
        raise RuntimeError(
            f"Hessian has {int((eig < 0).sum())} negative eigenvalue(s) "
            f"(min {eig.min():.3e}): L-BFGS stopped at a saddle, not a mode. "
            "A Gaussian here approximates nothing."
        )
    cov = (vecs / np.clip(eig, MIN_EIGENVALUE, None)) @ vecs.T

    # Convergence, measured in the metric the posterior actually uses. The raw
    # gradient norm is dominated by directions with curvature in the thousands
    # and says "not converged" at a mode that is fine; the Newton step
    # sqrt(g' H^-1 g) is the distance to the mode in posterior sds, and the
    # implied improvement in log-density is half its square.
    g = np.asarray(val_grad(mode_flat)[1], np.float64)
    newton_step = float(np.sqrt(max(g @ (cov @ g), 0.0)))

    chol = np.linalg.cholesky(cov + 1e-10 * np.eye(dim))
    rs = np.random.default_rng(seed)
    eps = rs.standard_normal((num_samples, dim))
    draws_flat = np.asarray(best.x)[None, :] + eps @ chol.T

    diagnostics = {
        "dim": dim,
        "logpost": float(-best.fun),
        "grad_evals": int(n_eval[0]),
        "converged": bool(best.success),
        "newton_step": newton_step,
        "eig_min": float(eig.min()),
        "eig_max": float(eig.max()),
        "n_nonpositive": n_nonpos,
        "n_draws": int(num_samples),
    }

    if importance_check:
        diagnostics.update(_importance_diagnostics(pot, draws_flat, eps, flat0.dtype))

    samples = jax.vmap(mi.postprocess_fn)(
        jax.vmap(unflatten)(jnp.asarray(draws_flat, flat0.dtype))
    )
    samples = {k: np.asarray(v) for k, v in samples.items()}
    mode = {k: np.asarray(v) for k, v in mi.postprocess_fn(unflatten(mode_flat)).items()}
    return LaplaceFit(samples=samples, mode=mode, diagnostics=diagnostics)


def _importance_diagnostics(pot, draws_flat, eps, dtype) -> dict:
    """How well the fitted Gaussian covers the true posterior.

    Both densities are on the unconstrained scale - numpyro's potential carries
    the log-Jacobian - so the ratio is a like-for-like comparison. Only the
    *shape* of log q matters here, so the normalising constant is dropped and
    the quadratic form is read straight off the standard normal draws that
    generated the sample.

    Returns the weight ESS as a fraction of the draw count (1.0 = the Gaussian
    is indistinguishable from the posterior over the region that matters) and
    the largest single normalised weight (a value near 1 means one draw is
    carrying the estimate).
    """
    log_p = -np.asarray(
        jax.vmap(pot)(jnp.asarray(draws_flat, dtype)), np.float64
    )
    log_q = -0.5 * np.sum(np.asarray(eps, np.float64) ** 2, axis=1)
    log_w = log_p - log_q

    # A Gaussian in ~100 dimensions will occasionally place a draw somewhere the
    # likelihood cannot be evaluated - the availability model's beta-binomial
    # underflows its concentration for an extreme draw and returns NaN. Those
    # draws carry no posterior mass and belong at weight zero, but they used to
    # poison `log_w.max()` and with it every weight, so six bad draws in four
    # thousand turned the entire diagnostic into NaN. Dropping them and counting
    # them is the difference between a diagnostic that reports "the Gaussian
    # misses badly here" and one that reports nothing at all.
    finite = np.isfinite(log_w)
    n_bad = int((~finite).sum())
    if not finite.any():
        return {"weight_ess": float("nan"), "max_weight": float("nan"),
                "n_nonfinite_draws": n_bad}
    lw = log_w[finite] - log_w[finite].max()
    w = np.exp(lw)
    w_sum = w.sum()
    if not np.isfinite(w_sum) or w_sum <= 0:
        return {"weight_ess": float("nan"), "max_weight": float("nan"),
                "n_nonfinite_draws": n_bad}
    w /= w_sum

    # PSIS k-hat is the diagnostic with a threshold behind it. The tail index of
    # the importance weights says whether the ratio p/q has finite variance at
    # all: k <= 0.5 is fine, 0.5-0.7 is usable with reweighting, and above 0.7
    # the estimator has no useful error bound and importance correction cannot
    # rescue it - the answer is a different q, not a bigger sample. Unlike the
    # Hessian's eigenvalues this fires on the failure that actually occurs here:
    # a positive-definite mode whose Gaussian still misses where p lives.
    khat = float("nan")
    try:
        from arviz.stats import psislw

        _, khat = psislw(lw[None, :].copy())
        khat = float(np.asarray(khat).ravel()[0])
    except Exception:                                          # pragma: no cover
        pass

    return {
        "weight_ess": float(1.0 / np.sum(w ** 2) / len(w)),
        "weight_ess_count": float(1.0 / np.sum(w ** 2)),
        "max_weight": float(w.max()),
        "n_nonfinite_draws": n_bad,
        "psis_khat": khat,
    }
