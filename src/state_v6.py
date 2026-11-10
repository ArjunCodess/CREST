"""Experimental state features and local conservation, never official LWR scores.

All error scales are fitted on training cells. Missing ramp or connector flux
disables the associated feature/equation rather than supplying a zero flux.
"""
import gc
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, diags
from .models import booster
from .refinement import interpolation
from .state_v4 import CrossSensorFactor, context_rows


def ramp_features(values, params, flux, variance=None):
    incoming, outgoing, net, known = flux
    q = values[..., 1]
    observed = np.isfinite(q)
    upstream = np.nan_to_num(q) @ incoming.T + net
    downstream = np.nan_to_num(q) @ outgoing.T - net @ outgoing.T
    up_valid = known & (observed.astype(int) @ (incoming != 0).T == (incoming != 0).sum(1))
    dn_valid = known & (observed.astype(int) @ (outgoing != 0).T == (outgoing != 0).sum(1))
    dn_valid &= known.astype(int) @ (outgoing != 0).T == (outgoing != 0).sum(1)
    estimates = np.stack([np.where(up_valid, upstream, np.nan),
                          np.where(dn_valid, downstream, np.nan)], -1)
    # Same-time station balance omits accumulation. Its empirical variance must
    # absorb that approximation, and no official flux identity is claimed.
    if variance is None:
        variance = np.ones((len(params), 2))
    precision = np.where(np.isfinite(estimates), 1 / variance[None], 0)
    total = precision.sum(-1)
    fused = np.divide((np.nan_to_num(estimates)*precision).sum(-1), total,
                      out=np.full(q.shape, np.nan), where=total > 0)
    uncertainty = np.divide(1., total, out=np.full(q.shape, np.nan), where=total > 0)
    cap = params.capacity_vph.to_numpy()
    base = interpolation(values)
    vf, kj = params.free_speed_kmh.to_numpy(), params.k_jam.to_numpy()
    wave = cap / (kj-cap/vf)
    speed = np.maximum(base[..., 0], 1)
    fd_flow = wave*kj*speed/(speed+wave)
    return np.concatenate([estimates/cap[None, :, None],
        (estimates-base[..., 1, None])/cap[None, :, None],
        (fused/cap)[..., None], (uncertainty/cap**2)[..., None],
        np.stack([up_valid, dn_valid], -1).astype(float),
        (fd_flow/cap)[..., None], (speed < .8*vf)[..., None]], -1).astype(np.float32)


def panel_weights(regimes, residual, lanes):
    """Training-only approximation to the derivative of macro panel RMSE.

    Residuals are native km/h and vph. Flow is converted to per-lane units.
    The caller trains normalized per-lane flow, so lane weighting is not applied
    a second time. Each channel is a separate tree objective.
    """
    normalized = residual / np.stack([np.ones_like(lanes), lanes], -1)
    weights = np.empty_like(normalized, dtype=float)
    scales = {}
    for regime in np.unique(regimes):
        take = regimes == regime
        rmse = np.maximum(np.sqrt(np.mean(normalized[take]**2, axis=0)), [.1, 1.])
        weights[take] = 1 / (take.sum()*rmse)
        scales[str(float(regime))] = rmse.tolist()
    weights /= weights.mean(axis=0)
    return weights, scales


class RampAwareReconstructor:
    def __init__(self, rounds=220, max_rows=120000, use_ramps=True, weighted=True, block_steps=72):
        self.rounds, self.max_rows = rounds, max_rows
        self.use_ramps, self.weighted = use_ramps, weighted
        self.block_steps = block_steps

    def fit(self, values, times, up, down, params, flux):
        self.factor = CrossSensorFactor().fit(values, params)
        rng = np.random.default_rng(664)
        dates = np.asarray(times.normalize())
        unique, day = np.unique(dates, return_inverse=True)
        daily = rng.choice([.2, .3, .5], len(unique))
        rates = daily[day]
        hidden = (rng.random(values.shape[:2]) < rates[:, None]) & np.isfinite(values).all(-1)
        masked = values.copy(); masked[hidden] = np.nan
        # Estimate station-balance error on fit data only, including storage error.
        count = np.zeros((len(params), 2), dtype=int)
        squared_error = np.zeros((len(params), 2))
        for start in range(0, len(values), self.block_steps):
            end = min(start+self.block_steps, len(values))
            local_flux = (*flux[:2], flux[2][start:end], flux[3][start:end])
            raw = ramp_features(masked[start:end], params, local_flux)[..., :2]*params.capacity_vph.to_numpy()[None, :, None]
            error = raw-values[start:end, :, 1, None]
            count += np.isfinite(error).sum(0)
            squared_error += np.nansum(error**2, axis=0)
        self.ramp_variance = np.divide(squared_error, count,
            out=np.full(count.shape, 600.**2), where=count >= 20)
        self.ramp_variance = np.maximum(self.ramp_variance, 100.)
        factor = self.factor.reconstruct(masked)
        eligible = np.flatnonzero(hidden.ravel())
        chosen = np.sort(rng.choice(eligible, min(self.max_rows, len(eligible)), replace=False))
        if not len(chosen):
            raise ValueError('No eligible training targets')
        x_train = None
        residual = np.empty((len(chosen), 2))
        n = len(params); lanes = params.lanes.to_numpy()
        for start in range(0, len(values), self.block_steps):
            end = min(start+self.block_steps, len(values))
            left, right = np.searchsorted(chosen, [start*n, end*n])
            selected = chosen[left:right]
            if not len(selected):
                continue
            lo, hi = max(0, start-144), min(len(values), end+144)
            x, base = context_rows(masked[lo:hi], times[lo:hi], up, down, params,
                                  factor[lo:hi], (start-lo, end-lo))
            ti, ni = selected//n-start, selected % n
            local_flux = (*flux[:2], flux[2][start:end], flux[3][start:end])
            extra = ramp_features(masked[start:end], params, local_flux, self.ramp_variance)
            extension = extra if self.use_ramps else extra[..., -2:]
            rows = np.concatenate([x[ti, ni], extension[ti, ni]], -1)
            if x_train is None:
                x_train = np.empty((len(chosen), rows.shape[-1]), dtype=np.float32)
            x_train[left:right] = rows
            residual[left:right] = values.reshape(-1, 2)[selected]-base[ti, ni]
        lane_rows, regime_rows = lanes[chosen % n], rates[chosen//n]
        weights, self.training_rmse = panel_weights(regime_rows, residual, lane_rows)
        target = residual/np.stack([np.full(len(residual), 25.), 600*lane_rows], -1)
        del masked, factor, extra, x
        gc.collect()
        self.models = [booster(seed=664+c, rounds=self.rounds).fit(x_train, target[:, c],
                       sample_weight=weights[:, c] if self.weighted else None) for c in range(2)]
        self.error_scale = np.stack([
            np.full(len(params), max(float(np.std(residual[:, 0])), 1.)),
            600*lanes*max(float(np.std(target[:, 1])), .05)], -1)
        self.training_targets = len(chosen)
        del x_train
        gc.collect()
        return self

    def reconstruct(self, values, times, up, down, params, flux):
        factor = self.factor.reconstruct(values)
        scale = np.stack([np.full(len(params), 25.), 600*params.lanes.to_numpy()], -1)
        output = np.empty_like(values)
        for start in range(0, len(values), self.block_steps):
            end = min(start+self.block_steps, len(values)); lo, hi = max(0, start-144), min(len(values), end+144)
            x, base = context_rows(values[lo:hi], times[lo:hi], up, down, params,
                                  factor[lo:hi], (start-lo, end-lo))
            local_flux = (*flux[:2], flux[2][start:end], flux[3][start:end])
            extra = ramp_features(values[start:end], params, local_flux, self.ramp_variance)
            extension = extra if self.use_ramps else extra[..., -2:]
            x = np.concatenate([x, extension], -1)
            predicted = np.stack([m.booster_.predict(x.reshape(-1, x.shape[-1]), num_threads=4)
                                  .reshape(x.shape[:2]) for m in self.models], -1)*scale+base
            output[start:end] = np.where(np.isfinite(values[start:end]), values[start:end], np.maximum(predicted, 0))
        return output


def mass_diagnostic(states, params, flux):
    incoming, outgoing, net, known = flux
    q = states[..., 1]
    accumulation = q/np.maximum(states[..., 0], 1)*params.length_km.to_numpy()
    rhs = (q@incoming.T+net-q@outgoing.T)/12
    valid = known[:-1] & np.isfinite(states[:-1]).all(-1) & np.isfinite(states[1:]).all(-1)
    valid &= (np.isfinite(q[:-1]).astype(int) @ ((incoming+outgoing) != 0).T
              == ((incoming+outgoing) != 0).sum(1))
    residual = np.diff(accumulation, axis=0)-rhs[:-1]
    valid &= np.isfinite(residual)
    if not valid.any():
        return {'n_transitions': 0, 'proxy_S_LWR': None, 'mass_rmse': None}
    return {'n_transitions': int(valid.sum()),
            'proxy_S_LWR': float(max(0, 1-np.abs(residual[valid]).sum()/max(np.abs(rhs[:-1][valid]).sum(), 1e-9))),
            'mass_rmse': float(np.sqrt(np.mean(residual[valid]**2)))}


def uncertainty_projection(states, observed, params, flux, error_scale, weight=.05, max_nfev=15):
    """One linearization, soft-L1 local conservation, fixed observed cells.

    Bounds keep density's linearization away from zero speed. This is an
    uncertainty-weighted approximate projection, not the hidden official LWR.
    """
    incoming, outgoing, net, known = flux
    base = np.asarray(states, dtype=float).copy()
    fixed = np.isfinite(observed)
    base[fixed] = observed[fixed]
    before = mass_diagnostic(base, params, flux)
    if weight == 0 or before['n_transitions'] == 0 or fixed.all():
        return base, {'optimized': False, 'before': before, 'after': before}
    nt, nn, _ = base.shape
    movable = np.flatnonzero(~fixed.ravel())
    lookup = np.full(base.size, -1, dtype=int); lookup[movable] = np.arange(len(movable))
    sigma = np.broadcast_to(error_scale, base.shape).ravel()[movable]
    speed = np.maximum(base[..., 0], 1); flow = base[..., 1]
    length = params.length_km.to_numpy()
    dn_v = -flow/speed**2*length; dn_q = length/speed
    rhs = (flow@incoming.T+net-flow@outgoing.T)/12
    residual = np.diff(flow/speed*length, axis=0)-rhs[:-1]
    valid = known[:-1] & np.isfinite(residual)
    rr, cc, vv, target = [], [], [], []
    for t, i in np.argwhere(valid):
        row = len(target); target.append(residual[t, i])
        terms = [(t, i, 0, -dn_v[t, i]), (t, i, 1, -dn_q[t, i]),
                 (t+1, i, 0, dn_v[t+1, i]), (t+1, i, 1, dn_q[t+1, i])]
        for j in np.flatnonzero(incoming[i]-outgoing[i]):
            terms.append((t, j, 1, -(incoming[i, j]-outgoing[i, j])/12))
        for tt, j, c, derivative in terms:
            col = lookup[(tt*nn+j)*2+c]
            if col >= 0:
                rr.append(row); cc.append(col); vv.append(derivative*sigma[col])
    jac = coo_matrix((vv, (rr, cc)), shape=(len(target), len(movable))).tocsr()
    target = np.asarray(target); multiplier = np.sqrt(weight)/10
    def objective(z):
        r = (target+jac@z)*multiplier
        # Quadratic prediction penalty and differentiable L1 mass penalty.
        transformed = np.sign(r)*np.sqrt(2*(np.sqrt(1+r*r)-1))
        return np.r_[z, transformed]
    def derivative(z):
        from scipy.sparse import vstack
        r = (target+jac@z)*multiplier
        factor = np.sqrt((np.sqrt(1+r*r)+1)/2)/np.sqrt(1+r*r)
        return vstack([diags(np.ones(len(z))), diags(factor*multiplier)@jac], format='csr')
    lower = (np.tile([1., 0.], nt*nn)[movable]-base.ravel()[movable])/sigma
    upper_native = np.broadcast_to(np.stack([params.free_speed_kmh, params.capacity_vph], -1), base.shape)
    upper = (upper_native.ravel()[movable]-base.ravel()[movable])/sigma
    # Starting predictions can exceed bounds; bring only movable cells inside.
    start = np.clip(np.zeros(len(movable)), lower+1e-9, upper-1e-9)
    solved = least_squares(objective, start, jac=derivative, bounds=(lower, upper), max_nfev=max_nfev)
    base.ravel()[movable] += sigma*solved.x
    return base, {'optimized': True, 'success': bool(solved.success), 'nfev': solved.nfev,
                  'before': before, 'after': mass_diagnostic(base, params, flux),
                  'max_standardized_move': float(np.max(np.abs(solved.x)))}
