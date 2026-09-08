#!/usr/bin/env python3
"""What a feedback loop on the vibration frequency would buy, in beam time.

Today a Ramsey scan is a list of transmissions. The absorber of region V removes
the neutrons that climb too high, a counter adds up the rest, and each vibration
frequency yields one number. Thirty frequencies, about 141 s each to reach one
percent of statistics, and the shape of the resonance is known when the shift is
over.

Replace that absorber by the pixel sensor of this thesis and each frequency
returns a vertical profile instead of a number. The profile is a mixture of two
known densities -- the Airy functions of n=1 and n=2 -- so the transferred
fraction f can be fitted from it, with its error bar, at any moment during the
acquisition. The pipeline classifies a frame in 169 ms against the 555 ms
between frames, so that fit can be repeated on every frame while the beam is
still running.

Once f and its error are known live, three decisions become possible that are
impossible with a transmission:

  1. stop a frequency point as soon as its error bar reaches what the
     measurement needs, instead of dwelling for a time fixed in advance;
  2. choose the next frequency from what the previous ones showed -- a coarse
     survey first, then a fine grid only where contrast appeared;
  3. abandon a working point that produces no contrast at all, after ten
     seconds rather than after six minutes.

This file simulates all three against a fixed-step scan that uses the same
detector and the same estimator, so that what it measures is the gain of the
scheduling alone and not the gain of the sensor over the absorber. The summary
separates them into four rungs and gives each its own factor.

Every rung aims at the same error on f, one percent, which is the precision the
laboratory already works to. That matters twice: it stops the ladder measuring
a choice of precision as though it were a property of a detector, and it makes
the first rung the scan as it is actually run today. Over three hundred scans of
a thirty-percent transfer:

    absorber, fixed dwell     4219 s          the scan as it is run today
    sensor,   fixed dwell     3098 s    x1.4  dwell from the moment formula
    sensor,   live stop       1270 s    x2.4  each point stopped on its error
    sensor,   feedback loop    459 s    x2.8  survey, decide, zoom

x9.2 end to end -- an hour and a quarter down to eight minutes -- with the loop
placing the fringe at least as well as the fixed grid does, 0.39 Hz against
0.42 Hz. Only x1.4 of that is the profile against the rate. The live stop is a
sensor property even so, and for a reason worth stating: 1/sqrt(N) does not
depend on the working point, so a counter learns nothing by watching itself
finish, while a profile fit does.

Beam time converts into precision at fixed shift length -- more scans, averaged
-- so x9.2 in time is x3.0 on the placement of the fringe. That conversion holds
only if the scans agree on which fringe they are centring, and neither strategy
settles that: 110 of 300 for the loop against 107 for the fixed grid.

Run at a coarser target the loop is worth much less, and the earlier version of
this file said so at 0.03 without noticing that the target was doing the work.
The survey costs what the abandon contract demands and not what the fine grid
demands, so its price is fixed while everything else scales as one over sigma
squared: the finer the measurement, the smaller the survey looks beside it.

And what it finds against itself, because both belong in the same file:

  * chasing a ten-percent transfer instead of thirty, the loop saves nothing.
    The survey scales as one over the transfer squared, and at that level it
    costs as much as the zoom saves: 98 s against the fixed grid's 97 s. The
    loop is worth running for strong effects and for dead working points, not
    for weak ones.
  * neither strategy identifies which Ramsey fringe is the central one, and
    that is physics rather than scheduling: at this geometry the first side
    fringe transfers 0.279 against 0.300, a gap smaller than the error bars.
    The fringe order comes from knowing nu12 beforehand.

Nothing here is a measurement. The lineshape is the textbook separated-
oscillatory-fields result at the geometry of Section 2.4, the densities are Airy
functions, and the counting is Poisson at the rate this detector was measured
to sustain. It is a demonstration meant to be read, cited, and taken further by
whoever inherits the apparatus.

Usage
    python ramsey_feedback.py                    # one scan, full report
    python ramsey_feedback.py --repeats 300      # the gain over many scans
    python ramsey_feedback.py --f-max 0          # the dead-working-point case
    python ramsey_feedback.py --f-max .1 --min-transfer .1   # where it fails

Python 3, numpy and scipy. No other dependency, no data file, no camera.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
from scipy.special import airy, ai_zeros

# ---------------------------------------------------------------------------
# Constants. Everything below is quoted from the thesis; nothing is tuned here.
# ---------------------------------------------------------------------------
HBAR = 1.054571817e-34
M_N = 1.67492750056e-27
G = 9.80665

PITCH = 2.4            # um, the sensor pixel pitch of Section 2.4
DISPLACEMENT = 2.0     # um, the flight of the capture daughter in the silicon
Z_MAX = 50.0           # um, the height range the sensor rows cover

RATE = 71.1            # neutrons per second, measured in Section 4
FRAME_RATE = 1.80      # frames per second, the rate the camera sustained
PER_FRAME = RATE / FRAME_RATE          # ~39.5 neutrons in one frame

NU12 = 254.6           # Hz, the 1 -> 2 transition frequency of Section 2.3
V_X = 8.0              # m/s, horizontal velocity
L_PULSE = 0.150        # m, one oscillating region  (II and IV)
L_FREE = 0.340         # m, the free flight between them (III)

# The transmission measurement this would replace. The per-point time is
# derived, not tabulated: 1/sigma^2 counts at RATE. It used to be written here
# as 141.0 s, which is 140.65 rounded up, and multiplying a rounded value by
# thirty put 4230 s in the thesis beside the 4219 s the ladder computes. One
# number, derived once.
TRANSMISSION_POINTS = 30
TRANSMISSION_SIGMA = 0.01          # the one percent per point of the outlook
TRANSMISSION_S_PER_POINT = 1.0 / TRANSMISSION_SIGMA ** 2 / RATE

Z0 = (HBAR ** 2 / (2 * M_N ** 2 * G)) ** (1 / 3) * 1e6      # um
ZEROS = ai_zeros(6)[0]


# ---------------------------------------------------------------------------
# The detector: what one frequency point actually records.
# density(), blur() and the fit are those of make_absorber_cmos.py, which draws
# Figure "absorber_cmos". They are repeated here rather than imported because
# that file lives in the thesis tree and this one in the code tree; the summary
# printed at the end recomputes the three numbers the thesis quotes from them
# (9.16 um, 16.00 um, 5.9 um) so that a drift between the two copies shows up
# on the first run rather than in the defence.
# ---------------------------------------------------------------------------
def density(z: np.ndarray, n: int) -> np.ndarray:
    """|psi_n(z)|^2, normalised to unit integral. n counts from one."""
    v = airy(z / Z0 + ZEROS[n - 1])[0] ** 2
    return v / np.trapezoid(v, z)


def blur(z: np.ndarray, p: np.ndarray, sigma: float) -> np.ndarray:
    """Convolve with a Gaussian of width sigma, on the grid z."""
    dz = z[1] - z[0]
    half = int(np.ceil(5 * sigma / dz))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) * dz / sigma) ** 2)
    return np.convolve(p, k / k.sum(), mode="same")


def moments(z: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Mean height and RMS width of a density, both in micrometres."""
    m = float(np.trapezoid(z * p, z))
    return m, float(np.sqrt(np.trapezoid((z - m) ** 2 * p, z)))


def sensor_templates() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """The two states as the sensor rows would record them.

    Returns the row centres and the two normalised templates b1, b2, plus the
    handful of scalars the report quotes. The densities are computed on a fine
    grid, blurred by the daughter's flight, and only then summed into rows of
    one pixel pitch -- in that order, because the blur happens in the silicon,
    before anything is binned.
    """
    z = np.linspace(0, 60, 40000)
    p1 = blur(z, density(z, 1), DISPLACEMENT)
    p2 = blur(z, density(z, 2), DISPLACEMENT)

    edges = np.arange(0, Z_MAX + PITCH, PITCH)
    centres = 0.5 * (edges[1:] + edges[:-1])
    b1, _ = np.histogram(z, bins=edges, weights=p1)
    b2, _ = np.histogram(z, bins=edges, weights=p2)
    b1, b2 = b1 / b1.sum(), b2 / b2.sum()

    m1, s1 = moments(z, p1)
    m2, s2 = moments(z, p2)
    scales = {"mean1": m1, "mean2": m2, "shift": m2 - m1,
              "width": 0.5 * (s1 + s2), "rows": len(centres)}
    return centres, b1, b2, scales


class ProfileFit:
    """fit_fraction() of make_absorber_cmos.py, with two changes.

    Same estimator: the two densities are known, the recorded profile is a
    mixture of them, the only unknown is the weight. The weight is scanned on
    the same 4001-point grid, the Poisson log-likelihood is evaluated, and the
    half-unit drop on either side of the maximum gives the standard error.

    The two changes are there because this file calls the estimator tens of
    thousands of times instead of once, and because it calls it at f = 0 where
    the figure never does:

    * the log of the model is tabulated once, in the constructor. b1 and b2 do
      not change during a scan, so a fit is then a single matrix-vector
      product and the loop below can afford to refit on every frame.

    * when the maximum lands on an end of the grid -- which is what happens off
      resonance, where the true fraction is zero -- the interval is cut in half
      by the boundary and the original expression, half the width of the
      interval, returns half the error. The boundary case is handled by taking
      the surviving side as the standard error rather than half the span. The
      figure is unaffected: its panel sits at f = 0.30, far from either end.
    """

    def __init__(self, b1: np.ndarray, b2: np.ndarray, n_grid: int = 4001):
        self.grid = np.linspace(0.0, 1.0, n_grid)
        model = np.outer(1.0 - self.grid, b1) + np.outer(self.grid, b2)
        self.log_model = np.log(np.clip(model, 1e-12, None))

    def __call__(self, counts: np.ndarray) -> tuple[float, float]:
        ll = self.log_model @ counts
        best = float(self.grid[int(np.argmax(ll))])
        inside = self.grid[ll >= ll.max() - 0.5]
        lo, hi = float(inside[0]), float(inside[-1])
        if lo <= self.grid[0] or hi >= self.grid[-1]:
            return best, max(hi - best, best - lo)   # one flank only
        return best, 0.5 * (hi - lo)


# ---------------------------------------------------------------------------
# The physics being scanned: how much population a given frequency transfers.
# ---------------------------------------------------------------------------
def ramsey_transfer(nu: np.ndarray | float) -> np.ndarray:
    """Ramsey transition probability against absolute frequency, peak = 1.

    Two pi/2 regions of duration tau separated by a free flight T, the standard
    separated-oscillatory-fields result, written exactly as in
    make_ramsey_analogy.py so the two figures cannot disagree about the
    lineshape. Only the free-flight phase carries the fringes; at zero detuning
    a pi/2 pulse pair gives probability one, which is why no normalisation
    constant appears.
    """
    tau = L_PULSE / V_X                  # s, time inside one oscillating region
    free = L_FREE / V_X                  # s, time of flight between the two
    omega = np.pi / (2.0 * tau)          # rad/s, a pi/2 pulse in that time
    d = 2.0 * np.pi * (np.asarray(nu, dtype=float) - NU12)

    eff = np.hypot(omega, d)
    theta = 0.5 * eff * tau
    phi = 0.5 * d * free
    front = 4.0 * (omega / eff) ** 2 * np.sin(theta) ** 2
    bracket = np.cos(theta) * np.cos(phi) - (d / eff) * np.sin(theta) * np.sin(phi)
    return front * bracket ** 2


def fringe_spacing() -> float:
    """Hz, the distance between two Ramsey fringes, measured on the lineshape.

    This is the number that decides how coarse a survey is allowed to be, so it
    is worth not guessing it. The tempting guess is 1/T, the inverse of the free
    flight, and it is wrong here: the neutron also accumulates phase inside the
    two oscillating regions, so the fringes are periodic in T + 4*tau/pi rather
    than in T. With this geometry that is 15 Hz and not the 23.5 Hz of 1/T --
    a factor 1.6, and a survey stepped on the wrong one lands on the fringe
    minima and reports a flat curve over a live resonance. Both numbers are
    printed by the summary so the reader can see the difference.

    Measuring the spacing off the curve costs nothing and cannot be wrong about
    which of the two formulas applies.
    """
    nu = np.linspace(NU12 - 60.0, NU12 + 60.0, 24001)
    p = ramsey_transfer(nu)
    tops = [i for i in range(1, len(p) - 1) if p[i] > p[i - 1] and p[i] >= p[i + 1]]
    if len(tops) < 2:
        return V_X / L_FREE
    return float(np.median(np.diff(nu[tops])))


FRINGE = fringe_spacing()


# ---------------------------------------------------------------------------
# One frequency point, measured frame by frame.
# ---------------------------------------------------------------------------
@dataclass
class Point:
    nu: float                 # Hz, the vibration frequency
    f_hat: float              # fitted fraction transferred to n = 2
    f_err: float              # its standard error
    events: int               # neutrons detected at this frequency
    converged: bool           # did it reach the target error before the cap
    counts: np.ndarray = field(repr=False, default=None)
    trace: list = field(repr=False, default_factory=list)


def measure_point(rng, nu, fit, b1, b2, f_max, sigma_target,
                  max_events, start=None, keep_trace=False) -> Point:
    """Sit on one frequency and count until the error bar is small enough.

    Neutrons arrive frame by frame -- Poisson around 39.5 per frame, the 71.1
    per second of Section 4 divided by the 1.80 frames per second the camera
    sustained -- and each one lands in a sensor row drawn from the mixture the
    frequency produces. The fit is redone after every frame, which is the
    cadence the live pipeline already runs at, and the loop leaves as soon as
    the error bar reaches the target.

    `start` continues a point already begun, so the survey measurement at a
    frequency is not thrown away when the fine grid revisits it.

    A floor of two frames is imposed before the stop is allowed. With a handful
    of events the likelihood is nearly flat, its half-unit interval covers most
    of the grid, and the error returned is not yet an error: stopping on it
    would be stopping on noise.
    """
    f_true = f_max * float(ramsey_transfer(nu))
    truth = (1.0 - f_true) * b1 + f_true * b2
    truth = truth / truth.sum()

    counts = np.zeros_like(b1) if start is None else start.copy()
    events = 0 if start is None else int(counts.sum())
    f_hat, f_err = fit(counts) if events else (0.0, 0.5)
    trace, frames = [], 0

    while events < max_events:
        counts += rng.multinomial(rng.poisson(PER_FRAME), truth)
        events = int(counts.sum())
        frames += 1
        f_hat, f_err = fit(counts)
        if keep_trace:
            trace.append((events / RATE, f_hat, f_err))
        if f_err <= sigma_target and frames >= 2:
            return Point(nu, f_hat, f_err, events, True, counts, trace)

    return Point(nu, f_hat, f_err, events, False, counts, trace)


# ---------------------------------------------------------------------------
# Reading a resonance out of a set of points.
# ---------------------------------------------------------------------------
def peak_centre(points: list[Point]) -> float:
    """Frequency of the tallest fringe, from a weighted parabola through it.

    The points are sorted in frequency and the fit uses the run of consecutive
    points around the maximum that stay above half of it. The word consecutive
    is what matters: a Ramsey pattern has several fringes of similar height
    inside one scan, and taking every point above half the maximum wherever it
    lies mixes three fringes into one parabola and returns a vertex that belongs
    to none of them.

    Each point is weighted by its own inverse variance, because after this loop
    the points no longer carry the same statistics -- which is the whole purpose
    of stopping them individually. nan is returned when the top is not a
    maximum, which is the honest answer when there is nothing there.
    """
    order = np.argsort([p.nu for p in points])
    nu = np.array([points[i].nu for i in order])
    f = np.array([points[i].f_hat for i in order])
    err = np.array([max(points[i].f_err, 1e-6) for i in order])
    if f.max() <= 0.0:
        return float("nan")

    top = int(np.argmax(f))
    half = 0.5 * f[top]
    lo, hi = top, top
    while lo > 0 and f[lo - 1] >= half:
        lo -= 1
    while hi < len(f) - 1 and f[hi + 1] >= half:
        hi += 1
    # A parabola needs three points; if the fringe is sampled too sparsely to
    # give three above half maximum, take the immediate neighbours anyway.
    lo, hi = max(0, min(lo, top - 1)), min(len(f) - 1, max(hi, top + 1))
    if hi - lo < 2:
        return float("nan")

    x0 = nu[lo:hi + 1].mean()                  # centre the abscissa, or the
    x = nu[lo:hi + 1] - x0                     # normal equations are ill-posed
    w = 1.0 / err[lo:hi + 1] ** 2
    root = np.sqrt(w)
    design = np.vstack([x ** 2, x, np.ones_like(x)]).T * root[:, None]
    coef, *_ = np.linalg.lstsq(design, f[lo:hi + 1] * root, rcond=None)
    a, b, _ = coef
    if a >= 0:
        return float("nan")
    return float(x0 - b / (2.0 * a))


# ---------------------------------------------------------------------------
# The two strategies.
# ---------------------------------------------------------------------------
@dataclass
class Scan:
    label: str
    points: list[Point]
    events: int
    seconds: float
    centre: float
    abandoned: bool = False
    excluded: float = float("nan")   # the transfer an abandoned scan ruled out


def _cost(points, settle):
    events = sum(p.events for p in points)
    return events, events / RATE + settle * len(points)


def naive_scan(rng, fit, b1, b2, cfg) -> Scan:
    """The scan that is run today: a fixed grid, every point to full precision.

    Same detector, same estimator, same target error as the adaptive scan. The
    only thing it lacks is the right to decide where to go next, so it pays at
    thirty frequencies whether or not the resonance is anywhere near them -- and
    whether or not it is there at all.

    It does keep the right to stop a point on its error bar, which is already
    something only the live profile allows. That is deliberate: the comparison
    printed at the end separates the two rights, because they are worth
    different amounts and only one of them needs a scheduler.
    """
    grid = np.linspace(NU12 - cfg.span, NU12 + cfg.span, cfg.naive_points)
    pts = [measure_point(rng, nu, fit, b1, b2, cfg.f_max, cfg.sigma_fine,
                         cfg.max_events) for nu in grid]
    events, seconds = _cost(pts, cfg.settle)
    return Scan("fixed grid", pts, events, seconds, peak_centre(pts))


def worst_case_dwell(fit, b1, b2, cfg, draws: int = 25) -> float:
    """Events the hardest point of the scan actually needs.

    The hardest point is the one on resonance: the error bar on f is widest
    where the mixture is most even, and narrowest off resonance where the
    profile is one pure state and the other state's rows are empty. This is a
    diagnostic, not a baseline -- it is measured on the truth, and an operator
    setting a dwell before the beam does not have the truth. What they have is
    the moment formula of the outlook, so that is what the fixed-dwell rung of
    the summary uses; this number says how much that formula over-buys.
    """
    rng = np.random.default_rng(20250902)
    costs = [measure_point(rng, NU12, fit, b1, b2, cfg.f_max, cfg.sigma_fine,
                           cfg.max_events).events for _ in range(draws)]
    return float(np.mean(costs))


def adaptive_scan(rng, fit, b1, b2, cfg) -> Scan:
    """Survey coarsely, decide, then spend the beam only where it pays.

    Stage 1 -- survey. The grid step is set by the fringe spacing and not by
    taste: sampled coarser than a fringe, a survey aliases the pattern and can
    report a flat curve over a live resonance. Half a fringe is the textbook
    Nyquist limit and it is not enough here, because sampling exactly at Nyquist
    is free to land on every minimum and miss every maximum -- which is what the
    first version of this file did, and it abandoned a resonance that was there.
    A third of a fringe puts a sample within a sixth of a fringe of every peak
    and costs three points instead of two. Their precision is deliberately poor:
    the survey only has to answer "is there contrast, and roughly where".

    Stage 2 -- decide. If no survey point stands three of its own error bars
    above zero, there is nothing here. The scan stops and reports the transfer
    it has excluded, which is the useful statement -- "nothing found" alone is
    not one.

    The survey precision is not a free parameter and is not chosen for speed:
    it is min_transfer / n_sigma, where min_transfer is the smallest population
    transfer the operator refuses to walk away from. That is the whole contract
    of the abandon rule, and it is worth stating out loud because it costs
    money: the survey scales as one over min_transfer squared, so halving what
    you are willing to miss quadruples the price of finding out there is
    nothing there. The gain this file measures is against a stated
    min_transfer, and it shrinks as that number does.

    Stage 3 -- zoom. A fine grid is centred on the best survey frequency, half
    a fringe either side, at the target precision. The survey point at the
    centre is continued rather than restarted: its neutrons are as good as any
    other.

    "Best" is read off the survey smoothed over one fringe width, not off the
    single tallest survey point. What marks the resonance is the envelope, and
    the envelope is what a one-fringe average estimates; the individual fringes
    inside it differ by six percent, which is less than the survey resolves.
    Over two hundred surveys the smoothed choice lands on the central fringe 43
    percent of the time against 36 for the raw maximum -- an improvement, and
    nowhere near enough. See the note on fringe order in main().
    """
    step_max = FRINGE / cfg.survey_per_fringe
    n_survey = int(np.ceil(2 * cfg.span / step_max)) + 1
    survey_grid = np.linspace(NU12 - cfg.span, NU12 + cfg.span, n_survey)
    survey = [measure_point(rng, nu, fit, b1, b2, cfg.f_max, cfg.sigma_coarse,
                            cfg.survey_cap) for nu in survey_grid]

    significance = max(p.f_hat / max(p.f_err, 1e-9) for p in survey)
    if significance < cfg.n_sigma:
        events, seconds = _cost(survey, cfg.settle)
        # Quoted as a transfer at the peak, which is what an operator decides
        # on, so the survey's own blindness between samples has to be divided
        # back out. The survey usually beats its target -- off resonance the
        # profile is one pure state and the fit converges fast -- so this is
        # normally a stronger statement than min_transfer.
        typical = float(np.median([p.f_err for p in survey]))
        return Scan("adaptive", survey, events, seconds, float("nan"),
                    abandoned=True,
                    excluded=cfg.n_sigma * typical / cfg.survey_loss)

    # Two steps, because they answer two different questions. The envelope, a
    # one-fringe running mean, says which group of fringes the resonance sits
    # in. The raw survey maximum inside that group says which crest, and it is
    # the crest the zoom window has to be centred on: centred on a minimum
    # instead, the window straddles two half-fringes and the parabola fits
    # something that is not a fringe. That was the one scan in sixty that came
    # back five hertz out before this line existed.
    raw = np.array([p.f_hat for p in survey])
    width = max(3, int(round(cfg.survey_per_fringe)) | 1)      # odd, one fringe
    envelope = np.convolve(raw, np.ones(width) / width, mode="same")
    guess = int(np.argmax(envelope))
    lo = max(0, guess - width // 2)
    crest = lo + int(np.argmax(raw[lo:guess + width // 2 + 1]))
    best = survey[crest]
    half = 0.5 * FRINGE
    fine_grid = np.linspace(best.nu - half, best.nu + half, cfg.fine_points)

    fine = []
    for nu in fine_grid:
        # The survey already sat at the centre of this window; carry its counts
        # over instead of paying for them twice.
        seed = best.counts if abs(nu - best.nu) < 1e-9 else None
        trace = cfg.trace and abs(nu - best.nu) < 1e-9
        fine.append(measure_point(rng, nu, fit, b1, b2, cfg.f_max,
                                  cfg.sigma_fine, cfg.max_events,
                                  start=seed, keep_trace=trace))

    kept_survey = [p for p in survey if p is not best]
    points = kept_survey + fine
    events, seconds = _cost(points, cfg.settle)
    return Scan("adaptive", points, events, seconds, peak_centre(fine))


# ---------------------------------------------------------------------------
# Driving both, and comparing them.
# ---------------------------------------------------------------------------
@dataclass
class Config:
    span: float = 40.0            # Hz, half-width of the scanned range
    f_max: float = 0.30           # transfer at exact resonance
    min_transfer: float = 0.30    # the smallest transfer worth not abandoning
    sigma_fine: float = 0.01      # error bar the measurement needs
    # One percent on the fraction, which is the precision the laboratory
    # already works to: at this target the absorber rung reproduces the scan
    # as it is run today, so the ladder starts from a number the group
    # recognises instead of from one that has to be translated.
    fine_points: int = 9
    naive_points: int = TRANSMISSION_POINTS
    survey_per_fringe: float = 3.0  # survey samples per fringe, see adaptive_scan
    n_sigma: float = 3.0          # contrast threshold to keep going
    survey_cap: int = 400         # events, the ceiling on one survey point
    max_events: int = 6000        # events, the ceiling on one fine point
    settle: float = 0.0           # s, dead time to retune the piezo per point
    trace: bool = False

    @property
    def survey_step(self) -> float:
        """Hz between survey frequencies. See adaptive_scan for why it is a
        third of a fringe and not a half."""
        return FRINGE / self.survey_per_fringe

    @property
    def survey_loss(self) -> float:
        """The fraction of the peak transfer the survey is guaranteed to see.

        A survey sample sits anywhere up to half a step from the nearest fringe
        crest, and half a step down the flank of a fringe the transfer has
        already fallen. Rejecting a transfer of min_transfer at the peak
        therefore means measuring min_transfer times this factor, not
        min_transfer -- forgetting it is what made the loop walk away from a
        real ten-percent resonance in three scans out of ten."""
        return float(ramsey_transfer(NU12 + 0.5 * self.survey_step))

    @property
    def sigma_coarse(self) -> float:
        """The survey error bar follows from the abandon contract, not the
        other way round: to reject a transfer of min_transfer at n_sigma, on a
        grid that only sees survey_loss of it, the survey has to measure f to
        survey_loss * min_transfer / n_sigma and no better."""
        return self.survey_loss * self.min_transfer / self.n_sigma


def run_pair(seed, fit, b1, b2, cfg):
    """One adaptive scan and one fixed-grid scan, on independent draws.

    Separate generators, both derived from the same seed, so that changing the
    adaptive strategy does not silently change the neutrons the fixed grid sees
    and the comparison stays a comparison.
    """
    a = adaptive_scan(np.random.default_rng([seed, 1]), fit, b1, b2, cfg)
    n = naive_scan(np.random.default_rng([seed, 2]), fit, b1, b2, cfg)
    return a, n


def in_fringe(centre: float) -> float:
    """How far the measured fringe sits from where a fringe should be, in Hz.

    A Ramsey pattern repeats, so a scan that reports a fringe at nu12 + 14.7 Hz
    has not made a 14.7 Hz error: it has measured the neighbouring fringe, to
    whatever precision, and the fringe order is fixed by prior knowledge of nu12
    rather than by the scan. Folding the residual into one fringe separates the
    two questions -- which fringe, and how well placed -- and the summary
    reports both."""
    return (centre - NU12 + 0.5 * FRINGE) % FRINGE - 0.5 * FRINGE


def describe(scan: Scan) -> str:
    if scan.abandoned:
        return (f"  {scan.label:<12s} {len(scan.points):3d} points  "
                f"{scan.events:6d} events  {scan.seconds:8.1f} s   "
                f"ABANDONED (no transfer above {scan.excluded:.2f})")
    order = round((scan.centre - NU12) / FRINGE)
    return (f"  {scan.label:<12s} {len(scan.points):3d} points  "
            f"{scan.events:6d} events  {scan.seconds:8.1f} s   "
            f"fringe {order:+d} at {scan.centre:7.2f} Hz  "
            f"({in_fringe(scan.centre):+.2f} Hz off centre)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=1,
                    help="reproducibility; every number below is a function of it")
    ap.add_argument("--repeats", type=int, default=1,
                    help="scans to run; >1 reports the spread instead of one draw")
    ap.add_argument("--f-max", type=float, default=Config.f_max,
                    help="population transferred at exact resonance; 0 is a dead "
                         "working point and exercises the abandon path")
    ap.add_argument("--sigma-fine", type=float, default=Config.sigma_fine,
                    help="the error bar on f the measurement is after")
    ap.add_argument("--min-transfer", type=float, default=Config.min_transfer,
                    help="the smallest transfer the loop refuses to abandon; "
                         "the survey costs one over its square, so this is the "
                         "knob that decides how much the loop actually saves")
    ap.add_argument("--span", type=float, default=Config.span,
                    help="Hz, half-width of the frequency range scanned")
    ap.add_argument("--settle", type=float, default=Config.settle,
                    help="s of dead time per frequency point; 0 is the choice "
                         "that flatters the fixed grid, which visits more points")
    a = ap.parse_args()

    cfg = Config(span=a.span, f_max=a.f_max, sigma_fine=a.sigma_fine,
                 min_transfer=a.min_transfer, settle=a.settle,
                 trace=(a.repeats == 1))

    _centres, b1, b2, sc = sensor_templates()
    fit = ProfileFit(b1, b2)

    print("Ramsey feedback loop -- what the live profile buys in beam time")
    print("=" * 70)
    print(f"detector    {sc['rows']} sensor rows of {PITCH:g} um over "
          f"{Z_MAX:g} um, blur {DISPLACEMENT:g} um")
    print(f"            mean height n=1 {sc['mean1']:.2f} um, n=2 "
          f"{sc['mean2']:.2f} um, shift {sc['shift']:.2f} um")
    print(f"            profile width {sc['width']:.1f} um, "
          f"{RATE:g} neutrons/s at {FRAME_RATE:g} frames/s "
          f"({PER_FRAME:.1f} per frame)")
    print(f"resonance   nu12 = {NU12:g} Hz, transfer at peak {cfg.f_max:g}")
    print(f"            fringe spacing {FRINGE:.1f} Hz measured on the "
          f"lineshape (1/T alone would say {V_X / L_FREE:.1f})")
    print(f"scan        +/-{cfg.span:g} Hz, target error on f "
          f"{cfg.sigma_fine:g}")
    print(f"            abandon below a transfer of {cfg.min_transfer:g}: "
          f"survey stepped {cfg.survey_step:.1f} Hz, which sees at worst "
          f"{cfg.survey_loss:.2f}")
    print(f"            of the peak, so the survey measures f to "
          f"{cfg.sigma_coarse:.3f}")

    # The closed form the outlook quotes, against what the simulation actually
    # pays. N = (3 sigma / f dz)^2 at three sigma is the same statement as
    # sigma_f = width / (shift sqrt(N)); the likelihood fit uses the whole shape
    # and not just the first moment, so it is allowed to do better, and off
    # resonance -- where the profile is one pure state and the other state's
    # rows are empty -- it does. On resonance the two should agree, and the two
    # numbers printed here are the check that they do.
    n_closed = (sc["width"] / (cfg.sigma_fine * sc["shift"])) ** 2
    dwell = worst_case_dwell(fit, b1, b2, cfg)
    print(f"            worst point costs {dwell:.0f} events "
          f"({dwell / RATE:.1f} s); the moment formula of the outlook says "
          f"{n_closed:.0f}")
    print()

    if a.repeats == 1:
        adapt, naive = run_pair(a.seed, fit, b1, b2, cfg)
        print("one scan")
        print(describe(adapt))
        print(describe(naive))
        print()

        # The error bar shrinking on the on-resonance point, which is the thing
        # a transmission measurement can never show while it is running.
        traced = [p for p in adapt.points if p.trace]
        if traced:
            t = traced[0]
            print(f"live fit at nu = {t.nu:.1f} Hz "
                  f"(true f = {cfg.f_max * float(ramsey_transfer(t.nu)):.3f})")
            for i, (secs, f_hat, f_err) in enumerate(t.trace):
                if i % 4 == 0 or i == len(t.trace) - 1:
                    print(f"    {secs:6.2f} s   f = {f_hat:.3f} +/- {f_err:.3f}")
            print()

        adapt_s, naive_s = adapt.seconds, naive.seconds
        spread = ""
    else:
        rows = [run_pair(a.seed + k, fit, b1, b2, cfg) for k in range(a.repeats)]
        at = np.array([r[0].seconds for r in rows])
        nt = np.array([r[1].seconds for r in rows])
        ac = np.array([r[0].centre for r in rows])
        nc = np.array([r[1].centre for r in rows])
        n_ab = sum(r[0].abandoned for r in rows)

        print(f"{a.repeats} scans")
        print(f"  adaptive     {at.mean():8.1f} s  "
              f"(spread {at.std():.1f})   abandoned {n_ab}/{a.repeats}")
        print(f"  fixed grid   {nt.mean():8.1f} s  (spread {nt.std():.1f})")
        if n_ab:
            worst = max(r[0].excluded for r in rows if r[0].abandoned)
            print(f"               an abandoned scan excludes any transfer "
                  f"above {worst:.2f}, and says so")
        # A cheaper scan that reads the resonance less well is not a cheaper
        # scan, so both are judged the same way and on two separate questions:
        # how well the fringe they measured is placed, and whether it was the
        # central one. The second question is the one neither method answers
        # reliably, and the reason is physics rather than scheduling -- see the
        # note printed below.
        # Median and ninetieth percentile rather than mean and standard
        # deviation: both distributions have a tail of scans that fitted the
        # wrong side of a fringe, and one such scan moves a standard deviation
        # far more than it moves the quality of the method.
        for name, c in (("adaptive", ac), ("fixed grid", nc)):
            ok = np.isfinite(c)
            if not ok.any():
                print(f"  {name:<12s} no scan returned a fringe")
                continue
            res = np.abs([in_fringe(v) for v in c[ok]])
            right = int(np.sum(np.abs(c[ok] - NU12) < 0.5 * FRINGE))
            print(f"  {name:<12s} fringe placed to {np.median(res):.2f} Hz "
                  f"(90% under {np.percentile(res, 90):.2f}) on "
                  f"{ok.sum()}/{a.repeats}   central fringe "
                  f"{right}/{a.repeats}")
        if cfg.f_max == 0:
            print("               there is no resonance here: every fringe "
                  "the fixed grid reports is noise,")
            print("               which is what the loop is refusing to "
                  "spend six minutes on")
        adapt_s, naive_s = float(at.mean()), float(nt.mean())
        spread = " (mean over the repeats)"
        print()

    # Four rungs, each one right giving up exactly one thing the rung above it
    # kept, so that the total is attributable instead of being one impressive
    # ratio with three causes inside it.
    today = TRANSMISSION_S_PER_POINT * TRANSMISSION_POINTS
    # Every rung has to aim at the same error on f, or the ladder measures a
    # choice of precision as though it were a property of a detector.
    #
    # The absorber reads f off a rate. Counting N over a dwell at a frequency
    # where a fraction f has been transferred, N is Poisson with mean R0*t*(1-f)
    # and the estimator is f = 1 - N/(R0*t), so
    #
    #     sigma_f = sqrt((1 - f) / N0),   N0 = R0*t the count at f = 0.
    #
    # R0 itself is free here: a thirty-point scan measures it on its own wings,
    # unlike the isolated detection of the outlook where a change is the
    # difference of two measurements and the count doubles.
    #
    # A fixed dwell is ONE duration for all thirty frequencies, so it is sized
    # for the frequency that needs the most counts. That is the worst case
    # (1-f) = 1, off resonance, where the expression collapses to 1/sqrt(N).
    # Sizing on the resonance instead -- (1-f) = 0.7 at a 30% transfer -- would
    # give 328 s and a dwell too short for every point in the wings, which is
    # most of the scan. The sensor rung is sized the same way, on the moment
    # formula, which carries no f either.
    absorber = cfg.naive_points * (1.0 / cfg.sigma_fine ** 2 / RATE + cfg.settle)
    fixed_dwell = (cfg.naive_points * n_closed / RATE
                   + cfg.settle * cfg.naive_points)
    ladder = [
        ("absorber, fixed dwell", absorber,
         "one number per frequency, same error on f"),
        ("sensor, fixed dwell", fixed_dwell,
         "profile per frequency, dwell set for the worst point"),
        ("sensor, live stop", naive_s,
         "same grid, each point stopped on its own error bar"),
        ("sensor, feedback loop", adapt_s,
         "survey, decide, zoom"),
    ]
    print(f"beam time{spread}")
    prev = None
    for name, secs, why in ladder:
        gain = "" if prev is None else f"x{prev / max(secs, 1e-9):.1f}"
        print(f"  {name:<24s}{secs:8.1f} s  {gain:>5s}   {why}")
        prev = secs
    print()
    print(f"  end to end x{absorber / max(adapt_s, 1e-9):.1f} at one error bar "
          f"for every rung. Of that, x{absorber / max(fixed_dwell, 1e-9):.1f} is")
    print("  the profile against the rate and the rest is the scheduling. The "
          "live stop is a sensor")
    print("  property even so: 1/sqrt(N) does not depend on the working point, "
          "so a counter gains")
    print("  nothing by watching itself, while a profile fit does.")
    print()
    if abs(cfg.sigma_fine - 0.01) < 1e-9:
        print(f"  the first rung is the scan as the laboratory runs it: "
              f"{TRANSMISSION_S_PER_POINT:.1f} s per point at 1%,")
        print(f"  {TRANSMISSION_POINTS} points, {today:.0f} s -- the same "
              f"number as the rung, by construction.")
    else:
        print(f"  at 1% per point -- the precision the laboratory works to -- "
              f"the same scan takes {today:.0f} s.")
        print(f"  These rungs aim at {cfg.sigma_fine:g}, so their seconds are "
              f"not comparable with it.")
    print()
    print(f"  what the saving is worth: beam time converts into precision at "
          f"fixed shift length, so")
    print(f"  x{absorber / max(adapt_s, 1e-9):.1f} in time is "
          f"x{(absorber / max(adapt_s, 1e-9)) ** 0.5:.1f} on the centre of the "
          f"fringe once the scan is repeated.")

    if a.repeats > 1 and cfg.f_max > 0:
        side = float(ramsey_transfer(NU12 + FRINGE)) * cfg.f_max
        print()
        print("what neither method does")
        print(f"  the first side fringe transfers {side:.3f} against "
              f"{cfg.f_max:.3f} at the centre: a gap of "
              f"{cfg.f_max - side:.3f}")
        print(f"  against error bars of {cfg.sigma_fine:g}. Picking the "
              f"central fringe out of a scan this wide is")
        print("  therefore not a question of scheduling, and neither strategy "
              "settles it. The fringe")
        print("  order comes from knowing nu12 beforehand; what the scan "
              "measures is the position")
        print("  of a fringe, and that is the residual quoted above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
