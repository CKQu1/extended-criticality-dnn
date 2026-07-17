"""Real-time viewer for the population-dynamics cavity of the Jacobian
singular-value resolvent: the pool of resolvent diagonal entries G_ii (both
bipartite families g1, g2 of RMT.cavity_svd_resolvent) as a live scatter in
the complex plane at one singular value s, at a fixed pool size, sweeping
forever until stopped.

Engine: numba_cavity._run_steps -- the exact Gauss-Seidel semantics of
RMT.cavity_svd_resolvent, complex128, JIT-compiled (~4 ms/sweep at pop=128
vs ~53 ms for the torch kernel, whose per-step op dispatch dominates). The
kernel runs on numpy views of torch pool tensors, so pool reshaping (tiling,
dtype, randomise) is plain tensor work. chi disorder and q* follow the
production construction (RMT.jac_cavity_svd_log_pdf / q_star_MC).

Rendering: ipycanvas -- the server sends ~KBs of point coordinates per frame
and the browser rasterizes them at display refresh rate (a server-rendered
matplotlib PNG costs ~140 ms/frame and caps ipympl at roughly 7-15 fps).

Front-end: cavity_pool_viewer.ipynb (same directory), for VS Code Remote SSH.

(Removed 2026-07: the matplotlib ipympl LiveScatter, the PoolViewer
doubling-ladder dashboard, and the CLI GIF recorder -- see git history.)
"""
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
import RMT
from numba_cavity import _run_steps

# dataviz reference palette (light mode, pre-validated categorical order)
C_G1 = "#2a78d6"  # slot 1 blue: g1 pool
C_G2 = "#008300"  # slot 2 green: g2 pool
C_TRAIL = "#52514e"  # text-secondary: motion trails
C_GRID = "#d9d8d4"

IMLOG_FLOOR = -30.0  # default display floor for log10 |Im G|


class Timer:
    """Context manager: prints '[label] elapsed' and logs into a shared dict."""

    def __init__(self, label, log=None):
        self.label, self.log = label, log

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self.t0
        print(f"[{self.label}] {dt:.2f} s", flush=True)
        if self.log is not None:
            self.log[self.label] = self.log.get(self.label, 0.0) + dt


def build_chi(alpha, sigma_W, q, num_chis, pop, phi="tanh"):
    """chi draw exactly as RMT.jac_cavity_svd_log_pdf (fresh per pool size).
    phi in {"tanh", "linear"}: linear has phi' = 1, so chi = sigma_W
    uniformly (no chi disorder -- the unstructured Levy ensemble) and q
    drops out."""
    if phi == "linear":
        return sigma_W * torch.ones((num_chis, pop))
    phi_prime = torch.vmap(torch.vmap(torch.func.grad(torch.tanh)))
    if alpha != 2:
        stable_samples = RMT.stable_dist_sample(
            alpha, scale=2 ** (-1 / alpha), size=(num_chis, pop)
        )
    else:
        stable_samples = torch.randn(num_chis, pop)
    return sigma_W * phi_prime(q ** (1 / alpha) * stable_samples)


class LiveCanvas:
    """Live ipycanvas scatter of the cavity resolvent diagonal pool.

    Build it, then live.show() displays the canvas, pause/continue and
    randomise buttons, and one widget per knob; the sweep loop runs in a
    background thread so the cell returns immediately and the controls stay
    responsive. The viewer STARTS PAUSED with the initial pool drawn and the
    linear Im axis -- press continue (paused=False / log_im=True override).
    End the run with live.stop() (prints sweep-rate timing). Creating a new
    LiveCanvas stops any previous one (safe cell re-runs).

    Knobs (widgets, or live.configure(...)): alpha, sigma_W, s (log slider),
    pop (64..8192), eta, phi ("tanh" or "linear" activation: linear has
    phi' = 1, so chi = sigma_W uniformly -- the unstructured Levy ensemble,
    with q* dropping out), dtype (complex128/complex64), the log/linear
    Im-axis toggle, axis limits (xmin/xmax/ymin/ymax FloatTexts, in axis
    units: with the log Im-axis, ymin/ymax are log10 exponents, e.g.
    ymin=-30 means 1e-30; toggling the scale resets them to that mode's
    defaults), steps/frame, and trails; pace (extra sleep per frame) is
    constructor/configure-only, with no widget. Changes are queued and
    applied by the loop thread BETWEEN sweeps, so the run never restarts:
    moving a slider is a live quench of the running population.
    alpha/sigma_W/phi changes recompute q* (tanh only) and redraw the chi
    disorder; pop changes tile or truncate the pool.

    One frame per sweep by default, capped at max_fps (default 60) so the
    comm channel is not flooded. Sweep cost scales as pop^2: pop<=256
    sustains 55-60 fps whole-sweep, pop=512 ~15 fps. For larger pools set
    steps_per_frame < pop: each frame advances k steps and then rolls the
    pools and chi by -k, so the kernel's `i % pop` scan continues where it
    left off. Chunked frames are a different REALIZATION of the same update
    law as unchunked sweeps (not bitwise identical: the fresh iid stable
    weights are drawn by position, so they pair with relabeled slots -- a
    pairing that leaves the process law unchanged since the weights are iid
    and the permutation is deterministic). E.g. pop=1024 with
    steps_per_frame=32 draws 60 fps while completing ~3.7 sweeps/s.

    The title line shows the running pool means over both families: mean
    Im G and mean log10 |Im G| -- the latter is the growth-rate observable
    of RMT/localisation.py (its per-sweep slope is phi(s)); exact zeros
    from a flushed complex64 pool pin at the 1e-300 clip. All static
    parameters live in the widgets, not the title.

    Motion trails on a tracked subset of g1 entries are drawn under the
    points (toggle: the "trails" checkbox or trails=). Trails follow slot
    identity through the sub-sweep rolls and restart on any change that
    reshuffles or re-draws the pool.

    dtype note: "complex64" rounds the pool STORAGE to complex64 on every
    update, while the kernel's accumulator remains double precision -- not
    bitwise the torch cfloat arithmetic, but it reproduces the production
    artifact: marginal Im G underflows float32 (~1e-38) and flushes to
    exact zero, and with real z and eta=0 a flushed pool never recovers
    (verified: 150 sweeps at s=5 zero out the whole pool in complex64;
    complex128 collapses smoothly to ~1e-190). Lower ymin below -30 to see
    the difference on screen.

    If the canvas renders blank in VS Code, allow widget scripts from the
    CDN: setting "Jupyter: Widget Script Sources" -> add "jsdelivr.com".
    """

    # canvas margins: left, right, top, bottom
    ML, MR, MT, MB = 60, 12, 34, 38

    _active = None

    def __init__(self, alpha=1.5, sigma_W=1.0, s=2.0, pop=256, q=None,
                 seed=0, eta=0.0, phi="tanh", log_im=False, pace=0.0,
                 width=680, height=460, max_fps=60, steps_per_frame=0,
                 trails=True, paused=True):
        if LiveCanvas._active is not None:
            LiveCanvas._active.stop()
        LiveCanvas._active = self
        self.tlog = {}
        self.phi = "linear" if str(phi).lower().startswith("lin") else "tanh"
        with Timer("setup: q_star", self.tlog):
            if q is None:
                if self.phi == "tanh":
                    torch.manual_seed(0)  # fixed: all runs of a cell share one q
                    q = float(RMT.q_star_MC(alpha, sigma_W)[-1])
                else:
                    q = float("nan")  # linear phi: chi = sigma_W, q drops out
        self.alpha, self.sigma_W, self.q = alpha, sigma_W, q
        self.s, self.pop, self.eta = s, pop, eta
        self.log_im, self.pace = log_im, pace
        self.W, self.H = int(width), int(height)
        self.max_fps = max_fps
        self.steps_per_frame = max(0, int(steps_per_frame))

        torch.manual_seed(seed)
        self.g1 = torch.rand((1, 1, pop), dtype=torch.cdouble)
        self.g2 = torch.rand((1, 1, pop), dtype=torch.cdouble)
        self.chi = build_chi(alpha, sigma_W, q, 1, pop, self.phi)

        self.sweeps = 0
        self._paused = bool(paused)
        self._stop = False
        self._thread = None
        self._t_run = 0.0
        self._pending = {}
        self._fps = 0.0
        self._last_frame = None
        self._step_accum = 0
        self.trails_on = bool(trails)
        self.n_trail, trail_len = 8, 20
        self._trails = [deque(maxlen=trail_len) for _ in range(self.n_trail)]
        self._roll_phase = 0  # cumulative slot relabeling from sub-sweep rolls
        self.xmin, self.xmax = -5.0, 5.0
        self.ymin, self.ymax = self._default_ylim()

        with Timer("setup: canvas", self.tlog):
            self._build_canvas()
        with Timer("setup: numba warmup", self.tlog):
            self._sync_engine()
            _run_steps(self._g1v, self._g2v, self._chi2, self._zv,
                       self.alpha, self._scale, 0)  # compile only
        print(f"alpha={alpha} sigma_W={sigma_W} q*={q:.4f} pop={pop} "
              f"s={s} seed={seed} eta={eta}")

    # -- axis helpers -------------------------------------------------------
    def _default_ylim(self):
        """y limits in axis units: log10 exponents in log mode, raw linear."""
        return (IMLOG_FLOOR, 3.0) if self.log_im else (-3.0, 3.0)

    def _floor_exp(self):
        """Numeric clip exponent for |Im G|: follows ymin below the default
        display floor so lowering ymin reveals deeper values (capped at
        1e-300 to stay clear of float64 underflow; exact zeros pin there)."""
        return max(min(IMLOG_FLOOR, self.ymin), -300.0)

    # -- live reconfiguration ----------------------------------------------
    def configure(self, **kw):
        """Queue knob changes (alpha, sigma_W, s, pop, eta, pace, log_im,
        xmin/xmax/ymin/ymax, dtype, steps_per_frame, trails, randomise);
        the loop thread applies them between sweeps so the run never
        restarts. Applied immediately if the loop is not running."""
        allowed = {"alpha", "sigma_W", "s", "pop", "eta", "phi", "pace",
                   "log_im", "xmin", "xmax", "ymin", "ymax", "randomise",
                   "dtype", "steps_per_frame", "trails"}
        bad = set(kw) - allowed
        if bad:
            raise ValueError(f"unknown knobs: {sorted(bad)}")
        self._pending.update(kw)
        if self._thread is None or not self._thread.is_alive():
            self._apply_pending()
            self._update_artists()
            self._render()

    def _apply_pending(self):
        pend, self._pending = self._pending, {}
        if not pend:
            return
        if "trails" in pend:
            self.trails_on = bool(pend["trails"])
            for t in self._trails:
                t.clear()
        if "pace" in pend:
            self.pace = float(pend["pace"])
        if "steps_per_frame" in pend:
            self.steps_per_frame = max(0, int(pend["steps_per_frame"]))
        if "log_im" in pend:
            self.log_im = bool(pend["log_im"])
            # y limits are in axis units, so switching scale resets them
            self.ymin, self.ymax = self._default_ylim()
            self._sync_limit_widgets()
        for k in ("xmin", "xmax", "ymin", "ymax"):
            if k in pend:
                setattr(self, k, float(pend[k]))
        if self.xmin >= self.xmax:
            self.xmax = self.xmin + 1.0
        if self.ymin >= self.ymax:
            self.ymax = self.ymin + 1.0
        if "s" in pend:
            self.s = float(pend["s"])
        if "eta" in pend:
            self.eta = float(pend["eta"])
        if "phi" in pend:
            self.phi = ("linear" if str(pend["phi"]).lower().startswith("lin")
                        else "tanh")
        new_params = ("alpha" in pend or "sigma_W" in pend or "phi" in pend)
        if new_params:
            self.alpha = float(pend.get("alpha", self.alpha))
            self.sigma_W = float(pend.get("sigma_W", self.sigma_W))
            if self.phi == "tanh":
                with Timer("reconfig: q_star", self.tlog):
                    torch.manual_seed(0)
                    self.q = float(RMT.q_star_MC(self.alpha, self.sigma_W)[-1])
            else:
                self.q = float("nan")  # linear phi: q drops out of chi
            print(f"[reconfig] alpha={self.alpha:g} sigma_W={self.sigma_W:g} "
                  f"phi={self.phi} q*={self.q:.4f}", flush=True)
        if "dtype" in pend:
            dt = (torch.cdouble if "128" in str(pend["dtype"])
                  else torch.cfloat)
            if self.g1.dtype != dt:
                self.g1 = self.g1.to(dt)
                self.g2 = self.g2.to(dt)
        if "pop" in pend and int(pend["pop"]) != self.pop:
            new_pop = int(pend["pop"])
            if new_pop > self.pop:
                reps = -(-new_pop // self.pop)  # ceil division
                self.g1 = self.g1.repeat(1, 1, reps)[..., :new_pop].contiguous()
                self.g2 = self.g2.repeat(1, 1, reps)[..., :new_pop].contiguous()
            else:
                self.g1 = self.g1[..., :new_pop].contiguous()
                self.g2 = self.g2[..., :new_pop].contiguous()
            self.pop = new_pop
            self._roll_phase = 0
            new_params = True  # chi must match the new pool size
        if pend.get("randomise"):
            self.g1 = torch.rand(self.g1.shape, dtype=self.g1.dtype)
            self.g2 = torch.rand(self.g2.shape, dtype=self.g2.dtype)
        if new_params:
            self.chi = build_chi(self.alpha, self.sigma_W, self.q, 1,
                                 self.pop, self.phi)
        self._sync_engine()
        # trails follow slot identity, so anything that reshuffles or
        # re-draws the pool restarts them
        if set(pend) & {"pop", "randomise", "dtype", "s", "alpha", "sigma_W",
                        "phi"}:
            for t in self._trails:
                t.clear()
        self._draw_background()

    def _sync_limit_widgets(self):
        """Push current limits into the FloatText widgets (after a scale
        toggle resets them); traitlets only fires observe on real changes."""
        for k in ("xmin", "xmax", "ymin", "ymax"):
            w = getattr(self, "_limit_widgets", {}).get(k)
            if w is not None:
                w.value = getattr(self, k)

    def _sync_engine(self):
        """Refresh numpy views/derived arrays after any pool/param change."""
        self._g1v = self.g1.numpy()
        self._g2v = self.g2.numpy()
        self._chi2 = self.chi.numpy().astype(np.float64) ** 2
        self._zv = (np.array([self.s + 1j * self.eta])
                    if self.eta > 0 else np.array([float(self.s)]))
        self._scale = (2.0 * self.pop) ** (-1.0 / self.alpha)

    def randomise(self):
        """Re-draw both pools uniformly at random (fresh initial condition,
        same size and parameters); applied between sweeps like any knob."""
        self.configure(randomise=True)

    # -- engine -------------------------------------------------------------
    def step(self, n=1):
        """Advance n frames. A frame is one full sweep (pop steps), or
        steps_per_frame steps if that is set smaller: after a k-step chunk
        the pools and chi are rolled by -k so the kernel's `i % pop` scan
        continues where it left off (chi rolls with the g2 slots it is
        bound to). Equivalent in law -- not bitwise -- to unchunked sweeps;
        see the class docstring."""
        spf = self.steps_per_frame
        chunk = spf if 0 < spf < self.pop else self.pop
        for _ in range(n):
            _run_steps(self._g1v, self._g2v, self._chi2, self._zv,
                       self.alpha, self._scale, chunk)
            if chunk < self.pop:
                self._g1v[:] = np.roll(self._g1v, -chunk, -1)
                self._g2v[:] = np.roll(self._g2v, -chunk, -1)
                self._chi2[:] = np.roll(self._chi2, -chunk, -1)
                self.chi = torch.roll(self.chi, -chunk, -1)
                self._roll_phase = (self._roll_phase + chunk) % self.pop
            self._step_accum += chunk
            self.sweeps += self._step_accum // self.pop
            self._step_accum %= self.pop
            if self.trails_on:
                self._append_trails()
        self._update_artists()

    def _append_trails(self):
        """Record current positions of the tracked members (the first few
        original slots, followed through the roll relabeling)."""
        for m, t in enumerate(self._trails[:min(self.n_trail, self.pop)]):
            j = (m - self._roll_phase) % self.pop
            t.append(complex(self._g1v[0, 0, j]))

    # -- view: two-layer canvas (static axes below, moving pool above) -----
    def _build_canvas(self):
        from ipycanvas import MultiCanvas
        self.canvas = MultiCanvas(2, width=self.W, height=self.H)
        self._draw_background()

    def _xy_pix(self, gv):
        x = np.clip(gv.real, self.xmin, self.xmax)
        if self.log_im:
            # float64 before clipping (float32 imag underflows sub-1e-38 floors)
            v = np.log10(np.clip(np.abs(gv.imag).astype(np.float64),
                                 10.0**self._floor_exp(), None))
        else:
            v = gv.imag
        v = np.clip(v, self.ymin, self.ymax)
        ynorm = (self.ymax - v) / (self.ymax - self.ymin)
        xp = self.ML + ((x - self.xmin) / (self.xmax - self.xmin)
                        * (self.W - self.ML - self.MR))
        yp = self.MT + ynorm * (self.H - self.MT - self.MB)
        return xp, yp

    def _draw_background(self):
        from ipycanvas import hold_canvas
        if getattr(self, "canvas", None) is None:
            return
        bg = self.canvas[0]
        x0, x1 = self.ML, self.W - self.MR
        y0, y1 = self.MT, self.H - self.MB
        with hold_canvas(bg):
            bg.clear()
            bg.fill_style = "#ffffff"
            bg.fill_rect(0, 0, self.W, self.H)
            # gridlines + tick labels (locations follow the live limits)
            from matplotlib.ticker import MaxNLocator
            bg.stroke_style = C_GRID
            bg.line_width = 1
            bg.fill_style = "#52514e"
            bg.font = "11px sans-serif"
            bg.text_align = "center"
            xticks = [t for t in MaxNLocator(nbins=6)
                      .tick_values(self.xmin, self.xmax)
                      if self.xmin <= t <= self.xmax]
            for xv in xticks:
                xp = x0 + (xv - self.xmin) / (self.xmax - self.xmin) * (x1 - x0)
                bg.stroke_line(xp, y0, xp, y1)
                bg.fill_text(f"{xv:g}", xp, y1 + 16)
            bg.text_align = "right"
            yticks = [t for t in MaxNLocator(nbins=8, integer=self.log_im)
                      .tick_values(self.ymin, self.ymax)
                      if self.ymin <= t <= self.ymax]
            for v in yticks:
                lab = f"1e{v:g}" if self.log_im else f"{v:g}"
                yp = y0 + (self.ymax - v) / (self.ymax - self.ymin) * (y1 - y0)
                bg.stroke_line(x0, yp, x1, yp)
                bg.fill_text(lab, x0 - 6, yp + 4)
            # frame
            bg.stroke_style = "#52514e"
            bg.stroke_rect(x0, y0, x1 - x0, y1 - y0)
            # axis labels + title
            bg.text_align = "center"
            bg.fill_text("Re G_ii", (x0 + x1) / 2, self.H - 6)
            ylab = "|Im G_ii| (log)" if self.log_im else "Im G_ii"
            bg.fill_text(ylab, x0, y0 - 8)
            # title is per-frame (pool means) and lives on the foreground
            # legend
            bg.font = "11px sans-serif"
            bg.text_align = "left"
            for i, (c, lab) in enumerate([(C_G1, "g1 pool"),
                                          (C_G2, "g2 pool")]):
                bg.fill_style = c
                bg.fill_circle(x0 + 12, y1 - 28 + 14 * i, 3)
                bg.fill_style = "#52514e"
                bg.fill_text(lab, x0 + 20, y1 - 24 + 14 * i)

    def _update_artists(self):
        from ipycanvas import hold_canvas
        now = time.perf_counter()
        if self._last_frame is not None:
            inst = 1.0 / max(now - self._last_frame, 1e-9)
            self._fps = 0.9 * self._fps + 0.1 * inst if self._fps else inst
        self._last_frame = now
        fg = self.canvas[1]
        with hold_canvas(fg):
            fg.clear()
            if self.trails_on:
                fg.stroke_style = C_TRAIL
                fg.global_alpha = 0.35
                fg.line_width = 1
                for t in self._trails:
                    if len(t) >= 2:
                        arr = np.array(t)
                        xp, yp = self._xy_pix(arr)
                        fg.stroke_lines(np.column_stack([xp, yp]))
            fg.global_alpha = 0.75
            for gv, color in ((self._g1v[0, 0], C_G1),
                              (self._g2v[0, 0], C_G2)):
                xp, yp = self._xy_pix(gv)
                fg.fill_style = color
                fg.fill_circles(xp, yp, np.full(len(xp), 2.5))
            fg.global_alpha = 1.0
            fg.fill_style = "#52514e"
            fg.font = "11px monospace"
            fg.text_align = "right"
            fg.fill_text(f"sweep {self.sweeps}  {self._fps:5.1f} fps",
                         self.W - self.MR - 6, self.H - self.MB - 8)
            # title: the running pool means (mean log10 |Im G| is the
            # growth-rate observable; exact zeros pin at the 1e-300 clip)
            im_all = np.concatenate(
                [self._g1v[0, 0].imag, self._g2v[0, 0].imag]
            ).astype(np.float64)
            mean_log = np.log10(np.clip(np.abs(im_all), 1e-300, None)).mean()
            fg.fill_style = "#0b0b0b"
            fg.font = "12px sans-serif"
            fg.text_align = "center"
            fg.fill_text(
                f"mean Im G = {im_all.mean():.3g}    "
                f"mean log10 |Im G| = {mean_log:.2f}",
                self.W / 2, 16)

    def _render(self):
        # _update_artists already pushed the draw commands; here just cap the
        # outgoing frame rate so the widget comm channel is not flooded
        if self._last_frame is not None and self.max_fps:
            remaining = (1.0 / self.max_fps
                         - (time.perf_counter() - self._last_frame))
            if remaining > 0:
                time.sleep(remaining)

    # -- thread loop ---------------------------------------------------------
    def _loop(self):
        while not self._stop:
            if self._pending:
                self._apply_pending()
                if self._paused:
                    self._update_artists()
            if self._paused:
                time.sleep(0.05)
                continue
            t0 = time.perf_counter()
            self.step()
            self._t_run += time.perf_counter() - t0
            self._render()
            if self.pace:
                time.sleep(self.pace)

    def start(self):
        import threading
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        already_stopped = self._stop
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._t_run > 0 and not already_stopped:
            print(f"[live] {self.sweeps} sweeps in {self._t_run:.2f} s compute "
                  f"({self.sweeps / self._t_run:.0f} sweeps/s)")

    # -- widgets -------------------------------------------------------------
    def _make_controls(self):
        """Pause/Continue and randomise buttons + one widget per knob, wired
        to configure(). Stopping is programmatic only (live.stop())."""
        import ipywidgets as W

        btn = W.Button(description="continue" if self._paused else "pause")

        def _toggle(b):
            if self._paused:
                self.resume()
                b.description = "pause"
            else:
                self.pause()
                b.description = "continue"

        btn.on_click(_toggle)
        rand_btn = W.Button(description="randomise")
        rand_btn.on_click(lambda b: self.randomise())
        phi_w = W.Dropdown(
            options=["tanh", "linear"], value=self.phi, description="phi",
            style={"description_width": "30px"},
            layout=W.Layout(width="140px"))
        phi_w.observe(lambda ch: self.configure(phi=ch["new"]),
                      names="value")
        dtype_w = W.Dropdown(
            options=["complex128", "complex64"],
            value=("complex128" if self.g1.dtype == torch.cdouble
                   else "complex64"),
            description="dtype", style={"description_width": "45px"},
            layout=W.Layout(width="180px"))
        dtype_w.observe(lambda ch: self.configure(dtype=ch["new"]),
                        names="value")

        st = {"description_width": "55px"}
        alpha_w = W.FloatSlider(value=self.alpha, min=1.0, max=2.0, step=0.05,
                                description="alpha", continuous_update=False,
                                readout_format=".2f", style=st)
        sig_w = W.FloatSlider(value=self.sigma_W, min=0.1, max=3.0, step=0.05,
                              description="sigma_W", continuous_update=False,
                              readout_format=".2f", style=st)
        s_w = W.FloatLogSlider(value=self.s, base=10, min=-2.0, max=2.3,
                               step=0.05, description="s",
                               continuous_update=False,
                               readout_format=".3g", style=st)
        pop_w = W.Dropdown(
            options=sorted({64, 128, 256, 512, 1024, 2048, 4096, 8192,
                            self.pop}),
            value=self.pop, description="pop", style=st)
        eta_w = W.FloatText(value=self.eta, description="eta", step=0.001,
                            style=st)
        logim_w = W.Checkbox(value=self.log_im, description="log |Im|",
                             style=st)
        lay = W.Layout(width="150px")
        lim_ws = {k: W.FloatText(value=getattr(self, k), description=k,
                                 style=st, layout=lay)
                  for k in ("xmin", "xmax", "ymin", "ymax")}
        self._limit_widgets = lim_ws
        spf_w = W.IntText(
            value=self.steps_per_frame, description="steps/frame",
            style={"description_width": "80px"},
            layout=W.Layout(width="180px"))
        trails_w = W.Checkbox(value=self.trails_on, description="trails",
                              style={"description_width": "45px"})
        for w, key in ([(alpha_w, "alpha"), (sig_w, "sigma_W"), (s_w, "s"),
                        (pop_w, "pop"), (eta_w, "eta"),
                        (logim_w, "log_im"), (spf_w, "steps_per_frame"),
                        (trails_w, "trails")]
                       + [(w, k) for k, w in lim_ws.items()]):
            w.observe(lambda ch, k=key: self.configure(**{k: ch["new"]}),
                      names="value")
        note = W.Label("steps/frame: 0 = full sweep; set < pop to draw "
                       "sub-sweep frames (roll continuation)")

        return W.VBox([W.HBox([btn, rand_btn, phi_w, dtype_w]),
                       W.HBox([alpha_w, sig_w, s_w]),
                       W.HBox([pop_w, eta_w, logim_w]),
                       W.HBox(list(lim_ws.values())),
                       W.HBox([spf_w, trails_w, note])])

    def show(self):
        """Display the canvas + knob widgets, draw the initial pool, and
        start the loop (paused by default -- press continue)."""
        from IPython.display import display

        display(self.canvas)
        display(self._make_controls())
        self._update_artists()
        self.start()
