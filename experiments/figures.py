"""Figures for the generated reports, drawn from the saved compact metrics.

Rendering, not computation: these live in the publishing layer so a change to a
label cannot invalidate a run's checkpoints, and so the numbers in a figure come
from the same exported CSVs as the tables. Captions describe measurements and
methods, not model-selection conclusions or significance verdicts.

Plain SVG with computed geometry, no plotting dependency. Colours are chosen to
carry on both a light and a dark page, because GitHub's theme is not the same
setting as the reader's `prefers-color-scheme` and a figure that guesses wrong
is unreadable rather than merely ugly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INK = "#6e7781"  # readable on white and on #0d1117
GRID = "#8b949e"
UNDER = "#1f9c8a"  # forecasts below outcomes
OVER = "#c4842a"  # forecasts above outcomes
FONT = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def _scale(lo, hi, out_lo, out_hi):
    span = (hi - lo) or 1.0
    return lambda v: out_lo + (v - lo) * (out_hi - out_lo) / span


def _ticks(lo, hi, step):
    """Gridline values strictly inside (lo, hi): a tick on the axis edge is noise."""
    first = np.ceil(lo / step) * step
    values = np.arange(first, hi + step / 1000, step)
    return [v for v in values if lo + step / 1000 < v < hi - step / 1000]


def _text(x, y, s, size=11, anchor="middle", fill=INK, weight="normal"):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
        f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{s}</text>'
    )


def _frame(width, height, title, subtitle):
    """Subtitle may be a string or several lines; monospace at 11px overflows near 105 chars."""
    lines = [subtitle] if isinstance(subtitle, str) else list(subtitle)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{title}">',
        _text(16, 22, title, size=13, anchor="start", fill=INK, weight="bold"),
    ]
    for i, line in enumerate(lines):
        out.append(_text(16, 40 + 15 * i, line, size=11, anchor="start"))
    return out


def _bin_label(raw):
    """`(0.05, 0.1]` -> `.05-.1`; the interval strings are unreadable on an axis."""
    lo, hi = raw.strip("([]").split(",")
    trim = lambda v: ("0" if float(v) == 0 else f"{float(v):g}").lstrip("0") or "0"  # noqa: E731
    if "inf" in hi:
        return f">{trim(lo)}"
    return f"≤{trim(hi)}" if float(lo) < 0 else f"{trim(lo)}-{trim(hi)}"


def reliability(data, baseline):
    """Ratio of summed forecasts to summed outcomes within each forecast bin."""
    rows = data["reliability"]
    rows = rows[rows.model.eq(baseline)]
    pooled = rows.groupby("bin", sort=False).apply(
        lambda g: pd.Series(
            {
                "n": g.n.sum(),
                "proj": (g.proj * g.n).sum() / g.n.sum(),
                "actual": (g.actual * g.n).sum() / g.n.sum(),
            }
        ),
        include_groups=False,
    )
    # A bin whose candidates scored nothing has no ratio at all. Dividing anyway gave
    # it an infinite one, which became the axis bound: the 2025 `vegas-environment`
    # bin of one candidate at 0.0483 and no touchdowns took the tick loop to
    # `Maximum allowed size exceeded`, in 17 of the 150 season/model combinations.
    pooled["ratio"] = np.where(pooled.actual > 0, pooled.proj / pooled.actual, np.nan)
    pooled = pooled.reset_index()
    defined = np.isfinite(pooled.ratio)

    w, h = 720, 360
    left, right, top, bottom = 62, 24, 66, 62
    ratios = pooled.ratio[defined]
    lo = min(0.55, ratios.min() - 0.05) if len(ratios) else 0.55
    hi = max(1.35, ratios.max() + 0.05) if len(ratios) else 1.35
    y = _scale(lo, hi, h - bottom, top)
    step = (w - left - right) / len(pooled)
    xs = [left + step * (i + 0.5) for i in range(len(pooled))]

    subtitle = [
        f"`{baseline}`, all seasons pooled. Below 1: under-projected. Above 1: over-projected."
    ]
    if not defined.all():
        subtitle.append("Bins whose candidates scored no touchdowns have no ratio, and are marked.")
    out = _frame(w, h, "Forecast / outcome by projection bin", subtitle)
    for tick in np.arange(0.6, hi + 0.001, 0.2):
        out.append(
            f'<line x1="{left}" y1="{y(tick):.1f}" x2="{w - right}" y2="{y(tick):.1f}" '
            f'stroke="{GRID}" stroke-opacity="0.25" stroke-width="1"/>'
        )
        out.append(_text(left - 10, y(tick) + 4, f"{tick:.1f}", anchor="end"))
    out.append(
        f'<line x1="{left}" y1="{y(1.0):.1f}" x2="{w - right}" y2="{y(1.0):.1f}" '
        f'stroke="{GRID}" stroke-width="1.5" stroke-dasharray="5 4"/>'
    )
    out.append(_text(w - right - 4, y(1.0) - 8, "forecast = outcome", size=10, anchor="end"))

    # One polyline per run of defined bins: a segment drawn across an undefined one
    # would imply a value the bin does not have.
    run = []
    for x, ratio in zip(xs, pooled.ratio, strict=True):
        if np.isfinite(ratio):
            run.append(f"{x:.1f},{y(ratio):.1f}")
            continue
        if len(run) > 1:
            out.append(
                f'<polyline points="{" ".join(run)}" fill="none" stroke="{INK}" '
                'stroke-width="1.5"/>'
            )
        run = []
    if len(run) > 1:
        out.append(
            f'<polyline points="{" ".join(run)}" fill="none" stroke="{INK}" stroke-width="1.5"/>'
        )
    for x, row in zip(xs, pooled.itertuples(), strict=True):
        if np.isfinite(row.ratio):
            colour = OVER if row.ratio > 1 else UNDER
            out.append(f'<circle cx="{x:.1f}" cy="{y(row.ratio):.1f}" r="4.5" fill="{colour}"/>')
            offset = 18 if abs(row.ratio - 1.0) < 0.09 else -12
            out.append(_text(x, y(row.ratio) + offset, f"{row.ratio:.2f}", size=10, fill=colour))
        else:
            out.append(_text(x, (top + h - bottom) / 2, "no TDs", size=10, fill=GRID))
        out.append(_text(x, h - bottom + 18, _bin_label(row.bin), size=10))
        out.append(_text(x, h - bottom + 34, f"{int(row.n):,}", size=9))
    out.append(_text(left, h - 12, "projection bin (lambda) / candidates", size=10, anchor="start"))
    out.append("</svg>")
    return "\n".join(out)


def slope_by_season(data, baseline):
    """Saved calibration slopes and their normal-approximation intervals by season."""
    cal = data["calibration"]
    cal = cal[cal.model.eq(baseline)].sort_values("season")
    # An unsupported fit has no coefficient to plot; a supported fit whose cluster SEs
    # were suppressed has a coefficient but no interval. Both are stated in the caption
    # rather than drawn: NaN geometry would emit literal `nan` coordinates into the SVG.
    unsupported = int(cal.slope.isna().sum())
    cal = cal[cal.slope.notna()]
    if cal.empty:
        out = _frame(
            720,
            160,
            "Poisson calibration slope by season",
            "No season produced a supported calibration fit; there is nothing to plot.",
        )
        out.append("</svg>")
        return "\n".join(out)
    spread = cal.se_slope.fillna(0.0)
    no_interval = int(cal.se_slope.isna().sum())
    # Both warnings on one line ran about 77px past the 720px canvas, so the reader lost
    # the end of whichever notice came second. They get their own line, and the plot
    # starts lower to make room rather than being drawn over.
    notes = []
    if unsupported:
        notes.append(f"{unsupported} unsupported fit(s) omitted")
    if no_interval:
        notes.append(f"{no_interval} plotted without cluster SEs")
    w, h = 720, 320 + (15 if notes else 0)
    left, right, bottom = 62, 24, 44
    top = 66 + (15 if notes else 0)
    # Both bounds follow the plotted intervals. A hardcoded top clipped every
    # supported baseline above it: `vegas-environment` reaches a slope of 1.137
    # and an upper bound of 1.234, outside a viewport that stopped at 1.03.
    lo = min(0.75, (cal.slope - 1.96 * spread).min() - 0.02)
    hi = max(1.03, (cal.slope + 1.96 * spread).max() + 0.02)
    y = _scale(lo, hi, h - bottom, top)
    step = (w - left - right) / len(cal)
    xs = [left + step * (i + 0.5) for i in range(len(cal))]
    mean = cal.slope.mean()

    out = _frame(
        w,
        h,
        "Poisson calibration slope by season",
        [
            f"`{baseline}`; estimate +/-1.96 SE, player-season clusters.",
            f"Dashed: slope 1 reference. Dotted: mean {cal.slope.mean():.3f}. Intercepts in table.",
            *(["; ".join(notes) + "."] if notes else []),
        ],
    )
    for tick in _ticks(lo, hi, 0.05):
        out.append(
            f'<line x1="{left}" y1="{y(tick):.1f}" x2="{w - right}" y2="{y(tick):.1f}" '
            f'stroke="{GRID}" stroke-opacity="0.25" stroke-width="1"/>'
        )
        out.append(_text(left - 10, y(tick) + 4, f"{tick:.2f}", anchor="end"))
    out.append(
        f'<line x1="{left}" y1="{y(1.0):.1f}" x2="{w - right}" y2="{y(1.0):.1f}" '
        f'stroke="{GRID}" stroke-width="1.5" stroke-dasharray="5 4"/>'
    )
    out.append(
        f'<line x1="{left}" y1="{y(mean):.1f}" x2="{w - right}" y2="{y(mean):.1f}" '
        f'stroke="{UNDER}" stroke-width="1" stroke-dasharray="2 3"/>'
    )
    for x, row in zip(xs, cal.itertuples(), strict=True):
        if pd.notna(row.se_slope):
            top_y = y(row.slope + 1.96 * row.se_slope)
            bot_y = y(row.slope - 1.96 * row.se_slope)
            out.append(
                f'<line x1="{x:.1f}" y1="{top_y:.1f}" x2="{x:.1f}" y2="{bot_y:.1f}" '
                f'stroke="{UNDER}" stroke-width="1.5" stroke-opacity="0.55"/>'
            )
        out.append(f'<circle cx="{x:.1f}" cy="{y(row.slope):.1f}" r="3.5" fill="{UNDER}"/>')
        out.append(_text(x, h - bottom + 18, str(int(row.season))[2:], size=10))
    out.append(_text(left, h - 12, "season", size=10, anchor="start"))
    out.append("</svg>")
    return "\n".join(out)


def bakeoff(data, baseline):
    """Paired season deltas with their own standard errors.

    The comparison set is whatever the run configured against the baseline, down to
    `random` alone or to nothing. No shared detection threshold or significance
    classification is inferred from these measurements.
    """
    wide = data["replays"]
    wide = wide[wide.strategy.eq("greedy")].pivot_table(
        index="season", columns="model", values="total", aggfunc="mean"
    )
    delta = wide.sub(wide[baseline], axis=0).drop(columns=[baseline])
    # Omit the shuffled comparison when others are present; report its mean in the caption.
    aside = delta.pop("random") if "random" in delta and delta.shape[1] > 1 else None
    if delta.columns.empty:
        out = _frame(
            720,
            160,
            f"Season score vs `{baseline}`, greedy replay",
            "No challenger was configured against the baseline; there is nothing to compare.",
        )
        out.append("</svg>")
        return "\n".join(out)
    stats = pd.DataFrame(
        {
            "mean": delta.mean(),
            "se": delta.std(ddof=1) / np.sqrt(len(delta)) if len(delta) > 1 else np.nan,
            "won": (delta > 0).sum(),
        }
    ).sort_values("mean")
    spread = stats.se.fillna(0.0)

    row_h = 26
    w = 720
    left, right, top, bottom = 168, 132, 92, 52
    h = top + row_h * len(stats) + bottom
    # Zero is always drawn as the reference line, so it is always in the domain: a
    # single season whose comparisons all lose put that line at x=819.9 on a 720 canvas.
    lo = min(0.0, (stats["mean"] - spread).min()) - 1.0
    hi = max(0.0, (stats["mean"] + spread).max()) + 1.0
    x = _scale(lo, hi, left, w - right)

    note = f" `random` omitted at {aside.mean():+.1f}." if aside is not None else ""
    seasons = f"{len(delta)} season" + ("s" if len(delta) != 1 else "")
    measured = "Mean and standard error over" if len(delta) > 1 else "Season score over"
    subtitle = [f"{measured} {seasons}; positive season deltas at right.{note}"]
    out = _frame(w, h, f"Season score vs `{baseline}`, greedy replay", subtitle)
    out.append(
        f'<line x1="{x(0):.1f}" y1="{top - 10:.1f}" x2="{x(0):.1f}" '
        f'y2="{top + row_h * len(stats) + 2:.1f}" stroke="{GRID}" stroke-width="1.5"/>'
    )
    for i, row in enumerate(stats.itertuples()):
        cy = top + row_h * i + 8
        if np.isfinite(row.se):
            lo_x, hi_x = x(row.mean - row.se), x(row.mean + row.se)
            out.append(
                f'<line x1="{lo_x:.1f}" y1="{cy:.1f}" x2="{hi_x:.1f}" y2="{cy:.1f}" '
                f'stroke="{INK}" stroke-width="1.5" stroke-opacity="0.6"/>'
            )
        out.append(f'<circle cx="{x(row.mean):.1f}" cy="{cy:.1f}" r="4" fill="{INK}"/>')
        out.append(_text(left - 12, cy + 4, row.Index, size=11, anchor="end"))
        readout = (
            f"{row.mean:+.2f}\u00b1{row.se:.2f}" if np.isfinite(row.se) else f"{row.mean:+.2f}"
        )
        out.append(
            _text(
                w - right + 8,
                cy + 4,
                f"{readout} {row.won}/{len(delta)}",
                size=10,
                anchor="start",
            )
        )
    base = top + row_h * len(stats) + 20
    for tick in range(int(np.ceil(lo / 4)) * 4, int(hi) + 1, 4):
        out.append(_text(x(tick), base, str(tick), size=10))
    out.append(
        _text(
            left,
            h - 14,
            "TDs per season vs baseline (mean, SE, positive season deltas)",
            size=10,
            anchor="start",
        )
    )
    out.append("</svg>")
    return "\n".join(out)


def optimizer_by_season(data, baseline):
    """Actual optimizer-minus-greedy scores by season, with a mean and SE."""
    rows = data["replays"]
    rows = rows[rows.model.eq(baseline) & rows.strategy.isin(["greedy", "optimizer"])]
    piv = rows.pivot_table(index="season", columns="strategy", values="total", aggfunc="mean")
    diff = (piv.optimizer - piv.greedy).sort_index()
    mean = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else np.nan

    w, h = 720, 330
    left, right, top, bottom = 56, 24, 74, 52
    span = max(abs(diff.min()), abs(diff.max())) + 2
    y = _scale(-span, span, h - bottom, top)
    step = (w - left - right) / len(diff)

    out = _frame(
        w,
        h,
        "Rolling assignment minus greedy, by season",
        [
            f"`{baseline}`. Dashed: mean {mean:+.2f}"
            + (f" ({se:.2f} SE)" if np.isfinite(se) else "")
            + f", positive in {int((diff > 0).sum())} of {len(diff)}.",
            "Bars: optimizer minus greedy in actual TDs per season.",
        ],
    )
    for tick in range(-int(span // 5) * 5, int(span) + 1, 5):
        out.append(
            f'<line x1="{left}" y1="{y(tick):.1f}" x2="{w - right}" y2="{y(tick):.1f}" '
            f'stroke="{GRID}" stroke-opacity="0.25" stroke-width="1"/>'
        )
        out.append(_text(left - 10, y(tick) + 4, f"{tick:+d}", anchor="end"))
    for i, (season, value) in enumerate(diff.items()):
        cx = left + step * (i + 0.5)
        colour = UNDER if value > 0 else OVER
        y0, y1 = sorted((y(0), y(value)))
        out.append(
            f'<rect x="{cx - step * 0.3:.1f}" y="{y0:.1f}" width="{step * 0.6:.1f}" '
            f'height="{max(y1 - y0, 0.5):.1f}" fill="{colour}" fill-opacity="0.75"/>'
        )
        out.append(_text(cx, h - bottom + 18, str(int(season))[2:], size=10))
    out.append(
        f'<line x1="{left}" y1="{y(0):.1f}" x2="{w - right}" y2="{y(0):.1f}" '
        f'stroke="{GRID}" stroke-width="1.5"/>'
    )
    out.append(
        f'<line x1="{left}" y1="{y(mean):.1f}" x2="{w - right}" y2="{y(mean):.1f}" '
        f'stroke="{INK}" stroke-width="1" stroke-dasharray="4 4"/>'
    )
    out.append(_text(left, h - 14, "season", size=10, anchor="start"))
    out.append("</svg>")
    return "\n".join(out)


def render(data, baseline):
    """Every figure, keyed by the file stem it is written to."""
    return {
        "reliability": reliability(data, baseline),
        "calibration-slope": slope_by_season(data, baseline),
        "bakeoff": bakeoff(data, baseline),
        "optimizer-by-season": optimizer_by_season(data, baseline),
    }
