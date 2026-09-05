"""Figures for the generated reports, drawn from the saved compact metrics.

Rendering, not computation: these live in the publishing layer so a change to a
label cannot invalidate a run's checkpoints, and so the numbers in a figure come
from the same exported CSVs the tables and findings are read from.

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
    """Projected/actual by forecast bin: the shape a pooled ratio of 1.0 hides."""
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
    pooled["ratio"] = pooled.proj / pooled.actual
    pooled = pooled.reset_index()

    w, h = 720, 360
    left, right, top, bottom = 62, 24, 66, 62
    lo = min(0.55, pooled.ratio.min() - 0.05)
    hi = max(1.35, pooled.ratio.max() + 0.05)
    y = _scale(lo, hi, h - bottom, top)
    step = (w - left - right) / len(pooled)
    xs = [left + step * (i + 0.5) for i in range(len(pooled))]

    out = _frame(
        w,
        h,
        "Forecast / outcome by projection bin",
        f"`{baseline}`, all seasons pooled. Below 1: under-projected. Above 1: over-projected.",
    )
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
    out.append(_text(w - right - 4, y(1.0) - 8, "calibrated", size=10, anchor="end"))

    # One polyline, then points coloured by which side of 1.0 they fall.
    path = " ".join(f"{x:.1f},{y(r):.1f}" for x, r in zip(xs, pooled.ratio, strict=True))
    out.append(f'<polyline points="{path}" fill="none" stroke="{INK}" stroke-width="1.5"/>')
    for x, row in zip(xs, pooled.itertuples(), strict=True):
        colour = OVER if row.ratio > 1 else UNDER
        out.append(f'<circle cx="{x:.1f}" cy="{y(row.ratio):.1f}" r="4.5" fill="{colour}"/>')
        offset = 18 if abs(row.ratio - 1.0) < 0.09 else -12
        out.append(_text(x, y(row.ratio) + offset, f"{row.ratio:.2f}", size=10, fill=colour))
        out.append(_text(x, h - bottom + 18, _bin_label(row.bin), size=10))
        out.append(_text(x, h - bottom + 34, f"{int(row.n):,}", size=9))
    out.append(_text(left, h - 12, "projection bin (lambda) / candidates", size=10, anchor="start"))
    out.append("</svg>")
    return "\n".join(out)


def slope_by_season(data, baseline):
    """Calibration slope per season: the point is how little it moves."""
    cal = data["calibration"]
    cal = cal[cal.model.eq(baseline)].sort_values("season")
    w, h = 720, 320
    left, right, top, bottom = 62, 24, 66, 44
    lo = min(0.75, (cal.slope - 1.96 * cal.se_slope).min() - 0.02)
    y = _scale(lo, 1.03, h - bottom, top)
    step = (w - left - right) / len(cal)
    xs = [left + step * (i + 0.5) for i in range(len(cal))]
    mean = cal.slope.mean()

    out = _frame(
        w,
        h,
        "Poisson calibration slope by season",
        [
            f"`{baseline}`, with 95% cluster-robust intervals. 1.0 is calibrated;",
            f"below 1 the forecasts are too extreme. Dotted: mean {cal.slope.mean():.3f}.",
        ],
    )
    for tick in np.arange(0.8, 1.001, 0.05):
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
        top_y, bot_y = y(row.slope + 1.96 * row.se_slope), y(row.slope - 1.96 * row.se_slope)
        out.append(
            f'<line x1="{x:.1f}" y1="{top_y:.1f}" x2="{x:.1f}" y2="{bot_y:.1f}" '
            f'stroke="{UNDER}" stroke-width="1.5" stroke-opacity="0.55"/>'
        )
        out.append(f'<circle cx="{x:.1f}" cy="{y(row.slope):.1f}" r="3.5" fill="{UNDER}"/>')
        out.append(_text(x, h - bottom + 18, str(int(row.season))[2:], size=10))
    out.append(_text(left, h - 12, "season", size=10, anchor="start"))
    out.append("</svg>")
    return "\n".join(out)


def bakeoff(data, baseline, floor):
    """Paired season deltas with their standard errors, against the resolution floor."""
    wide = data["replays"]
    wide = wide[wide.strategy.eq("greedy")].pivot_table(
        index="season", columns="model", values="total", aggfunc="mean"
    )
    delta = wide.sub(wide[baseline], axis=0).drop(columns=[baseline])
    # `random` sits an order of magnitude out and would squash every real comparison.
    aside = delta.pop("random") if "random" in delta else None
    stats = pd.DataFrame(
        {
            "mean": delta.mean(),
            "se": delta.std(ddof=1) / np.sqrt(len(delta)),
            "won": (delta > 0).sum(),
        }
    ).sort_values("mean")

    row_h = 26
    w = 720
    left, right, top, bottom = 168, 132, 92, 52
    h = top + row_h * len(stats) + bottom
    lo = min(-floor, (stats["mean"] - stats.se).min()) - 1.0
    hi = max(floor, (stats["mean"] + stats.se).max()) + 1.0
    x = _scale(lo, hi, left, w - right)

    note = f" `random` omitted at {aside.mean():+.1f}." if aside is not None else ""
    out = _frame(
        w,
        h,
        f"Season score vs `{baseline}`, greedy replay",
        [
            f"Mean and standard error over {len(delta)} seasons; seasons better at right.",
            f"Shaded: the \u00b1{floor:.1f} TD band these replays cannot resolve.{note}",
        ],
    )
    out.append(
        f'<rect x="{x(-floor):.1f}" y="{top - 10:.1f}" width="{x(floor) - x(-floor):.1f}" '
        f'height="{row_h * len(stats) + 12:.1f}" fill="{GRID}" fill-opacity="0.12"/>'
    )
    out.append(
        f'<line x1="{x(0):.1f}" y1="{top - 10:.1f}" x2="{x(0):.1f}" '
        f'y2="{top + row_h * len(stats) + 2:.1f}" stroke="{GRID}" stroke-width="1.5"/>'
    )
    for i, row in enumerate(stats.itertuples()):
        cy = top + row_h * i + 8
        lo_x, hi_x = x(row.mean - row.se), x(row.mean + row.se)
        colour = OVER if row.mean < -floor else INK
        out.append(
            f'<line x1="{lo_x:.1f}" y1="{cy:.1f}" x2="{hi_x:.1f}" y2="{cy:.1f}" '
            f'stroke="{colour}" stroke-width="1.5" stroke-opacity="0.6"/>'
        )
        out.append(f'<circle cx="{x(row.mean):.1f}" cy="{cy:.1f}" r="4" fill="{colour}"/>')
        out.append(_text(left - 12, cy + 4, row.Index, size=11, anchor="end"))
        out.append(
            _text(
                w - right + 8,
                cy + 4,
                f"{row.mean:+.2f}\u00b1{row.se:.2f} {row.won}/{len(delta)}",
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
            "TDs per season vs baseline (mean, SE, seasons better)",
            size=10,
            anchor="start",
        )
    )
    out.append("</svg>")
    return "\n".join(out)


def optimizer_by_season(data, baseline):
    """Per-season optimizer minus greedy: the scatter behind a mean worth doubting."""
    rows = data["replays"]
    rows = rows[rows.model.eq(baseline) & rows.strategy.isin(["greedy", "optimizer"])]
    piv = rows.pivot_table(index="season", columns="strategy", values="total", aggfunc="mean")
    diff = (piv.optimizer - piv.greedy).sort_index()
    mean = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(len(diff))

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
            f"`{baseline}`. Dashed: mean {mean:+.2f} ({se:.2f} SE), better in "
            f"{int((diff > 0).sum())} of {len(diff)}.",
            "The spread, not the mean, is the finding.",
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


def render(data, baseline, floor):
    """Every figure, keyed by the file stem it is written to."""
    return {
        "reliability": reliability(data, baseline),
        "calibration-slope": slope_by_season(data, baseline),
        "bakeoff": bakeoff(data, baseline, floor),
        "optimizer-by-season": optimizer_by_season(data, baseline),
    }
