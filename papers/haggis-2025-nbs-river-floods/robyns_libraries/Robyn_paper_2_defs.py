from __future__ import annotations
import numpy as np, geopandas as gpd, matplotlib as mpl, matplotlib.pyplot as plt, rasterio
from rasterio.plot import plotting_extent
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, NullLocator, MaxNLocator
from pathlib import Path
from matplotlib import font_manager as fm
from matplotlib.patches import Polygon, Rectangle, RegularPolygon
from matplotlib.colors import Normalize
from rasterio.warp import calculate_default_transform, reproject, Resampling

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


PREF = "Arial" if any(f.name == "Arial" for f in fm.fontManager.ttflist) else "Helvetica"

NATURE_RC = {
    # Output
    "figure.dpi": 300,         # on-screen
    "savefig.dpi": 600,        # PNG export (RGB)
    "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",  # editable text

    # Typeface & sizes (5–7 pt at 90 mm)
    "font.family": PREF,
    "font.sans-serif": [PREF, "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 6.5,
    "axes.titlesize": 7,
    "axes.labelsize": 6.5,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,

    # Greek letters to match sans (no LaTeX needed)
    "mathtext.fontset": "dejavusans",  # consistent Greek
    "axes.linewidth": 0.35,
}




def add_scale_bar(ax, length_km=20, location=(0.9, 0.8), linewidth=2, tick_height=0.01, label_offset=0.02, km_offset=0.03):
    x, y = location  # Adjusted location (higher y value to move the scale bar upward)
    bar_half_length = 0.05  # Half the scale bar length in axes fraction

    # Draw the scale bar
    ax.plot(
        [x - bar_half_length, x + bar_half_length], [y, y],  # Scale bar endpoints
        transform=ax.transAxes, color='black', linewidth=linewidth
    )

    # Draw perpendicular tick marks
    tick_positions = [x - bar_half_length, x, x + bar_half_length]
    for pos in tick_positions:
        ax.plot(
            [pos, pos], [y - tick_height / 2, y + tick_height / 2],  # Vertical line for ticks
            transform=ax.transAxes, color='black', linewidth=linewidth
        )

    # Add numeric labels below the tick marks
    ax.text(
        x - bar_half_length, y - tick_height - label_offset, "0", transform=ax.transAxes, 
        ha='center', va='center', fontsize=10
    )
    ax.text(
        x, y - tick_height - label_offset, f"{length_km // 2}", transform=ax.transAxes, 
        ha='center', va='center', fontsize=10
    )
    ax.text(
        x + bar_half_length, y - tick_height - label_offset, f"{length_km}", transform=ax.transAxes, 
        ha='center', va='center', fontsize=10
    )

    # Add "km" label slightly to the right of the scale bar
    ax.text(
        x + bar_half_length + km_offset, y, "km", transform=ax.transAxes, 
        ha='left', va='center', fontsize=12
    )

def add_north_arrow(ax, location=(0.9, 0.8), size=0.05, fontsize=12, label_offset=0.03):
    """
    Add a north arrow to the plot, with "N" positioned slightly above the arrow.
    """
    x, y = location

    # Draw the arrow
    ax.annotate(
        '', xy=(x, y + size), xycoords='axes fraction',
        xytext=(x, y), textcoords='axes fraction',
        arrowprops=dict(facecolor='black', edgecolor='black', headwidth=10, headlength=15, width=5)
    )

    # Add the "N" label slightly above the arrow
    ax.text(
        x, y + size + label_offset, "N", transform=ax.transAxes,
        fontsize=fontsize, fontweight="bold", ha="center", va="center", color="black"
    )



def draw_scale_bar(ax, length_km=20, location=(0.9, 0.8), linewidth=1, tick_height=0.01, label_offset=0.02, km_offset=0.01):
    x, y = location  # Adjusted location (higher y value to move the scale bar upward)
    bar_half_length = 0.05  # Half the scale bar length in axes fraction

    # Draw the scale bar
    ax.plot(
        [x - bar_half_length, x + bar_half_length], [y, y],  # Scale bar endpoints
        transform=ax.transAxes, color='black', linewidth=linewidth
    )

    # Draw perpendicular tick marks
    tick_positions = [x - bar_half_length, x, x + bar_half_length]
    for pos in tick_positions:
        ax.plot(
            [pos, pos], [y - tick_height / 2, y + tick_height / 2],  # Vertical line for ticks
            transform=ax.transAxes, color='black', linewidth=linewidth
        )

    # Add numeric labels below the tick marks
    ax.text(
        x - bar_half_length, y - tick_height - label_offset, "0", transform=ax.transAxes, 
        ha='center', va='center', fontsize=6
    )
    ax.text(
        x, y - tick_height - label_offset, f"{length_km // 2}", transform=ax.transAxes, 
        ha='center', va='center', fontsize=6
    )
    ax.text(
        x + bar_half_length, y - tick_height - label_offset, f"{length_km}", transform=ax.transAxes, 
        ha='center', va='center', fontsize=6
    )

    # Add "km" label slightly to the right of the scale bar
    ax.text(
        x + bar_half_length + km_offset, y, "km", transform=ax.transAxes, 
        ha='left', va='center', fontsize=7
    )

def draw_north_arrow(ax, location=(0.9, 0.87), size=0.05, fontsize=7, label_offset=0.03):
    """
    Add a north arrow to the plot, with "N" positioned slightly above the arrow.
    """
    x, y = location

    # Draw the arrow
    ax.annotate(
        '', xy=(x, y + size), xycoords='axes fraction',
        xytext=(x, y), textcoords='axes fraction',
        arrowprops=dict(facecolor='black', edgecolor='black', headwidth=6, headlength=6, width=2.5)
    )

    # Add the "N" label slightly above the arrow
    ax.text(
        x, y + size + label_offset, "N", transform=ax.transAxes,
        fontsize=fontsize, fontweight="bold", ha="center", va="center", color="black"
    )



# TRUE scale bar in map units (needs projected CRS in meters)
def add_scale_bar(ax, gdf, where="right-top", pad=0.04,
                   length_km="auto", max_frac=0.28,
                   lw=0.6, tick_h_frac=0.012,
                   fs_lab=6, fs_unit=6, unit_text="km"):
    crs = getattr(gdf, "crs", None)
    if crs is None or not crs.is_projected:
        return  # skip if not projected

    minx, miny, maxx, maxy = gdf.total_bounds
    W, H = (maxx - minx), (maxy - miny)

    # pleasant length that fits ≤ max_frac of map width
    if length_km == "auto":
        candidates = np.array([2, 5, 10, 20, 25, 50, 100], dtype=float)
        target = max_frac * (W/1000.0)
        valid = candidates[candidates <= max(1.0, target)]
        length_km = float(valid[-1]) if valid.size else 5.0
    L = length_km * 1000.0

    # anchor (x0,y0)
    x0 = minx + pad*W if "left"  in where else maxx - pad*W - L
    y0 = maxy - pad*H if "top"   in where else miny + pad*H

    # bar
    ax.plot([x0, x0+L], [y0, y0], color="black", lw=lw, clip_on=False)

    # ticks at 0, mid, end
    tick_h = tick_h_frac * H
    for xi in (x0, x0+L/2, x0+L):
        ax.plot([xi, xi], [y0 - tick_h/2, y0 + tick_h/2], color="black", lw=lw, clip_on=False)

    # numeric labels
    ylab = y0 - 2.1*tick_h
    ax.text(x0,     ylab, "0",                   ha="center", va="top", fontsize=fs_lab)
    ax.text(x0+L/2, ylab, f"{int(length_km//2)}",ha="center", va="top", fontsize=fs_lab)
    ax.text(x0+L,   ylab, f"{int(length_km)}",   ha="center", va="top", fontsize=fs_lab)
    ax.text(x0+L + 0.012*W, y0, unit_text, ha="left", va="center", fontsize=fs_unit)

    # return center-above point (DATA coords) for the arrow
    return (x0 + L/2, y0 + 2.2*tick_h)

def add_north_arrow_axes(
    ax, x_ax, y_ax, *,
    size_frac=0.05,      # overall height in axes coords
    gap_frac=0.035,      # vertical gap above the bar
    shaft_w_frac=0.18,   # shaft width (fraction of size)
    head_w_frac=0.65,    # head width (fraction of size)
    head_h_frac=0.70,    # head/shaft split
    color="black", fs=6, lw=0.6
):
    """Neat 'N' + upright arrow in AXES coords (0–1)."""
    y0 = y_ax + gap_frac
    shaft_h = size_frac * (1 - head_h_frac)
    head_h  = size_frac * head_h_frac
    shaft_w = size_frac * shaft_w_frac
    head_w  = size_frac * head_w_frac

    # shaft
    rect = Rectangle((x_ax - shaft_w/2, y0), shaft_w, shaft_h,
                     transform=ax.transAxes, facecolor=color, edgecolor=color,
                     linewidth=lw, zorder=15, clip_on=False)
    ax.add_patch(rect)

    # triangle head
    tip_y = y0 + shaft_h + head_h
    tri = Polygon([(x_ax, tip_y),
                   (x_ax - head_w/2, y0 + shaft_h),
                   (x_ax + head_w/2, y0 + shaft_h)],
                  closed=True, transform=ax.transAxes,
                  facecolor=color, edgecolor=color,
                  linewidth=lw, zorder=15, clip_on=False)
    ax.add_patch(tri)

    # "N"
    ax.text(x_ax, tip_y + size_frac*0.28, "N",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=fs, fontweight="bold", color=color)


def set_nature_style():
    """Apply a tiny Nature-ish rcParams preset, globally."""
    pref = "Arial" if any(f.name == "Arial" for f in fm.fontManager.ttflist) else "Helvetica"
    mpl.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",

        "font.family": "sans-serif",
        "font.sans-serif": [pref, "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 6.5,
        "axes.titlesize": 7,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,

        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 0.35,
    })
    return pref  # in case you want to print/use it