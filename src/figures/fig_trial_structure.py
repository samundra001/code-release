#!/usr/bin/env python3
"""
fig_trial_structure.py  -  Figure 1: trial structure (A) and stored data (B)

A  What the participant saw: one trial of a real List 1 item, syntactically
   invalid version (code 220; 5 items, critical item 2: "suit" in quotes),
   global variable -> fixation -> items one at a time -> judgment.
   Only the item screens are drawn to scale (700 ms each, 200 ms blank between
   items); the other screens are labeled with their durations.
B  The same items on a time axis relative to the critical item: which items have
   stored EEG epochs, and the window used in all analyses (-100 to +800 ms).

Timing (Kuo & Prat, 2024, Methods): global variable up to 15 s; fixation cross
1000 ms + 200 ms blank; each item 700 ms + 200 ms blank; Yes/No up to 30 s.

Needs no data. Output (current folder): fig_trial_structure.png / .pdf / .tiff
USAGE:  python fig_trial_structure.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

MONO = "DejaVu Sans Mono"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})

GREEN, GREEN_LIGHT = "#1e8449", "#d5efe0"
GRAY, GRAY_LIGHT, INK, MUTED = "#8a8f98", "#e6e8eb", "#1f2328", "#5b6270"

SOA, DUR = 900, 700                     # item onset interval, item duration (ms)
items = ["for", '"suit"', "in", "cards:", "print(suit)"]
crit = 1                                # index of the critical item
onset = {i: (i - crit) * SOA for i in range(len(items))}   # ms relative to critical

# Screens that are not items (x positions in ms on the same axis; widths drawn)
GLOBAL = (-3140, -1960)                 # not to scale (up to 15 s)
FIX = (-1780, -1100)                    # 1000 ms fixation (drawn narrower), 200 ms blank
RESP = (onset[4] + DUR + 200, onset[4] + DUR + 200 + 680)  # not to scale (up to 30 s)
XLIM = (GLOBAL[0] - 60, RESP[1] + 60)

fig, (axA, axB) = plt.subplots(2, 1, figsize=(7.1, 3.6), sharex=True,
                               gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.12})
for ax in (axA, axB):
    ax.set_xlim(*XLIM)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(left=False, labelleft=False)

# ---------------------------------------------------------------- Panel A
axA.set_ylim(-1.25, 1.65)
axA.tick_params(bottom=False)
H = 1.0                                  # screen height
y0 = -0.35


def screen(x0, x1, text, size=6.3, outline=None, lw=0.6):
    axA.add_patch(Rectangle((x0, y0), x1 - x0, H, fc="black",
                            ec=outline or "black", lw=lw, zorder=2))
    axA.text((x0 + x1) / 2, y0 + H / 2, text, color="white", ha="center", va="center",
             family=MONO, fontsize=size, zorder=3, linespacing=1.3)


screen(*GLOBAL, 'cards = ["spades",\n "clubs", "hearts",\n "diamonds"]', size=5.0)
screen(*FIX, "+", size=9)
for i, w in enumerate(items):
    crit_item = i == crit
    screen(onset[i], onset[i] + DUR, w, size=4.9 if len(w) > 8 else 6.3,
           outline=GREEN if crit_item else None, lw=2.0 if crit_item else 0.6)
screen(*RESP, "Yes / No", size=6.3)

lab_y = y0 - 0.18
axA.text(sum(GLOBAL) / 2, lab_y, "up to 15 s", ha="center", va="top", color=MUTED)
axA.text(sum(FIX) / 2, lab_y, "1000 ms", ha="center", va="top", color=MUTED)
axA.text(sum(RESP) / 2, lab_y, "up to 30 s", ha="center", va="top", color=MUTED)
# bracket under the items
bx0, bx1, by = onset[0], onset[4] + DUR, lab_y - 0.05
axA.plot([bx0, bx0, bx1, bx1], [by + 0.08, by, by, by + 0.08], color=MUTED, lw=0.7)
axA.text((bx0 + bx1) / 2, by - 0.08, "each item 700 ms, then 200 ms blank",
         ha="center", va="top", color=MUTED)
# critical-item label
axA.text(onset[crit] + DUR / 2, y0 + H + 0.30, "critical item", ha="center", va="bottom",
         color=GREEN, fontweight="bold")
axA.text(onset[crit] + DUR / 2, y0 + H + 0.08, "syntax error: variable in quotes",
         ha="center", va="bottom", color=MUTED, fontsize=6)
# not-to-scale breaks
for xb in ((GLOBAL[1] + FIX[0]) / 2, (onset[4] + DUR + RESP[0]) / 2):
    axA.text(xb, y0 + H / 2, "//", ha="center", va="center", color=MUTED, fontsize=6)
axA.text(XLIM[0], 1.62, "A", fontsize=10, fontweight="bold", va="top")

# ---------------------------------------------------------------- Panel B
axB.set_ylim(-1.55, 1.25)
by0, bh = 0.0, 0.62
role = {0: "none", 1: "crit", 2: "stored", 3: "stored", 4: "none"}
names = {0: "no epoch", 1: "critical item\n(analyzed)", 2: "item +1\n(stored)",
         3: "item +2\n(stored)", 4: "no epoch"}
for i, w in enumerate(items):
    x0 = onset[i]
    r = role[i]
    if r == "crit":
        fc, ec, ls, tc, lw = GREEN, GREEN, "-", "white", 1.0
    elif r == "stored":
        fc, ec, ls, tc, lw = GRAY_LIGHT, GRAY, "-", INK, 0.8
    else:
        fc, ec, ls, tc, lw = "white", GRAY, (0, (3, 2)), MUTED, 0.8
    axB.add_patch(Rectangle((x0, by0), DUR, bh, fc=fc, ec=ec, ls=ls, lw=lw, zorder=2))
    axB.text(x0 + DUR / 2, by0 + bh / 2, w, ha="center", va="center", family=MONO,
             fontsize=4.9 if len(w) > 8 else 6.3, color=tc,
             fontweight="bold" if r == "crit" else "normal", zorder=3)
    axB.text(x0 + DUR / 2, by0 + bh + 0.08, names[i], ha="center", va="bottom",
             fontsize=6.2, color=GREEN if r == "crit" else (INK if r == "stored" else MUTED),
             linespacing=1.15)

# analysis window bar under the critical item
wy = -0.25
axB.plot([-100, 800], [wy, wy], color=GREEN, lw=2.2, solid_capstyle="butt")
axB.plot([-100, -100], [wy - 0.08, wy + 0.08], color=GREEN, lw=1)
axB.plot([800, 800], [wy - 0.08, wy + 0.08], color=GREEN, lw=1)
axB.text(350, wy - 0.12, "analysis window: $-$100 to +800 ms", ha="center", va="top",
         color=GREEN, fontsize=6.2)

# time axis
ay = -0.85
axB.plot([onset[0] - 150, onset[4] + DUR + 150], [ay, ay], color=INK, lw=0.7)
for i in range(len(items)):
    t = onset[i]
    axB.plot([t, t], [ay - 0.06, ay + 0.06], color=INK, lw=0.7)
    lab = "0" if t == 0 else f"{t:+d}".replace("-", "−")
    axB.text(t, ay - 0.12, lab, ha="center", va="top", fontsize=6.3)
axB.text((onset[0] + onset[4] + DUR) / 2, ay - 0.42,
         "Time relative to the critical item (ms)", ha="center", va="top", fontsize=6.6)
axB.tick_params(bottom=False, labelbottom=False)
axB.text(XLIM[0], 1.22, "B", fontsize=10, fontweight="bold", va="top")

# faint guides linking the same items in A and B
for i in range(len(items)):
    for x in (onset[i], onset[i] + DUR):
        axB.plot([x, x], [by0 + bh, 1.25], color=GRAY, lw=0.4, ls=":", zorder=1,
                 clip_on=False)

for ext in ("png", "pdf", "tiff"):
    fig.savefig(f"fig_trial_structure.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
print("[SAVED] fig_trial_structure.png / .pdf / .tiff")