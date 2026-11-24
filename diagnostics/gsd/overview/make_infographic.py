"""The model in one picture: the GSD state, how carbon moves, the collision rules and one step in one cell.

Redrawn 2026-11-23 after the carbon mass-transfer decisions (report Sect. "How carbon moves ..."); it replaces the
carrier-network figure of 2026-10-30. Writes ``figs/gsd_model.html`` (inline SVG) in the report's build directory
and, with headless Chrome, ``figs/gsd_model.pdf`` (vector, for the report) and ``figs/gsd_model.png`` (preview).
Set $CHROME to the Chrome or Chromium binary if it is not at the macOS default.

    python diagnostics/gsd/overview/make_infographic.py
"""
import math
import os
import subprocess
from pathlib import Path

W, H = 1100, 846
FIGS = Path(__file__).resolve().parents[3] / "results" / "gsd_overview" / "figs"
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

CSS = f"""
@page {{ size: {W}px {H}px; margin: 0; }}
html, body {{ margin: 0; padding: 0; background: #fcfcfb; }}
.viz-root {{ --surface:#fcfcfb; --panel:#f3f2ee; --ink:#0b0b0b; --ink2:#52514e; --muted:#7a7974;
  --border:rgba(11,11,11,0.16); --carbon:#2a78d6; --hydrogen:#eb6834; --charge:#1baf7a; --sil:#eda100;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
svg {{ display:block; width:{W}px; height:{H}px; background:var(--surface); }}
text {{ fill:var(--ink2); font-size:15px; }}
.ttl {{ fill:var(--ink); font-size:19px; font-weight:650; }}
.let {{ fill:var(--ink); font-size:19px; font-weight:750; }}
.hd {{ fill:var(--ink); font-size:16px; font-weight:650; }}
.ink {{ fill:var(--ink); }}
.mu {{ fill:var(--muted); }}
.it {{ font-style:italic; }}
.sm {{ font-size:14px; }}
.gl {{ fill:var(--carbon); }}
.panel {{ fill:var(--panel); stroke:var(--border); stroke-width:1; }}
.node {{ fill:var(--surface); stroke:var(--border); stroke-width:1.2; }}
.chip {{ fill:var(--surface); stroke:var(--border); stroke-width:1; }}
.chipC {{ fill:var(--surface); stroke:var(--carbon); stroke-width:1.5; }}
.chipH {{ fill:var(--surface); stroke:var(--hydrogen); stroke-width:1.6; }}
.chipQ {{ fill:var(--surface); stroke:var(--charge); stroke-width:1.6; }}
.dash {{ stroke-dasharray:5 4; }}
.flow {{ fill:none; stroke:var(--carbon); stroke-width:2; stroke-linecap:round; stroke-linejoin:round; }}
.flowg {{ fill:none; stroke:var(--ink2); stroke-width:1.6; stroke-linecap:round; }}
.halo {{ fill:none; stroke:var(--panel); stroke-width:7; }}
.lab {{ paint-order:stroke; stroke:var(--panel); stroke-width:6px; stroke-linejoin:round; }}
.tick {{ stroke:var(--muted); stroke-width:1; }}
.sub {{ baseline-shift:sub; font-size:72%; }}
.sup {{ baseline-shift:super; font-size:72%; }}
"""


def esc(s):
    return s.replace("&", "&amp;")


def T(x, y, s, cls="", anchor="start"):
    c = f' class="{cls}"' if cls else ""
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    return f'<text x="{x:.1f}" y="{y:.1f}"{c}{a}>{s}</text>'


def R(x, y, w, h, cls, rx=7, extra=""):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" class="{cls}"{extra}/>'


def sub(s):
    return f'<tspan class="sub">{s}</tspan>'


def sup(s):
    return f'<tspan class="sup">{s}</tspan>'


NC, MH, Eg, Td = f"N{sub('C')}", f"M{sub('H')}", f"E{sub('g')}", f"T{sub('d')}"
H2, C2H2 = f"H{sub('2')}", f"C{sub('2')}H{sub('2')}"


def path(pts, cls="flow", mk="C", start=False, dash=False, halo=False):
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    m = f' marker-end="url(#a{mk})"' if mk else ""
    if mk and start:
        m += f' marker-start="url(#a{mk})"'
    dd = ' stroke-dasharray="6 4"' if dash else ""
    h = f'<path d="{d}" class="halo"/>' if halo else ""
    return h + f'<path d="{d}" class="{cls}"{dd}{m}/>'


def defs():
    out = ["<defs>"]
    for k, col in (("C", "var(--carbon)"), ("G", "var(--ink2)")):
        out.append(f'<marker id="a{k}" viewBox="0 0 10 10" refX="8.6" refY="5" markerWidth="6.5" markerHeight="6.5" '
                   f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:{col}"/></marker>')
    out.append('<pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
               '<line x1="0" y1="0" x2="0" y2="7" style="stroke:var(--muted);stroke-width:1;stroke-opacity:0.45"/>'
               '</pattern>')
    out.append("</defs>")
    return "\n".join(out)


def panel(x, y, w, h, letter, title):
    return R(x, y, w, h, "panel", 10) + T(x + 16, y + 27, letter, "let") + T(x + 40, y + 27, title, "ttl")


# ---------------------------------------------------------------------------------------------------------------
# a: the state
X0, SW = 152, 76.0            # first slot and slot width: every bin drawn at equal width


def sx(i):
    return X0 + i * SW


def panel_a():
    s = [panel(20, 14, 1060, 290, "a", "The state: one carbon grid and one silicate grid, two moments per bin")]
    s.append(T(1064, 41, "every bin stores N and M; bins drawn at equal width", "mu", "end"))
    # family headers
    fams = [(0, 4, "molecules (PAH-like)", f"{NC} 20–400"), (4, 7, "a-C(:H) nano-grains", "1.06–20 nm"),
            (7, 8, "small", "20–53 nm"), (8, 11, "large grains", "53 nm – 1 µm, core/mantle"),
            (11, 12, "overflow", "1–10 µm")]
    for i0, i1, name, rng in fams:
        xa, xb = sx(i0) + 3, sx(i1) - 3
        xc = 0.5 * (xa + xb)
        s.append(T(xc, 70, name, "hd", "middle"))
        s.append(T(xc, 88, rng, "mu", "middle"))
        s.append(f'<path d="M{xa:.1f},100 L{xa:.1f},95 L{xb:.1f},95 L{xb:.1f},100" class="tick" fill="none"/>')
    # carbon row
    y, h = 104, 42
    s.append(T(36, y + 19, "carbon", "hd"))
    s.append(T(36, y + 37, "a-C(:H), ρ 1.6", "mu sm"))
    alpha = [0.13] * 4 + [0.24] * 3 + [0.36] + [0.48] * 3
    for i in range(11):
        s.append(R(sx(i) + 1.5, y, SW - 3, h, "", 5,
                   f' style="fill:var(--carbon);fill-opacity:{alpha[i]};stroke:var(--carbon);stroke-opacity:0.5"'))
        s.append(T(sx(i) + SW / 2, y + 26, "N, M", "ink", "middle"))
    s.append(R(sx(11) + 1.5, y, SW - 3, h, "chipC dash", 5, ' style="fill:url(#hatch)"'))
    s.append(T(sx(11) + SW / 2, y + 26, "N, M", "ink", "middle"))
    # edges
    edges = ["0.39", "0.50", "0.64", "0.83", "1.06", "2.8", "7.5", "20", "53", "141", "376", "1 µm", "10 µm"]
    for i, e in enumerate(edges):
        x = sx(i)
        s.append(f'<line x1="{x:.1f}" y1="{y + h + 2}" x2="{x:.1f}" y2="{y + h + 8}" class="tick"/>')
        s.append(T(x, y + h + 23, e, "mu sm", "end" if i == 12 else "middle"))
    s.append(T(36, y + h + 23, "edges (nm)", "mu sm"))
    # what the tables give, per family
    yt, ht = 182, 50
    s.append(T(36, yt + 21, "from", "hd"))
    s.append(T(36, yt + 40, "tables", "hd"))
    boxes = [(0, 4, "chip", ["H state, charge, heating, C loss:", "pahchem equilibrium tables"]),
             (4, 8, "chipH", ["band gap from the local UV–H balance;", f"tracked {MH} optional (0–4 bins, off)"]),
             (8, 11, "chip", ["core/mantle, H fixed:", "THEMIS optics, charging ac"]),
             (11, 12, "chip dash", ["only if", "filled"])]
    for i0, i1, cls, lines in boxes:
        s.append(R(sx(i0) + 1.5, yt, sx(i1) - sx(i0) - 3, ht, cls, 6))
        for k, ln in enumerate(lines):
            s.append(T(sx(i0) + 12, yt + 21 + 18 * k, ln, "ink" if k == 0 else ""))
    # silicate row
    ys = 246
    s.append(T(36, ys + 19, "silicate", "hd"))
    s.append(T(36, ys + 37, "8 bins, ρ 3.3", "mu sm"))
    sil = [(0, 4, "N, M   (0.45–1.18 nm)")] + [(i, i + 1, "N, M") for i in range(4, 11)]
    for i0, i1, lab in sil:
        s.append(R(sx(i0) + 1.5, ys, sx(i1) - sx(i0) - 3, h, "", 5,
                   ' style="fill:var(--sil);fill-opacity:0.22;stroke:var(--sil);stroke-opacity:0.7"'))
        s.append(T(0.5 * (sx(i0) + sx(i1)), ys + 26, lab, "ink", "middle"))
    s.append(R(sx(11) + 1.5, ys, SW - 3, h, "dash", 5,
               ' style="fill:url(#hatch);stroke:var(--sil);stroke-width:1.5"'))
    s.append(T(sx(11) + SW / 2, ys + 26, "N, M", "ink", "middle"))
    return "\n".join(s)


# ---------------------------------------------------------------------------------------------------------------
# b: how carbon moves
def panel_b():
    y0 = 318
    s = [panel(20, y0, 576, 318, "b", "How carbon moves between the families")]
    nodes = {"mol": (36, 156, "molecules", f"{NC} 20–400"), "nano": (248, 368, "nano-grains", "1–20 nm"),
             "gr": (460, 580, "grains", "20 nm – 10 µm")}
    yn, hn = 412, 56
    # freeze-out arc over the top: molecules -> grains
    xm, xg = 0.5 * (36 + 156), 0.5 * (460 + 580)
    s.append(path([(xm, yn), (xm, 384), (xg, 384), (xg, yn - 1)]))
    s.append(T(0.5 * (xm + xg), 376, "freeze-out: molecules stick to larger grains below ~4 km/s", "ink", "middle"))
    # gaps: upper -> and lower <-
    for (a, b, up, low) in (("mol", "nano", "sticking", "fragments"), ("nano", "gr", "coagulation", "fragments")):
        xa, xb = nodes[a][1], nodes[b][0]
        xc = 0.5 * (xa + xb)
        s.append(path([(xa + 4, yn + 22), (xb - 3, yn + 22)]))
        s.append(path([(xb - 4, yn + 36), (xa + 3, yn + 36)]))
        s.append(T(xc, yn + 15, up, "sm", "middle"))
        s.append(T(xc, yn + 53, low, "sm", "middle"))
    for k, (x0, x1, name, rng) in nodes.items():
        s.append(R(x0, yn, x1 - x0, hn, "node", 8))
        s.append(T(0.5 * (x0 + x1), yn + 24, name, "hd", "middle"))
        s.append(T(0.5 * (x0 + x1), yn + 43, rng, "mu", "middle"))
    # gas box and the exchanges
    yg, hg = 508, 116
    s.append(R(36, yg, 544, hg, "node", 8))
    s.append(T(48, yg + 22, "gas", "hd"))
    s.append(T(84, yg + 22, f"C⁺, C, CO and small hydrocarbons (C{sub('2')}H{sub('2')}, …)", "mu"))
    cols = {"mol": (44, [("▼", "photolysis (C₂H₂, C)", ""), ("▼", "impacts above ~7 km/s", ""),
                         ("▼", "hot-gas erosion", ""), ("▲", "bottom-up source, growth", "it")]),
            "nano": (254, [("▼", "sputtering", ""), ("▼", "Coulomb explosion", ""), ("▼", "UV fragmentation", "it")]),
            "gr": (424, [("▼", "sputtering", ""), ("▼", "sublimation", ""), ("▲", "accretion of C⁺, C", "")])}
    for k, (x, items) in cols.items():
        for j, (g, txt, cls) in enumerate(items):
            yy = yg + 44 + 19 * j
            s.append(T(x, yy, g, "gl sm"))
            s.append(T(x + 16, yy, esc(txt), ("ink " + cls).strip()))
    # vertical arrows between the nodes and the gas
    yb = yn + hn
    s.append(path([(70, yb + 1), (70, yg - 2)]))
    s.append(path([(126, yg - 1), (126, yb + 2)], dash=True))
    s.append(path([(286, yb + 1), (286, yg - 2)]))
    s.append(path([(330, yb + 1), (330, yg - 2)], dash=True))
    s.append(path([(494, yb + 1), (494, yg - 2)]))
    s.append(path([(546, yg - 1), (546, yb + 2)]))
    return "\n".join(s)


# ---------------------------------------------------------------------------------------------------------------
# c: who collides with whom
def panel_c():
    x0, y0, w = 612, 318, 468
    s = [panel(x0, y0, w, 318, "c", "Who collides with whom")]
    names = ["molecule", "nano-grain", "grain"]
    xh, cw, ch = 724, 113.0, 44.0
    yt = 372
    for j, n in enumerate(names):
        s.append(T(xh + cw * (j + 0.5), yt - 8, n, "hd", "middle"))
    cells = {(0, 0): ("—", "no outcome", "g"), (1, 0): ("by speed,", "+ UV release", "c"), (2, 0): ("by speed,", "+ UV release", "c"),
             (1, 1): ("coagulate", "or shatter", "n"), (2, 1): ("coagulate", "or shatter", "n"),
             (2, 2): ("coagulate", "or shatter", "n")}
    for i, n in enumerate(names):
        y = yt + i * ch
        s.append(T(xh - 10, y + ch / 2 + 5, n, "hd", "end"))
        for j in range(3):
            x = xh + j * cw
            if (i, j) not in cells:
                continue
            l1, l2, kind = cells[(i, j)]
            style = {"g": ' style="fill:var(--panel);stroke:var(--border)"',
                     "c": ' style="fill:var(--carbon);fill-opacity:0.16;stroke:var(--carbon);stroke-opacity:0.6"',
                     "n": ' style="fill:var(--surface);stroke:var(--border)"'}[kind]
            s.append(R(x + 2, y + 2, cw - 4, ch - 4, "", 5, style))
            if l2:
                s.append(T(x + cw / 2, y + 19, l1, "ink sm", "middle"))
                s.append(T(x + cw / 2, y + 36, l2, "ink sm", "middle"))
            else:
                s.append(T(x + cw / 2, y + 27, l1, "ink sm", "middle"))
    s.append(T(xh + 2 * cw, yt + 19, "grains coagulate below and", "mu sm", "middle"))
    s.append(T(xh + 2 * cw, yt + 36, "shatter above threshold speeds", "mu sm", "middle"))
    # the speed scale for a molecule hitting a larger particle
    xa, xb, v0, v1 = 640.0, 1060.0, 0.03, 40.0

    def xv(v):
        return xa + (math.log10(v) - math.log10(v0)) / (math.log10(v1) - math.log10(v0)) * (xb - xa)

    yb = 552
    s.append(T(xa, yb - 22, "molecule hitting a larger particle:", "ink"))
    s.append(T(xv(5.3), yb - 6, "rebound", "sm", "middle"))
    s.append(R(xa, yb, xv(4) - xa, 24, "", 3,
               ' style="fill:var(--carbon);fill-opacity:0.30;stroke:none"'))
    s.append(R(xv(4), yb, xv(7) - xv(4), 24, "", 0, ' style="fill:var(--muted);fill-opacity:0.18;stroke:none"'))
    s.append(R(xv(7), yb, xb - xv(7), 24, "", 3, ' style="fill:var(--ink2);fill-opacity:0.55;stroke:none"'))
    s.append(T(0.5 * (xa + xv(4)), yb + 17, "sticks (freeze-out)", "ink", "middle"))
    s.append(T(0.5 * (xv(7) + xb), yb + 17, "destroyed", "", "middle").replace("<text", '<text style="fill:#fcfcfb"'))
    for v, lab in ((0.1, "0.1"), (1, "1"), (4, "4"), (7, "7"), (30, "30 km/s")):
        x = xv(v)
        s.append(f'<line x1="{x:.1f}" y1="{yb + 24}" x2="{x:.1f}" y2="{yb + 29}" class="tick"/>')
        s.append(T(x, yb + 43, lab, "mu sm", "end" if v == 30 else "middle"))
    for va, vb, lab in ((0.05, 1.7, "cold gas: CNM, cores"), (9, 30, "warm gas")):
        a, b = xv(va), xv(vb)
        yy = yb + 52
        s.append(f'<path d="M{a:.1f},{yy - 4} L{a:.1f},{yy} L{b:.1f},{yy} L{b:.1f},{yy - 4}" class="tick" fill="none"/>')
        s.append(T(0.5 * (a + b), yy + 17, lab, "sm", "middle"))
    return "\n".join(s)


# ---------------------------------------------------------------------------------------------------------------
# d: one step in one cell
def chain(y, label, sub_label, chips, x_start=150, gap=18):
    s = [T(36, y + 20, label, "hd"), T(36, y + 38, sub_label, "mu sm")]
    x = x_start
    for k, (l1, l2, w, cls) in enumerate(chips):
        s.append(R(x, y, w, 46, cls, 6))
        s.append(T(x + w / 2, y + 20, l1, "ink", "middle"))
        s.append(T(x + w / 2, y + 38, l2, "", "middle"))
        if k < len(chips) - 1:
            s.append(path([(x + w + 3, y + 23), (x + w + gap - 3, y + 23)], cls="flowg", mk="G"))
        x += w + gap
    return "\n".join(s)


def panel_d():
    y0 = 650
    s = [panel(20, y0, 1060, 168, "d", "One step in one cell, as RAMSES will run it (A3c)")]
    s.append(chain(y0 + 44, "precool", "start of the step", [
        ("RT photon groups,", "T, n(H), n(e)", 136, "chip dash"),
        ("band gap of the", "covalent bins", 120, "chipH"),
        ("optics from 5", "pseudo-bins per bin", 144, "chip"),
        ("dust temperature", f"{Td} per bin", 126, "chip"),
        ("charge: one solve", "per bin, at a*", 136, "chipQ"),
        ("to the gas: heating,", f"cooling, {H2} formation", 160, "chip")]))
    s.append(chain(y0 + 104, "update", "over Δt", [
        ("collisions", "with the pair rule", 138, "chipC"),
        ("sinks per moment", "Coulomb explosion", 150, "chipC"),
        ("erosion shifts", "sputtering, C loss", 144, "chipC"),
        ("exchange", "accretion, sublimation", 168, "chipC"),
        ("checks: C, H, elements", "conserved; overflow share", 226, "chip")]))
    return "\n".join(s)


def legend():
    y = 836
    s = [path([(36, y - 5), (70, y - 5)]), T(78, y, "on by default", "sm")]
    s.append(path([(186, y - 5), (220, y - 5)], dash=True))
    s.append(T(228, y, "opt-in or off by default (labelled; italic in b); a* = the area node ∫a³dN / ∫a²dN", "sm"))
    s.append(R(812, y - 14, 30, 16, "", 3, ' style="fill:url(#hatch);stroke:var(--muted)"'))
    s.append(T(850, y, "computed only when filled", "sm"))
    return "\n".join(s)


def build():
    return "\n".join([f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" '
                      'aria-label="The CALIMA grain-size-distribution model: state, carbon flows, collision rules and '
                      'the per-cell step">', defs(), panel_a(), panel_b(), panel_c(), panel_d(), legend(), "</svg>"])


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    html = FIGS / "gsd_model.html"
    html.write_text(f'<!doctype html><html><head><meta charset="utf-8"><title>The GSD model in one picture</title>'
                    f'<style>{CSS}</style></head><body><div class="viz-root">{build()}</div></body></html>')
    print(html)
    if not Path(CHROME).exists():
        print("Chrome not found (set $CHROME): HTML only")
        return
    base = [CHROME, "--headless", "--disable-gpu", "--hide-scrollbars"]
    subprocess.run(base + ["--no-pdf-header-footer", f"--print-to-pdf={FIGS / 'gsd_model.pdf'}", html.as_uri()],
                   check=True, capture_output=True)
    subprocess.run(base + [f"--window-size={W},{H}", "--force-device-scale-factor=2",
                           f"--screenshot={FIGS / 'gsd_model.png'}", html.as_uri()], check=True, capture_output=True)
    print(FIGS / "gsd_model.pdf")


if __name__ == "__main__":
    main()
