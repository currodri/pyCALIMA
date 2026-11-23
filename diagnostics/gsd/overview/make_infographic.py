"""Infographic of the carbon-carrier plan (okay-create-again-the-flickering-summit.md, 2026-10-30).

Writes carbon_network.html (inline SVG, light and dark themes, hover titles) next to this
script. Render a PNG with headless Chrome:
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless \
      --screenshot=carbon_network.png --window-size=1600,1196 carbon_network.html
"""
from pathlib import Path

W, H = 1600, 1196
OUT = Path(__file__).resolve().parents[3] / "results" / "gsd_overview" / "figs" / "carbon_network.html"

CSS = """
.viz-root { color-scheme: light;
  --page:#f9f9f7; --surface:#fcfcfb; --panel:#f3f2ee; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --axis:#c3c2b7; --border:rgba(11,11,11,0.14);
  --carbon:#2a78d6; --hydrogen:#eb6834; --charge:#1baf7a; --photon:#52514e; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) .viz-root { color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19; --panel:#242423; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781;
  --axis:#383835; --border:rgba(255,255,255,0.14);
  --carbon:#3987e5; --hydrogen:#d95926; --charge:#199e70; --photon:#c3c2b7; } }
:root[data-theme="dark"] .viz-root { color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19; --panel:#242423; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781;
  --axis:#383835; --border:rgba(255,255,255,0.14);
  --carbon:#3987e5; --hydrogen:#d95926; --charge:#199e70; --photon:#c3c2b7; }
body { margin:0; background:var(--page); }
.viz-root { background:var(--page); font-family:system-ui,-apple-system,"Segoe UI",sans-serif; }
svg { display:block; width:100%; max-width:1600px; margin:0 auto; background:var(--surface); }
text { fill:var(--ink2); font-size:13px; }
.t1 { fill:var(--ink); font-size:22px; font-weight:650; }
.t2 { fill:var(--ink2); font-size:14px; }
.bt { fill:var(--ink); font-size:16px; font-weight:650; }
.hd { fill:var(--ink); font-size:14px; font-weight:650; }
.sm { font-size:11.5px; }
.mu { fill:var(--muted); }
.ink { fill:var(--ink); }
.sub { baseline-shift:sub; font-size:72%; }
.sup { baseline-shift:super; font-size:72%; }
.box { fill:var(--panel); stroke:var(--border); stroke-width:1; }
.opt { fill:none; stroke:var(--muted); stroke-width:1.5; stroke-dasharray:5 4; }
.chip { fill:var(--surface); stroke:var(--border); stroke-width:1; }
.chipH { fill:var(--surface); stroke:var(--hydrogen); stroke-width:1.5; }
.badge { fill:none; stroke:var(--muted); stroke-width:1; }
.axis { stroke:var(--axis); stroke-width:1.5; }
.halo { stroke:var(--surface); stroke-width:7; fill:none; }
.C { stroke:var(--carbon); } .Hh { stroke:var(--hydrogen); } .Q { stroke:var(--charge); } .P { stroke:var(--photon); }
.flow { fill:none; stroke-width:2; stroke-linecap:round; }
.ph { fill:none; stroke-width:1.6; }
g.proc:hover .flow { stroke-width:3.2; }
.sC { fill:var(--carbon); } .sHh { fill:var(--hydrogen); } .sQ { fill:var(--charge); }
.span { fill-opacity:0.22; }
.guide { stroke:var(--axis); stroke-width:1; stroke-dasharray:2 4; }
.lab { paint-order:stroke; stroke:var(--surface); stroke-width:5px; stroke-linejoin:round; }
.fC { fill:var(--carbon); } .fH { fill:var(--hydrogen); } .fQ { fill:var(--charge); } .fP { fill:var(--photon); }
"""


def t(x, y, s, cls="", anchor="start", extra=""):
    c = f' class="{cls}"' if cls else ""
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    return f'<text x="{x}" y="{y}"{c}{a}{extra}>{s}</text>'


def sub(s):
    return f'<tspan class="sub">{s}</tspan>'


MC, MH, NC, NH = f"M{sub('C')}", f"M{sub('H')}", f"N{sub('C')}", f"N{sub('H')}"
H2 = f"H{sub('2')}"


def markers():
    out = ["<defs>"]
    for k, cls in (("C", "fC"), ("H", "fH"), ("Q", "fQ"), ("P", "fP")):
        out.append(f'<marker id="ah{k}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" '
                   f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="{cls}"/></marker>')
    out.append('<linearGradient id="hgrad" x1="0" x2="1" y1="0" y2="0">'
               '<stop offset="0" style="stop-color:var(--hydrogen);stop-opacity:0.10"/>'
               '<stop offset="1" style="stop-color:var(--hydrogen);stop-opacity:0.95"/></linearGradient>')
    out.append("</defs>")
    return "\n".join(out)


def line(x1, y1, x2, y2, cls, mk=None, start=False, halo=False, dash=False):
    d = ' stroke-dasharray="6 4"' if dash else ""
    m = f' marker-end="url(#ah{mk})"' if mk else ""
    if mk and start:
        m += f' marker-start="url(#ah{mk})"'
    h = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="halo"/>' if halo else ""
    return h + f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="flow {cls}"{d}{m}/>'


def wavy_v(x, y0, y1, amp=4.0, half=7.0):
    """A photon: a vertical wavy line from y0 to y1 (either direction), straight tip, arrowhead."""
    sgn = 1 if y1 > y0 else -1
    n = int((abs(y1 - y0) - 12) // half)
    d = [f"M{x},{y0}"]
    for i in range(n):
        a = amp if i % 2 == 0 else -amp
        d.append(f"q{a},{sgn * half / 2} 0,{sgn * half}")
    d.append(f"L{x},{y1}")
    return f'<path d="{" ".join(d)}" class="ph P" marker-end="url(#ahP)"/>'


def chips(x, y, labels):
    """Small rounded chips in a row; label = (text, kind) with kind '', 'H' (hydrogen) or 'opt'."""
    out, cx = [], x
    for s, kind, w in labels:
        cls = {"": "chip", "H": "chipH", "opt": "opt"}[kind]
        out.append(f'<rect x="{cx}" y="{y}" width="{w}" height="22" rx="5" class="{cls}"/>')
        out.append(t(cx + w / 2, y + 15.5, s, "ink", "middle"))
        cx += w + 6
    return "\n".join(out)


def box(x, y, w, h, title, lines, chip_row, bar=None):
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" class="box"/>',
           t(x + 16, y + 26, title, "bt")]
    yy = y + 46
    for s in lines[:2]:
        out.append(t(x + 16, yy, s))
        yy += 18
    out.append(t(x + 16, y + 92, "stored:", "sm mu"))
    out.append(chips(x + 66, y + 76, chip_row))
    yy = y + 122
    if bar:
        head, ticks = bar
        out.append(t(x + 16, yy, head))
        bw = w - 32
        out.append(f'<rect x="{x + 16}" y="{yy + 9}" width="{bw}" height="8" rx="4" fill="url(#hgrad)"/>')
        for f, s, anc in ticks:
            out.append(t(x + 16 + f * bw, yy + 33, s, "sm", anc))
        yy += 56
    for s in lines[2:]:
        out.append(t(x + 16, yy, s))
        yy += 18
    return "\n".join(out)


def row(y, x0, x1, cls, mk, badge, label, tip, arrow, dash=False):
    """A process row: bracket over the carriers it acts on, label above, arrow to/from the gas column.
    arrow = ('out' | 'in' | 'both', x_gas)."""
    kind, xg = arrow
    tick = f'<rect x="{x0}" y="{y - 5}" width="{x1 - x0}" height="10" rx="5" class="span s{cls}"/>'
    dd = ' stroke-dasharray="6 4"' if dash else ""
    br = f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" class="flow {cls}"{dd}/>'
    if kind == "out":
        ar = line(x1, y, xg, y, cls, mk, dash=dash)
    elif kind == "in":
        ar = line(xg, y, x1, y, cls, mk, dash=dash)
    else:
        ar = line(x1 + 2, y, xg, y, cls, mk, start=True)
    bx = 44
    b = (f'<rect x="{bx}" y="{y - 24}" width="30" height="18" rx="4" class="badge"/>'
         + t(bx + 15, y - 10.5, badge, "sm mu", "middle"))
    return (f'<g class="proc"><title>{tip}</title>'
            f'<rect x="{x0}" y="{y - 30}" width="{xg - x0}" height="36" fill="transparent"/>'
            f"{tick}{br}{ar}{b}{t(84, y - 12, label, 'lab')}</g>")


def build():
    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" '
         f'aria-label="How the carbon carriers of the CALIMA GSD plan exchange carbon, hydrogen, charge and photons '
         f'with each other and with the gas">', markers()]

    s.append(t(40, 42, "Carbon carriers in the CALIMA grain-size plan: what they exchange", "t1"))
    s.append(t(40, 66, "One carbon population per size bin. Every bin stores N and " + MC
               + "; the covalent bins below 53 nm can also store the hydrogen mass " + MH + " (rung M1). "
               "Molecular-bin H state, charge, dust temperature and band emission are tabulated.", "t2"))

    # radiation band
    s.append(f'<rect x="40" y="88" width="1520" height="108" rx="10" class="box"/>')
    s.append(t(56, 112, "Radiation (RT groups, per cell)", "hd"))
    cx = 56
    for lab, w in (("FUV 6–13.6 eV", 130), ("EUV &gt; 13.6 eV (H II regions)", 220), ("optical", 80),
                   ("IR (trapped, re-emitted)", 190)):
        s.append(f'<rect x="{cx}" y="124" width="{w}" height="24" rx="5" class="chip"/>')
        s.append(t(cx + w / 2, 140.5, lab, "ink", "middle"))
        cx += w + 8
    s.append(t(56, 176, "Absorbed photons, through each bin's group cross-sections, drive heating (T"
               + sub("d") + ", band emission), photoemission, C–H bond breaking and fragmentation. "
               "The same cross-sections give the RT opacity and the radiation pressure (TVA drift)."))

    # photons down to each carrier, IR up
    for x in (100, 600, 775, 960):
        s.append(wavy_v(x, 198, 316))
    s.append(wavy_v(1140, 318, 200))
    s.append(t(1152, 252, "IR emission", "sm"))
    s.append(t(1152, 267, "radiation pressure → drift", "sm"))

    # PAH clusters (M2)
    s.append('<g class="proc"><title>M2 only: PAH clusters (van der Waals bonded), ~300–1000 C, around N_C,max. '
             'Photo-evaporation into free PAHs from PAHdb cluster modes and binding energies (Rapacioli+05/06, '
             'Montillaud &amp; Joblin 2014); gate: Pilleri+12 eVSG fraction against G0.</title>')
    s.append('<rect x="230" y="246" width="330" height="44" rx="8" class="opt"/>')
    s.append(t(244, 265, "PAH clusters (M2 only)", "ink"))
    s.append(t(244, 282, "~300–1000 C, van der Waals; own N, M", "sm"))
    s.append(line(270, 291, 270, 318, "C", "C", dash=True))
    s.append(t(280, 310, "photo-evaporation → free PAHs", "sm"))
    s.append("</g>")

    y0, hb = 320, 232
    s.append(box(40, y0, 300, hb, "Molecular carbon",
                 ["PAHs and their H-modified forms", f"{NC} 20–400 · 0.39–1.06 nm · 4 bins",
                  "H state, charge −1…+2, heating, C loss:", "equilibrium tables from pahchem's",
                  "joint (Z, N_H) chain (decided 2026-11-15)"],
                 [("N", "", 30), (MC, "", 40), ("H: table", "opt", 70)],
                 (f"H state (tabulated): mean {NH}/{NC}",
                  [(0.0, "dehydro.", "start"), (0.36, "normal", "middle"), (0.66, "super-H", "middle"),
                   (1.0, "aliphatic", "end")])))
    s.append(box(360, y0, 300, hb, "Covalent a-C nano-grains",
                 ["aromatic a-C ↔ hydrogenated a-C:H", "400 C – 20 nm · 3 bins",
                  "optics: Jones 2012 n,k on an E" + sub("g") + " axis", "charging: ac ↔ ach end members",
                  "fragmentation: Jones &amp; Ysard 2025 (M2)"],
                 [("N", "", 30), (MC, "", 40), (MH, "H", 40)],
                 ("H/C sets the band gap E" + sub("g"),
                  [(0.0, "0.1 eV (aromatic)", "start"), (1.0, "2.5 eV (a-C:H)", "end")])))
    s.append(box(680, y0, 190, hb, "Small grains",
                 ["20–53 nm · 1 bin", "", "whole grain processed", "(UV depth ≈ 80 nm)",
                  "M0: CM20 structure", "M1: tracked H/C"],
                 [("N", "", 30), (MC, "", 40), (MH, "H", 40)]))
    s.append(box(890, y0, 310, hb, "Large grains",
                 ["CM20: a-C:H core in an a-C mantle", "53 nm – 1 µm · 3 bins + overflow",
                  "core H inert, surface H-poor", "optics: coated Mie (Jones n,k)", "charging: ac"],
                 [("N", "", 30), (MC, "", 40), ("skin H (M3)", "opt", 100)]))

    # right panel: what is stored, the H step
    s.append(f'<rect x="1220" y="{y0}" width="340" height="{hb}" rx="10" class="box"/>')
    s.append(t(1236, y0 + 26, "Per bin and per cell", "bt"))
    lines = ["Stored and advected: N, " + MC + " in every bin;",
             MH + " in covalent bins below 53 nm (M1), a child of " + MC + ".",
             "Tabulated each step: charge f(Z), conditional",
             "on H; T" + sub("d") + "; band emission.",
             "Carbon moved between bins carries H at",
             "the source bin's H/C.",
             "H step, exact for rates fixed over Δt:"]
    yy = y0 + 50
    for ln in lines:
        s.append(t(1236, yy, ln))
        yy += 18
    s.append(t(1236, yy + 4, MH + " → M" + sub("eq") + " + (" + MH + " − M" + sub("eq") + ") e"
               + '<tspan class="sup">−(k' + '<tspan font-size="80%">UV</tspan>' + "+k"
               + '<tspan font-size="80%">H</tspan>' + ")Δt</tspan>", "ink"))
    s.append(t(1236, yy + 26, "equilibrium where (k" + sub("UV") + "+k" + sub("H") + ")Δt ≫ 1, history otherwise;", "sm"))
    s.append(t(1236, yy + 41, "the P2 tests decide where one scalar is enough", "sm"))

    # carbon moves between bins, and the size axis
    s.append('<g class="proc"><title>Collisions and growth move carbon between bins (gsd.collisions, remap); '
             'every move carries H at the source bin\'s H/C. Shattering fragments keep the parent\'s H/C '
             '(x_min for CM20 large grains). Photo-fragmentation: Jones &amp; Ysard 2025, opt-in until gated.</title>')
    s.append(t(44, 582, "coagulation · freeze-out · accretion growth"))
    s.append(line(345, 578, 1190, 578, "C", "C"))
    s.append(t(44, 608, "shattering · photo-fragmentation (M2)"))
    s.append(line(1190, 604, 345, 604, "C", "C"))
    s.append("</g>")
    s.append('<line x1="40" y1="632" x2="1200" y2="632" class="axis"/>')
    for x, lab, anc in ((40, "0.39 nm", "start"), (350, f"1.06 nm ({NC} = 400)", "middle"), (670, "20 nm", "middle"),
                        (880, "53 nm", "middle"), (1200, "1 µm", "end")):
        s.append(f'<line x1="{x}" y1="627" x2="{x}" y2="637" class="axis"/>')
        s.append(t(x, 651, lab, "sm mu", anc))
    s.append(t(510, 651, "grain radius →", "sm mu", "middle"))

    # gas column
    xg = 1260
    s.append(f'<rect x="1240" y="652" width="320" height="440" rx="10" class="box"/>')
    s.append(t(1256, 676, "Gas phase (per cell)", "hd"))

    def gchip(y, h, lines_):
        out = [f'<rect x="{xg}" y="{y}" width="284" height="{h}" rx="6" class="chip"/>']
        for i, ln in enumerate(lines_):
            out.append(t(xg + 12, y + 17 + 16 * i, ln, "ink" if i == 0 else "sm"))
        return "\n".join(out)

    s.append(gchip(690, 34, ["electrons e⁻ · ions X⁺"]))
    s.append(gchip(748, 34, [H2]))
    s.append(t(xg + 12, 802, "↑ " + H2 + " formation on all grain surfaces", "sm"))
    s.append(gchip(810, 34, ["H I"]))
    s.append(gchip(870, 50, ["small hydrocarbons", "CH₄, C₂H₂, C₄H" + sub("k") + " → gas C and H"]))
    s.append(gchip(944, 136, ["gas carbon", "C⁺, C, CO (RTZ element block);", "only C⁺ and C grow the carriers"]))

    rows = [
        (707, 40, 1200, "Q", "Q", "M0",
         "charging: photoelectrons out (photoelectric heating of the gas); electrons and ions in",
         "Grains: ac (default), ach, cm20 (DustEM benchmark); intermediate H/C interpolates the end members. "
         "Molecular bins: pahchem group_tables, conditional on the H state.", ("both", xg)),
        (765, 40, 870, "Hh", "H", "M1",
         "UV H loss: lab yield per absorbed photon, + multi-photon channels in small particles → " + H2
         + " (+ CH₄ ≈ 3 %)",
         "Yield 0.014–0.025 C–H per absorbed photon (Alata+14, Maté 2014); σ_des 1–3e-19 cm2 (Mennella+01); "
         "floor X_H ≈ 0.05. H2 release from cold grains diffusion-limited (Martín-Doménech+16) or immediate.",
         ("out", xg)),
        (827, 40, 870, "Hh", "H", "M1",
         "re-hydrogenation by H atoms (grains: Mennella+02 · molecules: Goumans H addition, Eley–Rideal)",
         "σ_f = 1.9e-18 cm2 per H atom, plateau H/C ≈ 0.65, temperature dependence from Mennella+06. "
         "No free efficiency factors.", ("in", xg)),
        (895, 40, 660, "C", "C", "M0",
         "carbon loss: C₂H₂, C₂, C₃ from molecules · C₄H" + sub("k") + " bridges from nano-grains (M2)",
         "Molecules: export_pah_tables. Nano-grains: Jones &amp; Ysard 2025 bridge loss (6.4 eV, k0 = 1e16 s-1), "
         "order of magnitude, opt-in until gated on a_min (Elyajouri+24/25).", ("out", xg)),
        (962, 40, 1200, "C", "C", "M0",
         "growth by C⁺ and C: grains stick (S·p" + sub("2") + "); molecules keep carbon only with f" + sub("acc")
         + "·p" + sub("ret") + " (0 by default); carbon only",
         "Grains: n v pi a2 J(Z) S(T) p2 (S 0.06-0.3, Bossion+24, Hirashita &amp; Kuo 11). Molecules: k_cap x f_acc x "
         "p_ret; PDR codes keep no carbon (default); anthracene f_acc = 0.4 (Canosa+95); net f_acc p_ret below 0.15 "
         "(Omont &amp; Bettinger 21). CO and ices grow nothing. Hydrogen comes from the re-hydrogenation row.",
         ("in", xg)),
        (1012, 40, 1200, "C", "C", "M0",
         "sputtering (all sizes; H first, or at the grain's H/C) · sublimation (large grains, near stars)",
         "Large grains: Nozawa / Bocchio; a-C(:H): Serra Díaz-Cano &amp; Jones 2008, Bocchio+12/14. "
         "Sublimation: graphite vapour pressure as a proxy.", ("out", xg)),
    ]
    for x in (350, 670, 880):
        s.append(f'<line x1="{x}" y1="640" x2="{x}" y2="1030" class="guide"/>')
    for r in rows:
        s.append(row(*r))
    s.append(row(1064, 40, 118, "C", "C", "opt",
                 "bottom-up PAH formation in shielded gas (opt-in, calibrated: κ or ε" + sub("ff")
                 + "), injected at " + NC + " 20–24",
                 "Form A: kappa n_H n(C) above A_c, below T_max. Form B: eps_ff f_sh rho_C,free / t_ff. "
                 "Calibrated on TMC-1 (McGuire+21, Wenzel+24/25), phase parity (Chastenet+19), the A_V 1-3 doubling "
                 "(Zhang+25) and q_PAH against metallicity. Off by default.", ("in", xg), dash=True))

    # legend
    ly = 1130
    s.append(f'<line x1="40" y1="1106" x2="1560" y2="1106" class="axis"/>')
    items = [("C", "carbon"), ("Hh", "hydrogen"), ("Q", "charge (electrons, ions)")]
    lx = 44
    for cls, lab in items:
        s.append(f'<line x1="{lx}" y1="{ly - 4}" x2="{lx + 34}" y2="{ly - 4}" class="flow {cls}"/>')
        s.append(t(lx + 42, ly, lab))
        lx += 42 + 7 * len(lab) + 30
    s.append(f'<path d="M{lx},{ly - 4} q4,-3.5 7,0 q3,3.5 7,0 q4,-3.5 7,0 q3,3.5 7,0 q4,-3.5 7,0" class="ph P"/>')
    s.append(t(lx + 44, ly, "photons and energy"))
    lx += 44 + 150
    s.append(f'<rect x="{lx}" y="{ly - 13}" width="34" height="18" rx="4" class="opt"/>')
    s.append(t(lx + 42, ly, "optional (M2, M3, opt-in)"))
    lx += 42 + 175
    s.append(f'<rect x="{lx}" y="{ly - 13}" width="30" height="18" rx="4" class="badge"/>')
    s.append(t(lx + 15, ly, "M1", "sm mu", "middle"))
    s.append(t(lx + 38, ly, "the rung that adds the process (M0 = baseline)"))
    s.append(t(44, 1158, "Graphite + astrosilicate run as a control at every rung. Hover a row or box for its rate "
               "sources; the plan file's Processes table has them all.", "sm mu"))
    s.append(t(44, 1176, "Plan: ~/.claude/plans/okay-create-again-the-flickering-summit.md (2026-10-30; molecular H tabulated since 2026-11-15).", "sm mu"))
    s.append("</svg>")
    return "\n".join(s)


html = (f'<!doctype html><html><head><meta charset="utf-8"><title>Carbon carriers: interactions</title>'
        f'<style>{CSS}</style></head><body><div class="viz-root">{build()}</div></body></html>')
OUT.write_text(html)
print(OUT)
