from dust_model import compare_charge_dist_mathis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cases = [
    ('CNM', 30.0, 0.0015, 100.0),
    ('WNM', 0.4, 0.1, 6000.0),
    ('WIM', 0.1, 0.99, 8000.0),
]

grain_sizes = [5e-7, 1e-6]  # 50A and 100A in cm

saved = []
for name, nH, xe, T in cases:
    ne = nH * xe
    for g in grain_sizes:
        print('Doing', g, ne, nH, T)
        res = compare_charge_dist_mathis('silicate', g, ne, nH, T, plot=False)
        Z = res['Z']; fwd = res['f_WD']; ffit = res['f_fit']
        plt.figure(figsize=(6,4))
        plt.step(Z, fwd + 1e-300, where='mid', label='WD01 solver')
        plt.step(Z, ffit + 1e-300, where='mid', label='Ibanez-Mejias fit')
        plt.yscale('log')
        plt.xlabel('Charge Z')
        plt.ylabel('Probability')
        plt.title(f'{name} - silicate a={g*1e8:.0f} A')
        plt.legend()
        fn = f'compare_mathis_{name}_silicate_{int(g*1e8)}A.png'
        plt.savefig(fn, dpi=200)
        plt.close()
        saved.append(fn)

        res = compare_charge_dist_mathis('graphite', g, ne, nH, T, plot=False)
        Z = res['Z']; fwd = res['f_WD']; ffit = res['f_fit']
        plt.figure(figsize=(6,4))
        plt.step(Z, fwd + 1e-300, where='mid', label='WD01 solver')
        plt.step(Z, ffit + 1e-300, where='mid', label='Ibanez-Mejias fit')
        plt.yscale('log')
        plt.xlabel('Charge Z')
        plt.ylabel('Probability')
        plt.title(f'{name} - graphite a={g*1e8:.0f} A')
        plt.legend()
        fn = f'compare_mathis_{name}_graphite_{int(g*1e8)}A.png'
        plt.savefig(fn, dpi=200)
        plt.close()
        saved.append(fn)

print('Saved files:', saved)
