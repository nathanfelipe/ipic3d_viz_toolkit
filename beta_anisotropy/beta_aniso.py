import h5py, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# --------- config ---------
path = "../../DoubleGEM-Fields_300000.h5"   # or any of your snapshots # T3D-Fields_005500.h5
species = 0                         # 0 = ions, 1 = electrons (Double check ecsim convention)
combine_species = False             # if True, use P_total = P(species) + P(other)
stride = 2                          # downsample factor (adjust 2–6)
order = "C"                         # you verified C works for your file
eps_B2 = 1e-10                      # guard for tiny B^2
eps_Ppar = 1e-12                    # guard for tiny/neg P_parallel
# add config flag for comparing growth rates
compare_growth_rates = True   # if True, show side-by-side panels: left=1e-3, right=1e-1 (for any species)
# --------------------------

def open_var(f, name):
    obj = f[f"Step#0/Block/{name}"]
    if isinstance(obj, h5py.Group):
        if "0" in obj: return obj["0"]
        for k in obj.keys():
            if isinstance(obj[k], h5py.Dataset):
                return obj[k]
        raise KeyError(f"No dataset in group {name}")
    return obj

# --- load B components (downsampled) ---
with h5py.File(path, "r") as f:
    Bx = open_var(f, "Bx")[::stride, ::stride, ::stride].astype(np.float32)
    By = open_var(f, "By")[::stride, ::stride, ::stride].astype(np.float32)
    Bz = open_var(f, "Bz")[::stride, ::stride, ::stride].astype(np.float32)

# --- load pressure tensor components for selected species ---
suf = f"_{species}"
with h5py.File(path, "r") as f:
    Pxx = open_var(f, "Pxx"+suf)[::stride, ::stride, ::stride].astype(np.float32)
    Pyy = open_var(f, "Pyy"+suf)[::stride, ::stride, ::stride].astype(np.float32)
    Pzz = open_var(f, "Pzz"+suf)[::stride, ::stride, ::stride].astype(np.float32)
    Pxy = open_var(f, "Pxy"+suf)[::stride, ::stride, ::stride].astype(np.float32)
    Pxz = open_var(f, "Pxz"+suf)[::stride, ::stride, ::stride].astype(np.float32)
    Pyz = open_var(f, "Pyz"+suf)[::stride, ::stride, ::stride].astype(np.float32)

    if combine_species:
        # add the other species component-wise
        other = f"_{1-species}"
        Pxx += open_var(f, "Pxx"+other)[::stride, ::stride, ::stride].astype(np.float32)
        Pyy += open_var(f, "Pyy"+other)[::stride, ::stride, ::stride].astype(np.float32)
        Pzz += open_var(f, "Pzz"+other)[::stride, ::stride, ::stride].astype(np.float32)
        Pxy += open_var(f, "Pxy"+other)[::stride, ::stride, ::stride].astype(np.float32)
        Pxz += open_var(f, "Pxz"+other)[::stride, ::stride, ::stride].astype(np.float32)
        Pyz += open_var(f, "Pyz"+other)[::stride, ::stride, ::stride].astype(np.float32)

    # ---- sign correction: some outputs store electron pressures with a negative sign
    trace_mean = np.nanmean(Pxx + Pyy + Pzz)
    if trace_mean < 0:
        print("[note] pressure trace is negative on average -> flipping sign of all P components")
        Pxx *= -1.0; Pyy *= -1.0; Pzz *= -1.0
        Pxy *= -1.0; Pxz *= -1.0; Pyz *= -1.0

# --- compute b-hat and projections ---
B2 = Bx*Bx + By*By + Bz*Bz
invB = 1.0 / np.sqrt(np.maximum(B2, eps_B2))
bx = Bx * invB
by = By * invB
bz = Bz * invB

# P_parallel = b^T P b (fully anisotropic tensor)
Ppar = (
    bx*bx*Pxx + by*by*Pyy + bz*bz*Pzz
    + 2.0*(bx*by*Pxy + bx*bz*Pxz + by*bz*Pyz)
)

# P_perp = (Tr(P) - P_parallel)/2
traceP = Pxx + Pyy + Pzz
Pperp = 0.5 * (traceP - Ppar)

# guard: pressure should be >= 0 for physical temps; clip tiny negatives from numerics
Ppar = np.maximum(Ppar, 0.0)
Pperp = np.maximum(Pperp, 0.0)

# --- beta and anisotropy ---
beta_par = 2.0 * Ppar / np.maximum(B2, eps_B2)
# avoid 0/0; ensure strictly positive denominator
A = Pperp / np.maximum(Ppar, eps_Ppar)

# --- diagnostics ---
def stats(name, arr):
    arr = arr[np.isfinite(arr)]
    if arr.size:
        print(f"{name}: min={arr.min():.3e}, max={arr.max():.3e}, mean={arr.mean():.3e}")
    else:
        print(f"{name}: (no finite values)")

stats("B2", B2)
stats("Ppar", Ppar)
stats("Pperp", Pperp)
stats("beta_par", beta_par)
stats("A", A)

# --- mask (robust, relaxed for electrons) ---
mask = np.isfinite(beta_par) & np.isfinite(A) & (B2 > eps_B2) & (Ppar > 0.0)

n_total = beta_par.size
n_kept  = int(mask.sum())
print(f"kept {n_kept}/{n_total} points after masking")

if n_kept == 0:
    raise RuntimeError("No valid points after masking (even after sign correction/clip). Check electron pressure outputs at this snapshot.")

x = beta_par[mask].ravel(order=order)
y = A[mask].ravel(order=order)

if x.size == 0 or y.size == 0:
    raise RuntimeError("Still no valid data for this selection. Check pressure tensors/normalization at this snapshot.")

# --- 2D log-density histogram ---
if compare_growth_rates:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True, constrained_layout=True)
    ax1, ax2 = axes
else:
    fig, ax1 = plt.subplots(figsize=(7.5, 6), constrained_layout=True)

# restrict to reasonable anisotropy range for stability
x = x[x > 0]
y = y[(y > 0) & np.isfinite(y)]

xmin = max(1e-4, np.nanmin(x))
xmax = max(xmin*10, np.nanmax(x))
xbins = np.logspace(np.log10(xmin), np.log10(xmax), 80)

y_lo = np.nanpercentile(y, 0.1) if y.size else 0.5
y_hi = np.nanpercentile(y, 99.9) if y.size else 2.0
# enforce monotonic, positive range
ymin = max(0.1, min(y_lo, y_hi*0.95))
ymax = max(ymin*1.1, min(5.0, y_hi))
ybins = np.linspace(ymin, ymax, 80)

# build a shared LogNorm based on the data histogram
H_counts, _, _ = np.histogram2d(x, y, bins=[xbins, ybins])
vmin = max(1, H_counts[H_counts > 0].min()) if np.any(H_counts > 0) else 1
vmax = H_counts.max() if H_counts.size else 1
shared_norm = LogNorm(vmin=vmin, vmax=max(vmin, vmax))

# LEFT PANEL (always exists): plot data and 1e-3 curves
h1 = ax1.hist2d(x, y, bins=[xbins, ybins], norm=shared_norm, cmap='jet')
ax1.set_xscale('log')
ax1.set_xlabel(r'$\beta_\parallel$')
ax1.set_ylabel(r'$T_\perp/T_\parallel$')
ax1.grid(True, alpha=0.3)
ax1.axhline(1.0, color='w', lw=1.0, alpha=0.6)
label = 'total' if combine_species else f'species {species}'
ax1.set_title(f'Beta–Anisotropy (1e-3 growth, {label}, stride={stride})')

beta_vals = np.logspace(-2, 2.5, 200)
if species == 0:
    # ions: gamma/Omega_p = 1e-3
    A_ic_3  = 1 + 0.43 / (beta_vals ** 0.42)
    A_mm_3  = 1 + 0.77 / (beta_vals ** 0.76)
    A_fh_3  = 1 - 1.40 / (beta_vals ** 0.47)
    ax1.plot(beta_vals, A_ic_3, 'r--',  lw=1.6, label='Ion Cyclotron (1e-3)')
    ax1.plot(beta_vals, A_mm_3, 'm-.',  lw=1.6, label='Mirror Mode (1e-3)')
    ax1.plot(beta_vals, A_fh_3, 'b:',   lw=1.6, label='Firehose (1e-3)')
else:
    # electrons: Lazar+2018 (no explicit gamma dependence in this fit)
    A_ew  = 1 + 0.25 / (beta_vals ** 0.5)
    A_efh = 1 - 1.29 / (beta_vals ** 0.98)
    ax1.plot(beta_vals, A_ew,  'r--', lw=1.5, label='Whistler (electrons)')
    ax1.plot(beta_vals, A_efh, 'b:',  lw=1.5, label='Firehose (electrons)')

ax1.legend(loc='best', frameon=False)

if compare_growth_rates:
    # RIGHT PANEL: same data and 1e-1 curves; single colorbar here
    h2 = ax2.hist2d(x, y, bins=[xbins, ybins], norm=shared_norm, cmap='jet')
    ax2.set_xscale('log')
    ax2.set_xlabel(r'$\beta_\parallel$')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(1.0, color='w', lw=1.0, alpha=0.6)
    ax2.set_title('Beta–Anisotropy (1e-1 growth)')

    beta_vals = np.logspace(-2, 2.5, 200)
    if species == 0:
        # ions: gamma/Omega_p = 1e-1
        A_ic_1  = 1 + 0.92 / (beta_vals ** 0.40)
        A_mm_1  = 1 + 1.47 / (beta_vals ** 0.79)
        A_fh_1  = 1 - 1.70 / (beta_vals ** 0.40)
        ax2.plot(beta_vals, A_ic_1, 'r--', lw=1.5, label='Ion Cyclotron (1e-1)')
        ax2.plot(beta_vals, A_mm_1, 'm-.', lw=1.5, label='Mirror Mode (1e-1)')
        ax2.plot(beta_vals, A_fh_1, 'b:',  lw=1.5, label='Firehose (1e-1)')
    else:
        # electrons: same Lazar curves (plotted again for side-by-side visual)
        A_ew  = 1 + 0.25 / (beta_vals ** 0.5)
        A_efh = 1 - 1.29 / (beta_vals ** 0.98)
        ax2.plot(beta_vals, A_ew,  'r--', lw=1.5, label='Whistler (electrons)')
        ax2.plot(beta_vals, A_efh, 'b:',  lw=1.5, label='Firehose (electrons)')

    ax2.legend(loc='best', frameon=False)

    # synchronize axis limits for better visual comparison
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())

    # two separate colorbars to ensure same canvas size
    plt.colorbar(h1[3], ax=ax1, label='counts (log)')
    plt.colorbar(h2[3], ax=ax2, label='counts (log)')
else:
    # single-panel case: put colorbar on this lone axis
    plt.colorbar(h1[3], ax=ax1, label='counts (log)')

plt.show()