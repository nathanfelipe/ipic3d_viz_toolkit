import h5py, numpy as np, matplotlib.pyplot as plt
from numpy.fft import rfftn, rfftfreq, fftfreq

# -------- config --------
path   = "../../DoubleGEM-Fields_300000.h5"
stride = 2                 # 2–4 is a good starting point
order  = "C"               # your file is C-ordered
DX, DY, DZ = 0.125, 0.125, 0.125    # cell sizes # 0.125, 0.125, 0.125
compute_kperp_kpar = False             # set True to also make a 2D map
# ------------------------

def open_var(f, name):
    obj = f[f"Step#0/Block/{name}"]
    if isinstance(obj, h5py.Group):
        return obj["0"] if "0" in obj else next(v for v in obj.values())
    return obj

# ---- load downsampled B (float32) ----
with h5py.File(path, "r") as f:
    Bx = open_var(f, "Bx")[::stride, ::stride, ::stride].astype(np.float32)
    By = open_var(f, "By")[::stride, ::stride, ::stride].astype(np.float32)
    Bz = open_var(f, "Bz")[::stride, ::stride, ::stride].astype(np.float32)

# shape and physical spacings of the downsampled grid
# Convention used throughout this script: arrays are (z, y, x) in C-order.
# i.e., B[z, y, x] with x varying fastest.
if order.upper() == "C":
    # Ensure contiguous arrays (FFT performance + predictable axis meaning)
    Bx = np.ascontiguousarray(Bx)
    By = np.ascontiguousarray(By)
    Bz = np.ascontiguousarray(Bz)
else:
    raise ValueError(f"Unsupported array order={order!r}. Expected 'C' for (z,y,x).")

nz, ny, nx = Bx.shape          # (z,y,x)
dx, dy, dz = DX*stride, DY*stride, DZ*stride

# Mean (background) field direction should be computed from the *undetrended* field
B0_undetrended = np.array([
    Bx.mean(dtype=np.float64),
    By.mean(dtype=np.float64),
    Bz.mean(dtype=np.float64),
], dtype=np.float64)

# ---- detrend + window (separable Hann) ----
def hann(n):  # 1D Hann
    return 0.5*(1 - np.cos(2*np.pi*np.arange(n)/(n-1))) if n > 1 else np.ones(1)

wx, wy, wz = hann(nx).astype(np.float32), hann(ny).astype(np.float32), hann(nz).astype(np.float32)
W = wz[:,None,None]*wy[None,:,None]*wx[None,None,:]  # (z,y,x)

def prep(comp):
    comp = comp - np.mean(comp, dtype=np.float64)
    return (comp * W).astype(np.float32)

Bxw, Byw, Bzw = prep(Bx), prep(By), prep(Bz)

# ---- FFTs ----
# Orthonormal FFTs so that sum |B|^2 == sum |F|^2 (Parseval)
Fb_x = rfftn(Bxw, axes=(0,1,2), norm='ortho')
Fb_y = rfftn(Byw, axes=(0,1,2), norm='ortho')
Fb_z = rfftn(Bzw, axes=(0,1,2), norm='ortho')

# compensate for Hann window energy loss
W2 = np.mean(W**2, dtype=np.float64)

# spectral power per mode ~ |F|^2, summed over components
P3 = (np.abs(Fb_x)**2 + np.abs(Fb_y)**2 + np.abs(Fb_z)**2).astype(np.float64) / W2

# rfftn stores only kx >= 0. For energy-consistent spectra we need to account for the
# missing negative-kx partners by doubling interior kx modes.
# Keep kx=0 and (if present) Nyquist (nx even) un-doubled.
kx_factor = np.ones(P3.shape[2], dtype=np.float64)
if kx_factor.size > 1:
    kx_factor[1:] = 2.0
    if nx % 2 == 0:
        # Nyquist exists at the last rfft index
        kx_factor[-1] = 1.0
P3 *= kx_factor[None, None, :]

# Wavenumber arrays (physical, rad/unit length)
kx = 2*np.pi * rfftfreq(nx, d=dx)         # length nx//2+1
ky = 2*np.pi * fftfreq(ny, d=dy)          # length ny
kz = 2*np.pi * fftfreq(nz, d=dz)          # length nz

Lx, Ly, Lz = nx*dx, ny*dy, nz*dz
# k-space cell volumes (for converting sums over modes to PSD densities)
Dkx = 2*np.pi / Lx
Dky = 2*np.pi / Ly
Dkz = 2*np.pi / Lz
Dk_cell = Dkx * Dky * Dkz

KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
Kmag = np.sqrt(KX**2 + KY**2 + KZ**2)

# ---- isotropic 1D spectrum: shell-average in |k| ----
# ignore the DC bin (k=0)
mask = Kmag > 0
kvals = Kmag[mask].ravel()
pvals = P3[mask].ravel()

# Choose log-spaced k-bins.
# Percentiles are robust to outliers, but ensure we don't skip the true smallest nonzero k.
kmin_p = np.percentile(kvals, 0.1)
kmax_p = np.percentile(kvals, 99.9)
kmin = max(kvals.min(), kmin_p)
kmax = min(kvals.max(), kmax_p)
kbins = np.logspace(np.log10(kmin), np.log10(kmax), 60)

# sum power in shells
shell_sum, _ = np.histogram(kvals, bins=kbins, weights=pvals)
kmid = np.sqrt(kbins[:-1]*kbins[1:])
Dk   = np.diff(kbins)
# physical shell volume in k-space ~ 4π k^2 Δk
shell_vol = 4*np.pi * (kmid**2) * Dk
# Convert per-mode power to spectral density using k-space cell volume
# so that ∫ E(k) dk ≈ total variance of B (apart from window shape factors)
E_iso = (shell_sum * Dk_cell) / np.maximum(shell_vol, 1e-300)

# Optional normalization:
# To approximate continuous E(k) (so that ∫E(k) dk ~ total energy),
# multiply by shell surface area and divide by box volume etc.
# For slope comparisons this average is usually sufficient.

# ---- reduced 1D spectra along axes ----
# Sum over the other two dimensions (note rfftn along x)
Ex_modes = np.sum(P3, axis=(0, 1))  # -> kx (len nx//2+1)
Ey_modes = np.sum(P3, axis=(0, 2))  # -> ky (len ny)
Ez_modes = np.sum(P3, axis=(1, 2))  # -> kz (len nz)

# Convert from per-mode sums to 1D PSDs by multiplying by the k-space area of the summed planes
# E1D(kx) ~ sum_{ky,kz} P * Δky Δkz ; similarly for ky,kz
Ex_psd = Ex_modes * (Dky * Dkz)
Ey_psd = Ey_modes * (Dkx * Dkz)
Ez_psd = Ez_modes * (Dkx * Dky)

# Fold ky,kz to positive k to avoid double-counting

def fold_to_positive(k, E):
    idx = np.argsort(k)
    k = k[idx]; E = E[idx]
    n = len(k)
    mid = n//2
    if n % 2 == 0:
        kpos = k[mid:]
        Epos = E[mid:].copy()
        Eneg = E[:mid][::-1]
        Epos[1:] += Eneg[:-1]  # add mirrored bins, skip DC
    else:
        kpos = k[mid:]
        Epos = E[mid:].copy()
        Eneg = E[:mid][::-1]
        Epos[1:] += Eneg
    return kpos, Epos

ky_pos, Ey_pos = fold_to_positive(ky, Ey_psd)
kz_pos, Ez_pos = fold_to_positive(kz, Ez_psd)

# ---- optional 2D E(k_perp, k_par) relative to mean B0 ----
E2D = None
if compute_kperp_kpar:
    # Mean-field direction must be computed from the *undetrended* field.
    B0 = B0_undetrended.copy()
    if np.linalg.norm(B0) < 1e-30:
        # Fall back to RMS direction if the mean is near-zero
        B0 = np.array([
            np.sqrt(np.mean(Bx.astype(np.float64)**2)),
            np.sqrt(np.mean(By.astype(np.float64)**2)),
            np.sqrt(np.mean(Bz.astype(np.float64)**2)),
        ], dtype=np.float64)
    bhat = B0 / (np.linalg.norm(B0) + 1e-30)

    # project K onto bhat -> k_par, and k_perp = sqrt(k^2 - k_par^2)
    kpar = (KX*bhat[0] + KY*bhat[1] + KZ*bhat[2])
    kperp = np.sqrt(np.maximum(0.0, Kmag**2 - kpar**2))

    km_perp = kperp[mask].ravel()
    km_par  = np.abs(kpar[mask].ravel())
    p2      = pvals

    # bin in log k_perp and linear (or log) k_par
    kperp_bins = np.logspace(np.log10(max(1e-6, km_perp.min())), np.log10(km_perp.max()), 60)
    kpar_bins  = np.linspace(0.0, km_par.max(), 60)
    H, _, _ = np.histogram2d(km_perp, km_par, bins=[kperp_bins, kpar_bins], weights=p2)
    E2D = (H, kperp_bins, kpar_bins)

# ---------------- plots ----------------
fig, axs = plt.subplots(1, 2, figsize=(11,4.3))

# Isotropic E(k)
axs[0].loglog(kmid, E_iso, label="isotropic")
# slope guides
for s, x0 in [(-5/3, 0.5), (-3/2, 0.5)]:
    idx0 = np.argmax(kmid > x0)
    y0 = E_iso[idx0 if idx0>0 else len(E_iso)//3]
    axs[0].plot(kmid, y0*(kmid/x0)**s, '--', alpha=0.4, label=f"k^{s:.2f}")
axs[0].set_xlabel("k")
axs[0].set_ylabel("E_B(k) (arb.)")
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# Reduced spectra (one-sided for ky,kz)
axs[1].loglog(kx, Ex_psd + 1e-300, label="E(kx)")
axs[1].loglog(ky_pos, Ey_pos + 1e-300, label="E(ky)")
axs[1].loglog(kz_pos, Ez_pos + 1e-300, label="E(kz)")
axs[1].set_xlabel("k (axis)")
axs[1].set_ylabel("reduced E(k)")
axs[1].legend()
axs[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Optional 2D map
if compute_kperp_kpar and E2D is not None:
    H, kperp_bins, kpar_bins = E2D
    plt.figure(figsize=(6,5))
    pcm = plt.pcolormesh(kperp_bins, kpar_bins, H.T, shading='auto', cmap='viridis', norm=plt.matplotlib.colors.LogNorm(vmin=max(1, H[H>0].min()), vmax=H.max()))
    plt.xscale('log')
    plt.xlabel(r'$k_\perp$')
    plt.ylabel(r'$k_\parallel$ (| |)')
    plt.colorbar(pcm, label='power (arb., log)')
    plt.tight_layout()
    plt.show()