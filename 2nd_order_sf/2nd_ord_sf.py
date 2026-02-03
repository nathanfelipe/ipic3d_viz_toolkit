#!/usr/bin/env python3
"""
Second-order structure function S2(r) for 3D ECsim outputs.

We compute
    S2_B(r) = ⟨ |B(x+r) - B(x)|^2 ⟩
    S2_u(r) = ⟨ |u(x+r) - u(x)|^2 ⟩
for separations r aligned with the grid axes (x, y, z), and also report an
"isotropic" estimate by averaging the three axial S2 curves at equal |r|.

Velocity u is read from Vx,Vy,Vz if available. If not, we try to build it from
species current J and density N via u_s = J_s / (q_s N_s). In normalized units
q_s is usually ±1; set q_sign accordingly for the chosen species.

This script is light on RAM and works on a downsampled grid (stride).
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------- helpers -------------------------

import h5py
def open_var(f: h5py.File, name: str):
    obj = f[f"Step#0/Block/{name}"]
    if isinstance(obj, h5py.Group):
        # prefer dataset named "0" if present
        if "0" in obj:
            return obj["0"]
        # otherwise first dataset within
        for k, v in obj.items():
            if isinstance(v, h5py.Dataset):
                return v
        raise KeyError(f"Group {name} has no datasets")
    return obj


def try_load_velocity(path: Path, species: int, stride: int):
    """Return (u_vec, used_from_current) where u_vec shape = (nz, ny, nx, 3).
    If Vx/Vy/Vz are absent, compute from J and N with q_sign.
    """
    sx = f"_{species}"
    with h5py.File(path, "r") as f:
        # Try direct velocity fields
        have_direct = all(
            (f"V{c}{sx}" in f["Step#0/Block"]) for c in ("x", "y", "z")
        )
        if have_direct:
            Vx = open_var(f, f"Vx{sx}")[::stride, ::stride, ::stride].astype(np.float32)
            Vy = open_var(f, f"Vy{sx}")[::stride, ::stride, ::stride].astype(np.float32)
            Vz = open_var(f, f"Vz{sx}")[::stride, ::stride, ::stride].astype(np.float32)
            u = np.stack([Vx, Vy, Vz], axis=-1)
            return u, False

        # Otherwise build from J and N
        Jx = open_var(f, f"Jx{sx}")[::stride, ::stride, ::stride].astype(np.float32)
        Jy = open_var(f, f"Jy{sx}")[::stride, ::stride, ::stride].astype(np.float32)
        Jz = open_var(f, f"Jz{sx}")[::stride, ::stride, ::stride].astype(np.float32)
        N  = open_var(f, f"N{sx}")[::stride, ::stride, ::stride].astype(np.float32)
    return (np.stack([Jx, Jy, Jz], axis=-1), True), N


def load_fields(path: Path, stride: int, order: str = "C"):
    """Load B-field (vector) downsampled. Returns B_vec with shape (nz,ny,nx,3)."""
    with h5py.File(path, "r") as f:
        Bx = open_var(f, "Bx")[::stride, ::stride, ::stride].astype(np.float32)
        By = open_var(f, "By")[::stride, ::stride, ::stride].astype(np.float32)
        Bz = open_var(f, "Bz")[::stride, ::stride, ::stride].astype(np.float32)
    B = np.stack([Bx, By, Bz], axis=-1)
    # C-order is correct for your data
    if order not in ("C", "F"):
        raise ValueError("order must be 'C' or 'F'")
    return B


def build_velocity(path: Path, species: int, stride: int, q_sign: float = 1.0):
    """Try to load velocity; if unavailable, compute from J and N.
    Returns u_vec (nz,ny,nx,3) and a boolean flag 'from_current'.
    """
    sx = f"_{species}"
    with h5py.File(path, "r") as f:
        have_direct = all(
            (f"V{c}{sx}" in f["Step#0/Block"]) for c in ("x", "y", "z")
        )
        if have_direct:
            Vx = open_var(f, f"Vx{sx}")[::stride, ::stride, ::stride].astype(np.float32)
            Vy = open_var(f, f"Vy{sx}")[::stride, ::stride, ::stride].astype(np.float32)
            Vz = open_var(f, f"Vz{sx}")[::stride, ::stride, ::stride].astype(np.float32)
            return np.stack([Vx, Vy, Vz], axis=-1), False

        # Fallback from J and N: u = J / (q N)
        Jx = open_var(f, f"Jx{sx}")[::stride, ::stride, ::stride].astype(np.float32)
        Jy = open_var(f, f"Jy{sx}")[::stride, ::stride, ::stride].astype(np.float32)
        Jz = open_var(f, f"Jz{sx}")[::stride, ::stride, ::stride].astype(np.float32)
        # N datasets may be named N_0/N_1 or rho_0/rho_1. Try both.
        if f"N{sx}" in f["Step#0/Block"]:
            N = open_var(f, f"N{sx}")[::stride, ::stride, ::stride].astype(np.float32)
        elif f"rho{sx}" in f["Step#0/Block"]:
            N = open_var(f, f"rho{sx}")[::stride, ::stride, ::stride].astype(np.float32)
        else:
            raise KeyError("Neither N_{species} nor rho_{species} found for density")

        eps = 1e-12
        inv = 1.0 / np.maximum(np.abs(q_sign) * np.maximum(N, eps), eps)
        Vx = Jx * inv
        Vy = Jy * inv
        Vz = Jz * inv
        return np.stack([Vx, Vy, Vz], axis=-1), True


def s2_axis(field_vec: np.ndarray, lag_steps: int, axis: int) -> float:
    """Second-order structure function for a *single* lag along one axis.
    field_vec: array (nz,ny,nx,3)
    lag_steps: integer >= 1
    axis: 0=z, 1=y, 2=x
    Returns scalar S2 for this lag.
    """
    if lag_steps <= 0:
        raise ValueError("lag_steps must be >= 1")
    slicer1 = [slice(None), slice(None), slice(None)]
    slicer2 = [slice(None), slice(None), slice(None)]
    slicer1[axis] = slice(0, -lag_steps)
    slicer2[axis] = slice(lag_steps, None)
    a = field_vec[tuple(slicer1)]
    b = field_vec[tuple(slicer2)]
    diff2 = np.sum((b - a) ** 2, axis=-1)  # |Δ|^2 over vector components
    return float(np.mean(diff2, dtype=np.float64))


def s4_axis(field_vec: np.ndarray, lag_steps: int, axis: int) -> float:
    """Fourth-order structure function for a single lag along one axis.
    Returns <|ΔF|^4> for vector field increments along the given axis."""
    if lag_steps <= 0:
        raise ValueError("lag_steps must be >= 1")
    slicer1 = [slice(None), slice(None), slice(None)]
    slicer2 = [slice(None), slice(None), slice(None)]
    slicer1[axis] = slice(0, -lag_steps)
    slicer2[axis] = slice(lag_steps, None)
    a = field_vec[tuple(slicer1)]
    b = field_vec[tuple(slicer2)]
    diff2 = np.sum((b - a) ** 2, axis=-1)  # |Δ|^2
    diff4 = diff2 ** 2                      # |Δ|^4
    return float(np.mean(diff4, dtype=np.float64))

# ------------------------- PDF helpers -------------------------
def sample_longitudinal_increments(field_vec: np.ndarray, lag_steps: int, axis: int, comp: int, max_samples: int = 200000) -> np.ndarray:
    """Return a 1D array of *signed* longitudinal component increments for a given
    spatial axis and matching component.
    field_vec shape: (nz, ny, nx, 3)
    axis: 0=z,1=y,2=x (spatial shift)
    comp: 0=x,1=y,2=z (vector component)
    We compute ΔF_comp = F_comp(x+e_axis*lag) - F_comp(x).
    Randomly subsample to at most max_samples to limit memory.
    """
    if lag_steps <= 0:
        raise ValueError("lag_steps must be >= 1")
    slicer1 = [slice(None), slice(None), slice(None)]
    slicer2 = [slice(None), slice(None), slice(None)]
    slicer1[axis] = slice(0, -lag_steps)
    slicer2[axis] = slice(lag_steps, None)
    a = field_vec[tuple(slicer1)][..., comp]
    b = field_vec[tuple(slicer2)][..., comp]
    d = (b - a).ravel(order='C')
    if d.size > max_samples:
        idx = np.random.choice(d.size, size=max_samples, replace=False)
        d = d[idx]
    return d

def build_pdf(data: np.ndarray, bins: int = 120, max_sigma: float = 6.0):
    """Normalize data by its standard deviation and return (centers, pdf).
    The histogram is computed over [-max_sigma, max_sigma] in units of σ.
    """
    data = data[np.isfinite(data)]
    if data.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])
    sigma = np.std(data)
    if sigma == 0:
        sigma = 1.0
    x = data / sigma
    edges = np.linspace(-max_sigma, max_sigma, bins + 1)
    hist, edges = np.histogram(x, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def make_axis_lags(max_lag_cells: int):
    """Return dictionary of lag arrays per axis: {axis: [1..max]}"""
    lags = np.arange(1, max_lag_cells + 1, dtype=int)
    return {0: lags.copy(), 1: lags.copy(), 2: lags.copy()}


def isotropic_average(Sz, Sy, Sx, dz, dy, dx):
    """Average S2 at equal |r| by linear interpolation between axial samples.
    Returns r_vals (meters/units) and S2_iso.
    """
    rz = dz * np.arange(1, len(Sz) + 1)
    ry = dy * np.arange(1, len(Sy) + 1)
    rx = dx * np.arange(1, len(Sx) + 1)
    r_min = max(rz.min(), ry.min(), rx.min())
    r_max = min(rz.max(), ry.max(), rx.max())
    n = min(len(Sz), len(Sy), len(Sx))
    r_common = np.linspace(r_min, r_max, n)

    def interp(r_src, S_src, r_tgt):
        return np.interp(r_tgt, r_src, S_src)

    Sz_i = interp(rz, Sz, r_common)
    Sy_i = interp(ry, Sy, r_common)
    Sx_i = interp(rx, Sx, r_common)
    S_iso = (Sz_i + Sy_i + Sx_i) / 3.0
    return r_common, S_iso


def isotropic_flatness(S2z, S2y, S2x, S4z, S4y, S4x, dz, dy, dx):
    """Interpolate S2 and S4 to a common r and compute F = S4/(S2^2)."""
    rz = dz * np.arange(1, len(S2z) + 1)
    ry = dy * np.arange(1, len(S2y) + 1)
    rx = dx * np.arange(1, len(S2x) + 1)
    r_min = max(rz.min(), ry.min(), rx.min())
    r_max = min(rz.max(), ry.max(), rx.max())
    n = min(len(S2z), len(S2y), len(S2x))
    r_common = np.linspace(r_min, r_max, n)

    def interp(r_src, S_src):
        return np.interp(r_common, r_src, S_src)

    S2z_i, S2y_i, S2x_i = interp(rz, S2z), interp(ry, S2y), interp(rx, S2x)
    S4z_i, S4y_i, S4x_i = interp(rz, S4z), interp(ry, S4y), interp(rx, S4x)
    S2_iso = (S2z_i + S2y_i + S2x_i) / 3.0
    S4_iso = (S4z_i + S4y_i + S4x_i) / 3.0
    eps = 1e-30
    F_iso = S4_iso / np.maximum(S2_iso**2, eps)
    return r_common, F_iso


# ------------------------- main -------------------------

def main():
    p = argparse.ArgumentParser(description="Second-order structure function for ECsim 3D data")
    p.add_argument("path", type=Path, default="../../DoubleGEM-Fields_300000.h5", help="HDF5 file, e.g., T3D-Fields_000500.h5")
    p.add_argument("--species", type=int, default=1, help="species index for velocity (1=ions, 0=electrons)")
    p.add_argument("--qsign", type=float, default=1.0, help="charge sign used only if building u from J/N (ions +1, electrons -1)")
    p.add_argument("--stride", type=int, default=2, help="downsample stride (>=1)")
    p.add_argument("--maxlag", type=int, default=64, help="maximum lag in cells along each axis")
    p.add_argument("--DX", type=float, default=0.125, help="grid spacing in x (simulation units)")
    p.add_argument("--DY", type=float, default=0.125, help="grid spacing in y")
    p.add_argument("--DZ", type=float, default=0.125, help="grid spacing in z")
    p.add_argument("--order", type=str, default="C", choices=["C", "F"], help="memory order assumption for ravel/reshapes (data are C-ordered)")
    p.add_argument("--noiso", action="store_true", help="do not compute isotropic average; only axial S2")
    p.add_argument("--metric", choices=["structure_function", "flatness"], default="structure_function",
                   help="select to plot S2 (structure_function) or flatness (S4/S2^2)")
    p.add_argument("--B0z", type=float, default=0.01,
                   help="Guide-field B0 along z in code units (for gyro scales; default from SimulationParameters)")
    p.add_argument("--qom_ion", type=float, default=1.0,
                   help="ion charge-to-mass ratio q/m in code units (QOM for species 1; default 1.0)")
    p.add_argument("--annotate_scales", action="store_true",
                   help="annotate r = d_i and r = rho_i (median) on both panels")
    p.add_argument("--pdf", action="store_true", help="also compute PDFs of longitudinal increments at selected r")
    p.add_argument("--pdf_fields", choices=["B","u","both"], default="both", help="which field(s) to plot PDFs for")
    p.add_argument("--pdf_r", type=str, default="", help="comma-separated r values in d_i units; if empty use [0.5*rhoi, rhoi, 2*rhoi, 2.0]")
    p.add_argument("--pdf_bins", type=int, default=120, help="number of bins for PDFs")
    p.add_argument("--pdf_maxsigma", type=float, default=6.0, help="range of PDF in units of sigma")
    p.add_argument("--pdf_samples", type=int, default=250000, help="max total samples per r per field")
    args = p.parse_args()

    path = args.path
    stride = max(1, args.stride)

    # Load fields
    B = load_fields(path, stride=stride, order=args.order)
    u, from_current = build_velocity(path, species=args.species, stride=stride, q_sign=args.qsign)

    # Load ion density and pressure to estimate thermal gyroradius (species 1)
    with h5py.File(path, "r") as f:
        # density (species 1): try N_1 then rho_1
        if "N_1" in f["Step#0/Block"]:
            Ni = open_var(f, "N_1")[::stride, ::stride, ::stride].astype(np.float32)
        else:
            Ni = open_var(f, "rho_1")[::stride, ::stride, ::stride].astype(np.float32)
        # perpendicular pressure estimate: average of Pxx_1 and Pyy_1 if available; fallback to Pxx_1
        if all(k in f["Step#0/Block"] for k in ("Pxx_1", "Pyy_1")):
            Pperp_i = 0.5*(open_var(f, "Pxx_1")[::stride, ::stride, ::stride].astype(np.float32) +
                           open_var(f, "Pyy_1")[::stride, ::stride, ::stride].astype(np.float32))
        else:
            Pperp_i = open_var(f, "Pxx_1")[::stride, ::stride, ::stride].astype(np.float32)
    # Medians to suppress outliers/zeros
    Ni_med = float(np.median(Ni))
    Pperp_med = float(np.median(Pperp_i))

    # In ECsim normalization, d_i is typically the unit of length -> take d_i = 1.0 in code units
    d_i = 1.0

    # Local beta_i,perp from downsampled fields: beta = 2 P_perp / B^2
    B2 = np.sum(B**2, axis=-1)
    eps = 1e-20
    beta_i_perp_local = 2.0 * Pperp_i / np.maximum(B2, eps)
    # Convert to gyro-radius in d_i units: rho_i/d_i = sqrt(beta/2)
    rho_i_local = np.sqrt(np.maximum(beta_i_perp_local, 0.0) / 2.0) * d_i

    # Robust summary stats for annotation
    rho_i_p25 = float(np.nanpercentile(rho_i_local, 25))
    rho_i_p50 = float(np.nanpercentile(rho_i_local, 50))  # median
    rho_i_p75 = float(np.nanpercentile(rho_i_local, 75))

    # B0-based estimate (using median P_perp and uniform B0)
    B0 = abs(args.B0z)
    beta_i_perp_B0_median = 2.0 * Pperp_med / max(B0*B0, eps)
    rho_i_B0 = np.sqrt(max(beta_i_perp_B0_median, 0.0) / 2.0) * d_i

    print(
        "[scales] d_i = %.4g  |  rho_i(local): p25=%.4g, med=%.4g, p75=%.4g  |  rho_i(B0, med P_perp)=%.4g"
        % (d_i, rho_i_p25, rho_i_p50, rho_i_p75, rho_i_B0)
    )

    nz, ny, nx, _ = B.shape
    print(f"grid (downsampled): nx={nx}, ny={ny}, nz={nz}; stride={stride}")

    # Make sure maxlag is feasible along each axis
    maxlag = args.maxlag
    maxlag_z = min(maxlag, nz - 1)
    maxlag_y = min(maxlag, ny - 1)
    maxlag_x = min(maxlag, nx - 1)
    if maxlag_z < 1 or maxlag_y < 1 or maxlag_x < 1:
        raise ValueError("maxlag too large for the current downsampled grid")

    # Compute axial S2 for B and u
    lags_z = np.arange(1, maxlag_z + 1, dtype=int)
    lags_y = np.arange(1, maxlag_y + 1, dtype=int)
    lags_x = np.arange(1, maxlag_x + 1, dtype=int)

    S2B_z = np.array([s2_axis(B, l, axis=0) for l in lags_z], dtype=np.float64)
    S2B_y = np.array([s2_axis(B, l, axis=1) for l in lags_y], dtype=np.float64)
    S2B_x = np.array([s2_axis(B, l, axis=2) for l in lags_x], dtype=np.float64)

    S2u_z = np.array([s2_axis(u, l, axis=0) for l in lags_z], dtype=np.float64)
    S2u_y = np.array([s2_axis(u, l, axis=1) for l in lags_y], dtype=np.float64)
    S2u_x = np.array([s2_axis(u, l, axis=2) for l in lags_x], dtype=np.float64)

    # If flatness is requested, compute 4th-order structure functions
    if args.metric == "flatness":
        S4B_z = np.array([s4_axis(B, l, axis=0) for l in lags_z], dtype=np.float64)
        S4B_y = np.array([s4_axis(B, l, axis=1) for l in lags_y], dtype=np.float64)
        S4B_x = np.array([s4_axis(B, l, axis=2) for l in lags_x], dtype=np.float64)

        S4u_z = np.array([s4_axis(u, l, axis=0) for l in lags_z], dtype=np.float64)
        S4u_y = np.array([s4_axis(u, l, axis=1) for l in lags_y], dtype=np.float64)
        S4u_x = np.array([s4_axis(u, l, axis=2) for l in lags_x], dtype=np.float64)

        eps = 1e-30
        FB_z = S4B_z / np.maximum(S2B_z**2, eps)
        FB_y = S4B_y / np.maximum(S2B_y**2, eps)
        FB_x = S4B_x / np.maximum(S2B_x**2, eps)

        Fu_z = S4u_z / np.maximum(S2u_z**2, eps)
        Fu_y = S4u_y / np.maximum(S2u_y**2, eps)
        Fu_x = S4u_x / np.maximum(S2u_x**2, eps)

    # Physical r for each axis (use downsampled spacings)
    dx = args.DX * stride
    dy = args.DY * stride
    dz = args.DZ * stride
    rz = dz * lags_z
    ry = dy * lags_y
    rx = dx * lags_x

    # Optional isotropic average (simple axial average at equal |r|)
    if not args.noiso:
        r_iso_B, S2B_iso = isotropic_average(S2B_z, S2B_y, S2B_x, dz, dy, dx)
        r_iso_u, S2u_iso = isotropic_average(S2u_z, S2u_y, S2u_x, dz, dy, dx)
        if args.metric == "flatness":
            rF_B, FB_iso = isotropic_flatness(S2B_z, S2B_y, S2B_x, S4B_z, S4B_y, S4B_x, dz, dy, dx)
            rF_u, Fu_iso = isotropic_flatness(S2u_z, S2u_y, S2u_x, S4u_z, S4u_y, S4u_x, dz, dy, dx)

    # ------------------------- PDFs of increments (optional) -------------------------
    if args.pdf:
        # choose r targets
        if args.pdf_r.strip():
            r_targets = [float(v) for v in args.pdf_r.split(',') if v.strip()]
        else:
            r_targets = [0.5*rho_i_p50, rho_i_p50, 2.0*rho_i_p50, 2.0]  # in d_i units
        # map desired r to integer lag per axis
        def r_to_lag(r, d):
            return max(1, int(round(r / d)))
        # longitudinal pairing: (axis, comp) = (x→2,0), (y→1,1), (z→0,2)
        pairs = [(2,0),(1,1),(0,2)]

        want_B = args.pdf_fields in ("B","both")
        want_u = args.pdf_fields in ("u","both")

        # Prepare figure rows depending on fields
        nrows = (1 if want_B else 0) + (1 if want_u else 0)
        fig_pdf, axes_pdf = plt.subplots(nrows, 1, figsize=(7, 3.2*nrows), sharex=True)
        if nrows == 1:
            axes_pdf = [axes_pdf]
        row = 0

        def plot_field_PDFs(field, name):
            nonlocal row
            axp = axes_pdf[row]
            for r in r_targets:
                lags = [r_to_lag(r, d) for d in (dx, dy, dz)]
                # accumulate longitudinal component increments across axes
                data_all = []
                total_budget = args.pdf_samples
                per_axis = max(1, total_budget // len(pairs))
                for (axis, comp), lag in zip(pairs, lags[::-1]):
                    # Note: lags list corresponds to (dx,dy,dz) but axis order pairs are (x,y,z)->(2,1,0)
                    # We reversed to align: dx->axis2, dy->axis1, dz->axis0
                    d = sample_longitudinal_increments(field, lag, axis=axis, comp=comp, max_samples=per_axis)
                    data_all.append(d)
                data_all = np.concatenate(data_all) if data_all else np.array([])
                c, pdf = build_pdf(data_all, bins=args.pdf_bins, max_sigma=args.pdf_maxsigma)
                axp.semilogy(c, pdf, label=f"r={r:.2g} d_i")
            # Gaussian reference
            xg = np.linspace(-args.pdf_maxsigma, args.pdf_maxsigma, 400)
            yg = 1/np.sqrt(2*np.pi) * np.exp(-0.5*xg**2)
            axp.semilogy(xg, yg, 'k--', alpha=0.6, label="Gaussian")
            axp.set_ylabel(f"PDF of Δ{name}/σ")
            axp.grid(True, which='both', alpha=0.3)
            axp.legend(frameon=False)
            row += 1

        if want_B:
            plot_field_PDFs(B, 'B')
        if want_u:
            plot_field_PDFs(u, 'u')
        axes_pdf[-1].set_xlabel("Δ/σ (longitudinal component)")
        fig_pdf.suptitle("Increment PDFs at selected r")
        plt.tight_layout()

    # ------------------------- plots -------------------------
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Helper to draw scale annotations on an axis
    def annotate_scales(ax):
        if not args.annotate_scales:
            return
        ymin = ax.get_ylim()[0]
        # d_i line
        ax.axvline(d_i, color='k', ls=':', lw=1.5, alpha=0.8)
        ax.text(d_i, ymin*1.2, r"$d_i$", rotation=90, va='bottom', ha='center', fontsize=9)
        # rho_i from local beta: shaded IQR + median line
        ax.axvspan(rho_i_p25, rho_i_p75, facecolor='tab:red', alpha=0.12, lw=0)
        ax.axvline(rho_i_p50, color='tab:red', ls='-', lw=1.8, alpha=0.9)
        ax.text(rho_i_p50, ymin*1.2, r"$\rho_i$ (local, med)", rotation=90, va='bottom', ha='center', color='tab:red', fontsize=9)
        # rho_i using B0 only (median P_perp)
        ax.axvline(rho_i_B0, color='tab:red', ls='--', lw=1.5, alpha=0.8)
        ax.text(rho_i_B0, ymin*1.2, r"$\rho_i$ ($B_0$)", rotation=90, va='bottom', ha='center', color='tab:red', fontsize=9)

    if args.metric == "structure_function":
        # B field S2
        ax = axs[0]
        ax.loglog(rx, S2B_x, label="x", alpha=0.9)
        ax.loglog(ry, S2B_y, label="y", alpha=0.9)
        ax.loglog(rz, S2B_z, label="z", alpha=0.9)
        if not args.noiso:
            ax.loglog(r_iso_B, S2B_iso, 'k', lw=2.0, label="iso avg")
        # reference 2/3 slope guide
        if len(rx) > 5:
            r0 = rx[len(rx)//3]
            y0 = S2B_x[len(rx)//3]
            guide = y0 * (r_iso_B / r0) ** (2/3) if not args.noiso else y0 * (r_iso_B / r0) ** (2/3)
            ax.loglog(r_iso_B if not args.noiso else r_iso_B, guide, '--', color='gray', alpha=0.6, label=r"$r^{2/3}$")
        ax.set_title("S$_2$(r) for |B|")
        ax.set_xlabel("r (d_i(=1.0) units)")
        ax.set_ylabel(r"$\langle |\Delta B|^2 \rangle$")
        ax.grid(True, which='both', alpha=0.3)
        annotate_scales(ax)
        ax.legend(frameon=False)

        # Velocity S2
        ax = axs[1]
        ax.loglog(rx, S2u_x, label="x", alpha=0.9)
        ax.loglog(ry, S2u_y, label="y", alpha=0.9)
        ax.loglog(rz, S2u_z, label="z", alpha=0.9)
        if not args.noiso:
            ax.loglog(r_iso_u, S2u_iso, 'k', lw=2.0, label="iso avg")
        if len(rx) > 5:
            r0 = rx[len(rx)//3]
            y0 = S2u_x[len(rx)//3]
            guide = y0 * (r_iso_u / r0) ** (2/3) if not args.noiso else y0 * (r_iso_u / r0) ** (2/3)
            ax.loglog(r_iso_u if not args.noiso else r_iso_u, guide, '--', color='gray', alpha=0.6, label=r"$r^{2/3}$")
        label_u = "u from V" if not from_current else "u from J/N"
        ax.set_title(f"S$_2$(r) for u ({label_u}, species={args.species})")
        ax.set_xlabel("r (d_i(=1.0) units)")
        ax.set_ylabel(r"$\langle |\Delta u|^2 \rangle$")
        ax.grid(True, which='both', alpha=0.3)
        annotate_scales(ax)
        ax.legend(frameon=False)

    else:  # flatness
        # B field flatness
        ax = axs[0]
        ax.loglog(rx, FB_x, label="x", alpha=0.9)
        ax.loglog(ry, FB_y, label="y", alpha=0.9)
        ax.loglog(rz, FB_z, label="z", alpha=0.9)
        if not args.noiso:
            ax.loglog(rF_B, FB_iso, 'k', lw=2.0, label="iso avg")
        ax.set_title("Flatness F(r) for |B|")
        ax.set_xlabel("r (d_i units)")
        ax.set_ylabel(r"$F = S_4/S_2^2$")
        ax.grid(True, which='both', alpha=0.3)
        annotate_scales(ax)
        ax.legend(frameon=False)

        # Velocity flatness
        ax = axs[1]
        ax.loglog(rx, Fu_x, label="x", alpha=0.9)
        ax.loglog(ry, Fu_y, label="y", alpha=0.9)
        ax.loglog(rz, Fu_z, label="z", alpha=0.9)
        if not args.noiso:
            ax.loglog(rF_u, Fu_iso, 'k', lw=2.0, label="iso avg")
        label_u = "u from V" if not from_current else "u from J/N"
        ax.set_title(f"Flatness F(r) for u ({label_u}, species={args.species})")
        ax.set_xlabel("r (d_i(=1.0) units)")
        ax.set_ylabel(r"$F = S_4/S_2^2$")
        ax.grid(True, which='both', alpha=0.3)
        annotate_scales(ax)
        ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()