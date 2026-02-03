import h5py, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

# ===== user inputs =====
path = "../../DoubleGEM-Fields_300000.h5"

# dataset names (exact, if you know them)
rho_name = "rho_0"
bx_name, by_name, bz_name = "Bx", "By", "Bz"
ex_name, ey_name, ez_name = "Ex", "Ey", "Ez"   # skipped if not present

# geometry (SimulationData.txt)
DX, DY, DZ = 0.125, 0.125, 0.125
X0, Y0, Z0 = 0.0, 0.0, 0.0

# probe path
p0 = np.array([1.0,  2.0,  0.5])      # start (x,y,z)
p1 = np.array([12.0, 14.0, 20.0])     # end   (x,y,z)
n_samples = 2001

# MMS-like virtual tetrahedron (4 probes)
n_probes = 4
MMS_separation = 0.5   # approximate inter-spacecraft separation in simulation units
probe_to_plot = 2      # which probe (1..4) to plot in the figures

# x-axis like your figure (set di or leave None to use distance)
ion_inertial_length_di = None   # e.g., 0.2
highlight_window = None         # e.g., (7.2, 8.8) in same x-units

n_flybys = 5
out_dir_csv = Path("flyby_csvs")
out_dir_figs = Path("flyby_figs")

# ========================

# Utility: list all dataset names in the HDF5 file


def list_all_fields(h5path):
    print("\n=== Available fields in file ===")
    with h5py.File(h5path, "r") as f:
        def recurse(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(name)
        f.visititems(recurse)
    print("=== End of list ===\n")

list_all_fields(path)

# ensure output directories exist for batch flybys
out_dir_csv.mkdir(parents=True, exist_ok=True)
out_dir_figs.mkdir(parents=True, exist_ok=True)

def open_var(f, name):
    obj = f[f"Step#0/Block/{name}"]
    if isinstance(obj, h5py.Group):
        return obj["0"] if "0" in obj else next(v for v in obj.values() if isinstance(v, h5py.Dataset))
    return obj

def dataset_exists(f, name):
    try:
        _ = open_var(f, name); return True
    except Exception:
        return False

def find_first_available(f, aliases):
    for n in aliases:
        if dataset_exists(f, n):
            return open_var(f, n), n
    return None, None

def trilerp(arr, xyz, spacing, origin):
    DX, DY, DZ = spacing
    X0, Y0, Z0 = origin
    nz, ny, nx = arr.shape
    x = (xyz[..., 0] - X0) / DX
    y = (xyz[..., 1] - Y0) / DY
    z = (xyz[..., 2] - Z0) / DZ
    x0 = np.floor(x).astype(int); x1 = x0 + 1
    y0 = np.floor(y).astype(int); y1 = y0 + 1
    z0 = np.floor(z).astype(int); z1 = z0 + 1
    x0 = np.clip(x0, 0, nx-1); x1 = np.clip(x1, 0, nx-1)
    y0 = np.clip(y0, 0, ny-1); y1 = np.clip(y1, 0, ny-1)
    z0 = np.clip(z0, 0, nz-1); z1 = np.clip(z1, 0, nz-1)
    fx = np.clip(x - x0, 0, 1); fy = np.clip(y - y0, 0, 1); fz = np.clip(z - z0, 0, 1)
    c000 = arr[z0, y0, x0]; c100 = arr[z0, y0, x1]
    c010 = arr[z0, y1, x0]; c110 = arr[z0, y1, x1]
    c001 = arr[z1, y0, x0]; c101 = arr[z1, y0, x1]
    c011 = arr[z1, y1, x0]; c111 = arr[z1, y1, x1]
    c00 = c000*(1-fx) + c100*fx; c01 = c001*(1-fx) + c101*fx
    c10 = c010*(1-fx) + c110*fx; c11 = c011*(1-fx) + c111*fx
    c0 = c00*(1-fy) + c10*fy;    c1 = c01*(1-fy) + c11*fy
    return c0*(1-fz) + c1*fz


#
# ---- load fields ----
with h5py.File(path, "r") as f:
    # densities (ions = species 0, electrons = species 1)
    rho0 = np.asarray(open_var(f, rho_name), dtype=np.float32)        # rho_0
    rho1 = np.asarray(open_var(f, "rho_1"),    dtype=np.float32)      # rho_1

    # magnetic field
    Bx  = np.asarray(open_var(f, bx_name),  dtype=np.float32)
    By  = np.asarray(open_var(f, by_name),  dtype=np.float32)
    Bz  = np.asarray(open_var(f, bz_name),  dtype=np.float32)

    Ex_ds, _ = find_first_available(f, [ex_name, "ex", "E_x", "Ex_", "Elec_x"])
    Ey_ds, _ = find_first_available(f, [ey_name, "ey", "E_y", "Ey_", "Elec_y"])
    Ez_ds, _ = find_first_available(f, [ez_name, "ez", "E_z", "Ez_", "Elec_z"])

    # Velocity aliases (add any others you use)
    Vx_ds, vx_used = find_first_available(f, ["Vx","vx","Ux","ux","ve_x","Vi_x","vel_x"])
    Vy_ds, vy_used = find_first_available(f, ["Vy","vy","Uy","uy","ve_y","Vi_y","vel_y"])
    Vz_ds, vz_used = find_first_available(f, ["Vz","vz","Uz","uz","ve_z","Vi_z","vel_z"])

    # convert datasets to numpy arrays while file is still open
    Ex = np.asarray(Ex_ds, dtype=np.float32) if Ex_ds is not None else None
    Ey = np.asarray(Ey_ds, dtype=np.float32) if Ey_ds is not None else None
    Ez = np.asarray(Ez_ds, dtype=np.float32) if Ez_ds is not None else None
    Vx = np.asarray(Vx_ds, dtype=np.float32) if Vx_ds is not None else None
    Vy = np.asarray(Vy_ds, dtype=np.float32) if Vy_ds is not None else None
    Vz = np.asarray(Vz_ds, dtype=np.float32) if Vz_ds is not None else None

    # --- currents for ions (0) and electrons (1) ---
    Jx0 = np.asarray(open_var(f, "Jx_0"), dtype=np.float32)
    Jy0 = np.asarray(open_var(f, "Jy_0"), dtype=np.float32)
    Jz0 = np.asarray(open_var(f, "Jz_0"), dtype=np.float32)

    Jx1 = np.asarray(open_var(f, "Jx_1"), dtype=np.float32)
    Jy1 = np.asarray(open_var(f, "Jy_1"), dtype=np.float32)
    Jz1 = np.asarray(open_var(f, "Jz_1"), dtype=np.float32)

    # --- pressure tensor components for ions (0) and electrons (1) ---
    Pxx0 = np.asarray(open_var(f, "Pxx_0"), dtype=np.float32)
    Pxy0 = np.asarray(open_var(f, "Pxy_0"), dtype=np.float32)
    Pxz0 = np.asarray(open_var(f, "Pxz_0"), dtype=np.float32)
    Pyy0 = np.asarray(open_var(f, "Pyy_0"), dtype=np.float32)
    Pyz0 = np.asarray(open_var(f, "Pyz_0"), dtype=np.float32)
    Pzz0 = np.asarray(open_var(f, "Pzz_0"), dtype=np.float32)

    Pxx1 = np.asarray(open_var(f, "Pxx_1"), dtype=np.float32)
    Pxy1 = np.asarray(open_var(f, "Pxy_1"), dtype=np.float32)
    Pxz1 = np.asarray(open_var(f, "Pxz_1"), dtype=np.float32)
    Pyy1 = np.asarray(open_var(f, "Pyy_1"), dtype=np.float32)
    Pyz1 = np.asarray(open_var(f, "Pyz_1"), dtype=np.float32)
    Pzz1 = np.asarray(open_var(f, "Pzz_1"), dtype=np.float32)

    # --- compute velocities on the grid ---
    eps = 1e-12

    # species 0: electrons (q0 < 0), rho0 = q0 * n_e < 0
    ve_x = Jx0 / (rho0 + eps)
    ve_y = Jy0 / (rho0 + eps)
    ve_z = Jz0 / (rho0 + eps)

    # species 1: ions (q1 > 0), rho1 = q1 * n_i > 0
    vi_x = Jx1 / (rho1 + eps)
    vi_y = Jy1 / (rho1 + eps)
    vi_z = Jz1 / (rho1 + eps)

    # number densities in code units (assuming |q_i| = |q_e| = 1)
    n_e =  rho1
    n_i =  rho0

    # bulk (barycentric) velocity by number density
    vb_x = (n_i * vi_x + n_e * ve_x) / (n_i + n_e + eps)
    vb_y = (n_i * vi_y + n_e * ve_y) / (n_i + n_e + eps)
    vb_z = (n_i * vi_z + n_e * ve_z) / (n_i + n_e + eps)

# infer simulation box extents from rho0 shape and grid spacing
nz, ny, nx = rho0.shape
BOX_X = X0 + DX * nx
BOX_Y = Y0 + DY * ny
BOX_Z = Z0 + DZ * nz

# random generator for multiple flybys
rng = np.random.default_rng(12345)  # fixed seed for reproducibility

# MMS-like tetrahedron offsets (centered at 0, scaled to MMS_separation)
base_offsets = np.array([
    [ 1.0,  1.0,  1.0],
    [ 1.0, -1.0, -1.0],
    [-1.0,  1.0, -1.0],
    [-1.0, -1.0,  1.0],
], dtype=float)
scale = MMS_separation / (2.0 * np.sqrt(2.0))  # so that edge lengths ≈ MMS_separation
offsets = base_offsets * scale  # shape (4, 3)
n_probes = offsets.shape[0]

# ---- multi-flyby loop ----
for k in range(n_flybys):
    # randomly sample new p0 and p1 inside the box
    p0 = np.array([rng.uniform(X0, BOX_X), rng.uniform(Y0, BOX_Y), rng.uniform(Z0, BOX_Z)], dtype=float)
    p1 = np.array([rng.uniform(X0, BOX_X), rng.uniform(Y0, BOX_Y), rng.uniform(Z0, BOX_Z)], dtype=float)

    # ---- path & x-axis ----
    t = np.linspace(0.0, 1.0, n_samples)

    # barycenter path (common to all probes)
    pts = (1.0 - t)[:, None] * p0[None, :] + t[:, None] * p1[None, :]
    dist = np.linalg.norm(pts - pts[0], axis=1)

    # positions of each probe along the path: shape (n_probes, n_samples, 3)
    pts_probes = pts[None, :, :] + offsets[:, None, :]

    # define x-axis for plotting
    if ion_inertial_length_di is not None:
        x_axis = dist / ion_inertial_length_di
        x_label = r"$x/d_i$"
    else:
        x_axis = dist
        x_label = "Distance along path"

    # ---- interpolate ----
    spacing = (DX, DY, DZ); origin = (X0, Y0, Z0)

    # allocate arrays for all probes: shape (n_probes, n_samples)
    rho0_p = np.empty((n_probes, n_samples), dtype=np.float32)
    rho1_p = np.empty((n_probes, n_samples), dtype=np.float32)
    Bx_p   = np.empty((n_probes, n_samples), dtype=np.float32)
    By_p   = np.empty((n_probes, n_samples), dtype=np.float32)
    Bz_p   = np.empty((n_probes, n_samples), dtype=np.float32)
    Bmag_p = np.empty((n_probes, n_samples), dtype=np.float32)

    for i in range(n_probes):
        rho0_p[i] = trilerp(rho0, pts_probes[i], spacing, origin)
        rho1_p[i] = trilerp(rho1, pts_probes[i], spacing, origin)
        Bx_p[i]   = trilerp(Bx,  pts_probes[i], spacing, origin)
        By_p[i]   = trilerp(By,  pts_probes[i], spacing, origin)
        Bz_p[i]   = trilerp(Bz,  pts_probes[i], spacing, origin)
        Bmag_p[i] = np.sqrt(Bx_p[i]**2 + By_p[i]**2 + Bz_p[i]**2)

    Ex_p = Ey_p = Ez_p = Emag_p = None
    if (Ex is not None) and (Ey is not None) and (Ez is not None):
        Ex_p    = np.empty((n_probes, n_samples), dtype=np.float32)
        Ey_p    = np.empty((n_probes, n_samples), dtype=np.float32)
        Ez_p    = np.empty((n_probes, n_samples), dtype=np.float32)
        Emag_p  = np.empty((n_probes, n_samples), dtype=np.float32)
        for i in range(n_probes):
            Ex_p[i]   = trilerp(Ex, pts_probes[i], spacing, origin)
            Ey_p[i]   = trilerp(Ey, pts_probes[i], spacing, origin)
            Ez_p[i]   = trilerp(Ez, pts_probes[i], spacing, origin)
            Emag_p[i] = np.sqrt(Ex_p[i]**2 + Ey_p[i]**2 + Ez_p[i]**2)

    # ion velocity along the path
    vi_x_p = np.empty((n_probes, n_samples), dtype=np.float32)
    vi_y_p = np.empty((n_probes, n_samples), dtype=np.float32)
    vi_z_p = np.empty((n_probes, n_samples), dtype=np.float32)

    # electron velocity along the path
    ve_x_p = np.empty((n_probes, n_samples), dtype=np.float32)
    ve_y_p = np.empty((n_probes, n_samples), dtype=np.float32)
    ve_z_p = np.empty((n_probes, n_samples), dtype=np.float32)

    # bulk velocity along the path
    vb_x_p = np.empty((n_probes, n_samples), dtype=np.float32)
    vb_y_p = np.empty((n_probes, n_samples), dtype=np.float32)
    vb_z_p = np.empty((n_probes, n_samples), dtype=np.float32)

    # pressure tensor along the path (species 0: ions)
    Pxx0_p = np.empty((n_probes, n_samples), dtype=np.float32)
    Pxy0_p = np.empty((n_probes, n_samples), dtype=np.float32)
    Pxz0_p = np.empty((n_probes, n_samples), dtype=np.float32)
    Pyy0_p = np.empty((n_probes, n_samples), dtype=np.float32)
    Pyz0_p = np.empty((n_probes, n_samples), dtype=np.float32)
    Pzz0_p = np.empty((n_probes, n_samples), dtype=np.float32)

    # pressure tensor along the path (species 1: electrons)
    Pxx1_p = np.empty((n_probes, n_samples), dtype=np.float32)
    Pxy1_p = np.empty((n_probes, n_samples), dtype=np.float32)
    Pxz1_p = np.empty((n_probes, n_samples), dtype=np.float32)
    Pyy1_p = np.empty((n_probes, n_samples), dtype=np.float32)
    Pyz1_p = np.empty((n_probes, n_samples), dtype=np.float32)
    Pzz1_p = np.empty((n_probes, n_samples), dtype=np.float32)

    for i in range(n_probes):
        vi_x_p[i] = trilerp(vi_x, pts_probes[i], spacing, origin)
        vi_y_p[i] = trilerp(vi_y, pts_probes[i], spacing, origin)
        vi_z_p[i] = trilerp(vi_z, pts_probes[i], spacing, origin)

        ve_x_p[i] = trilerp(ve_x, pts_probes[i], spacing, origin)
        ve_y_p[i] = trilerp(ve_y, pts_probes[i], spacing, origin)
        ve_z_p[i] = trilerp(ve_z, pts_probes[i], spacing, origin)

        vb_x_p[i] = trilerp(vb_x, pts_probes[i], spacing, origin)
        vb_y_p[i] = trilerp(vb_y, pts_probes[i], spacing, origin)
        vb_z_p[i] = trilerp(vb_z, pts_probes[i], spacing, origin)

        # ion pressure tensor components along the path
        Pxx0_p[i] = trilerp(Pxx0, pts_probes[i], spacing, origin)
        Pxy0_p[i] = trilerp(Pxy0, pts_probes[i], spacing, origin)
        Pxz0_p[i] = trilerp(Pxz0, pts_probes[i], spacing, origin)
        Pyy0_p[i] = trilerp(Pyy0, pts_probes[i], spacing, origin)
        Pyz0_p[i] = trilerp(Pyz0, pts_probes[i], spacing, origin)
        Pzz0_p[i] = trilerp(Pzz0, pts_probes[i], spacing, origin)

        # electron pressure tensor components along the path
        Pxx1_p[i] = trilerp(Pxx1, pts_probes[i], spacing, origin)
        Pxy1_p[i] = trilerp(Pxy1, pts_probes[i], spacing, origin)
        Pxz1_p[i] = trilerp(Pxz1, pts_probes[i], spacing, origin)
        Pyy1_p[i] = trilerp(Pyy1, pts_probes[i], spacing, origin)
        Pyz1_p[i] = trilerp(Pyz1, pts_probes[i], spacing, origin)
        Pzz1_p[i] = trilerp(Pzz1, pts_probes[i], spacing, origin)

    # which probe index (0..n_probes-1) to plot
    plot_idx = max(0, min(n_probes - 1, probe_to_plot - 1))

    # select data for the chosen probe
    Bx_sel   = Bx_p[plot_idx]
    By_sel   = By_p[plot_idx]
    Bz_sel   = Bz_p[plot_idx]
    Bmag_sel = Bmag_p[plot_idx]

    if Ex_p is not None:
        Ex_sel   = Ex_p[plot_idx]
        Ey_sel   = Ey_p[plot_idx]
        Ez_sel   = Ez_p[plot_idx]
        Emag_sel = Emag_p[plot_idx]
    else:
        Ex_sel = Ey_sel = Ez_sel = Emag_sel = None

    vi_x_sel = vi_x_p[plot_idx]
    vi_y_sel = vi_y_p[plot_idx]
    vi_z_sel = vi_z_p[plot_idx]

    ve_x_sel = ve_x_p[plot_idx]
    ve_y_sel = ve_y_p[plot_idx]
    ve_z_sel = ve_z_p[plot_idx]

    vb_x_sel = vb_x_p[plot_idx]
    vb_y_sel = vb_y_p[plot_idx]
    vb_z_sel = vb_z_p[plot_idx]

    rho0_sel = rho0_p[plot_idx]
    rho1_sel = rho1_p[plot_idx]

    # ---- plot canvas 1: B and E (four stacked panels) ----
    if k == 0:
        fig_BE, axes_BE = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

        # 1) B components
        ax = axes_BE[0]
        ax.plot(x_axis, Bx_sel, label=r"$B_x$")
        ax.plot(x_axis, By_sel, label=r"$B_y$")
        ax.plot(x_axis, Bz_sel, label=r"$B_z$")
        ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
        ax.set_ylabel("B comp.")
        ax.legend(frameon=False, ncol=3, loc="upper right")

        # 2) |B|
        ax = axes_BE[1]
        ax.plot(x_axis, Bmag_sel, label=r"$|B|$")
        ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
        ax.set_ylabel(r"$|B|$")
        ax.legend(frameon=False, loc="upper right")

        # 3) E components
        ax = axes_BE[2]
        if Ex_sel is not None:
            ax.plot(x_axis, Ex_sel, label=r"$E_x$")
            ax.plot(x_axis, Ey_sel, label=r"$E_y$")
            ax.plot(x_axis, Ez_sel, label=r"$E_z$")
            ax.legend(frameon=False, ncol=3, loc="upper right")
        else:
            ax.text(0.02, 0.85, "E-field not found", transform=ax.transAxes)
        ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
        ax.set_ylabel("E comp.")

        # 4) |E|
        ax = axes_BE[3]
        if Ex_sel is not None:
            ax.plot(x_axis, Emag_sel, label=r"$|E|$")
            ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
            ax.set_ylabel(r"$|E|$")
            ax.legend(frameon=False, loc="upper right")
        else:
            ax.text(0.02, 0.85, "E-field not found", transform=ax.transAxes)
        ax.set_xlabel(x_label)

        for ax in axes_BE:
            ax.grid(True, linewidth=0.6, alpha=0.6)

        fig_BE.tight_layout()
        fig_BE_path = out_dir_figs / f"probe_flyby_{k:03d}_BE.png"
        fig_BE.savefig(fig_BE_path, dpi=200, bbox_inches="tight")

    # ---- plot canvas 2: velocities and densities (four stacked panels) ----
    if k == 0:
        fig_VRho, axes_VRho = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

        # 1) ion velocity components
        ax = axes_VRho[0]
        ax.plot(x_axis, vi_x_sel, label=r"$v_{i,x}$")
        ax.plot(x_axis, vi_y_sel, label=r"$v_{i,y}$")
        ax.plot(x_axis, vi_z_sel, label=r"$v_{i,z}$")
        ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
        ax.set_ylabel(r"$\mathbf{v}_i$")
        ax.legend(frameon=False, ncol=3, loc="upper right")

        # 2) electron velocity components
        ax = axes_VRho[1]
        ax.plot(x_axis, ve_x_sel, label=r"$v_{e,x}$")
        ax.plot(x_axis, ve_y_sel, label=r"$v_{e,y}$")
        ax.plot(x_axis, ve_z_sel, label=r"$v_{e,z}$")
        ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
        ax.set_ylabel(r"$\mathbf{v}_e$")
        ax.legend(frameon=False, ncol=3, loc="upper right")

        # 3) bulk velocity components
        ax = axes_VRho[2]
        ax.plot(x_axis, vb_x_sel, label=r"$v_{b,x}$")
        ax.plot(x_axis, vb_y_sel, label=r"$v_{b,y}$")
        ax.plot(x_axis, vb_z_sel, label=r"$v_{b,z}$")
        ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
        ax.set_ylabel(r"$\mathbf{v}_{\text{bulk}}$")
        ax.legend(frameon=False, ncol=3, loc="upper right")

        # 4) densities rho_0 and rho_1
        ax = axes_VRho[3]
        ax.plot(x_axis, rho0_sel, label=r"$\rho_0$ (ions)")
        ax.plot(x_axis, rho1_sel, label=r"$\rho_1$ (electrons)")
        ax.set_ylabel(r"$\rho$")
        ax.set_xlabel(x_label)
        ax.legend(frameon=False, ncol=2, loc="upper right")

        for ax in axes_VRho:
            ax.grid(True, linewidth=0.6, alpha=0.6)

        fig_VRho.tight_layout()
        fig_VRho_path = out_dir_figs / f"probe_flyby_{k:03d}_vel_rho.png"
        fig_VRho.savefig(fig_VRho_path, dpi=200, bbox_inches="tight")
        # Optionally: plt.close instead of show
        plt.close(fig_BE)
        plt.close(fig_VRho)

        print(f"Saved B/E figure to {fig_BE_path.resolve()}")
        print(f"Saved velocity/density figure to {fig_VRho_path.resolve()}")

    # ---- save CSV (ready for SINDy, all 4 probes) ----
    csv_path = out_dir_csv / f"probe_flyby_{k:03d}.csv"
    cols = [
        ("s", t),
        ("xc", pts[:,0]), ("yc", pts[:,1]), ("zc", pts[:,2]),
        ("dist", dist),
    ]

    # add data for each probe as separate columns
    for i in range(n_probes):
        suffix = f"_{i+1}"
        cols.extend([
            (f"x{suffix}",   pts_probes[i, :, 0]),
            (f"y{suffix}",   pts_probes[i, :, 1]),
            (f"z{suffix}",   pts_probes[i, :, 2]),
            (f"rho0{suffix}", rho0_p[i]),
            (f"rho1{suffix}", rho1_p[i]),
            (f"Bx{suffix}",   Bx_p[i]),
            (f"By{suffix}",   By_p[i]),
            (f"Bz{suffix}",   Bz_p[i]),
            (f"Bmag{suffix}", Bmag_p[i]),
        ])
        # ion pressure tensor components
        cols.extend([
            (f"Pxx0{suffix}", Pxx0_p[i]),
            (f"Pxy0{suffix}", Pxy0_p[i]),
            (f"Pxz0{suffix}", Pxz0_p[i]),
            (f"Pyy0{suffix}", Pyy0_p[i]),
            (f"Pyz0{suffix}", Pyz0_p[i]),
            (f"Pzz0{suffix}", Pzz0_p[i]),
        ])
        # electron pressure tensor components
        cols.extend([
            (f"Pxx1{suffix}", Pxx1_p[i]),
            (f"Pxy1{suffix}", Pxy1_p[i]),
            (f"Pxz1{suffix}", Pxz1_p[i]),
            (f"Pyy1{suffix}", Pyy1_p[i]),
            (f"Pyz1{suffix}", Pyz1_p[i]),
            (f"Pzz1{suffix}", Pzz1_p[i]),
        ])
        if Ex_p is not None:
            cols.extend([
                (f"Ex{suffix}",   Ex_p[i]),
                (f"Ey{suffix}",   Ey_p[i]),
                (f"Ez{suffix}",   Ez_p[i]),
                (f"Emag{suffix}", Emag_p[i]),
            ])
        # ion velocity
        cols.extend([
            (f"vi_x{suffix}", vi_x_p[i]),
            (f"vi_y{suffix}", vi_y_p[i]),
            (f"vi_z{suffix}", vi_z_p[i]),
        ])
        # electron velocity
        cols.extend([
            (f"ve_x{suffix}", ve_x_p[i]),
            (f"ve_y{suffix}", ve_y_p[i]),
            (f"ve_z{suffix}", ve_z_p[i]),
        ])
        # bulk velocity
        cols.extend([
            (f"vb_x{suffix}", vb_x_p[i]),
            (f"vb_y{suffix}", vb_y_p[i]),
            (f"vb_z{suffix}", vb_z_p[i]),
        ])

    header = ",".join([name for name, _ in cols])
    data = np.column_stack([arr for _, arr in cols])
