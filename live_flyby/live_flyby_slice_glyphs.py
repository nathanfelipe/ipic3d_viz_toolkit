import os
import h5py
import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt


# =========================
# Inputs (from your context)
# =========================
path = "../../DoubleGEM-Fields_300000.h5" # "../T3D-Fields_005500.h5"

scalar_name = "rho_1" # rho_1 qom = 1 for ions and rho_0 qom = -64 for electrons (Double check ecsim convention)
bx_name, by_name, bz_name = "Bx", "By", "Bz"

# geometry from SimulationData.txt (as in your scripts)
DX, DY, DZ = 0.125, 0.125, 0.125  # 0.125
X0, Y0, Z0 = 0.0, 0.0, 0.0

# Strides (same spirit as isosurface_v3.py)
s_stride = 1
v_stride = 2

# Probe path (edit to taste)
p0 = np.array([24.0,  36.0,  2.0]) # y = 36 for crossing x point
p1 = np.array([1.0, 36.0, 2.0])
n_samples = 2001
stop_at_domain_exit = True  # truncate path when probe leaves the data domain (prevents boundary-clamped fake measurements)
print_domain_info = True  # print valid coordinate bounds and probe endpoints when outside-domain errors occur

# Video + animation controls
out_dir = "frames"
clear_existing_frames = True  # delete existing frame_*.png in out_dir before rendering
fps = 30
frame_stride = 10     # render every Nth sample as one frame
max_frames = None     # set an int to limit (e.g., 300)

# Probe visuals
probe_radius = 0.25
# NOTE: B is loaded on a coarser grid (v_stride). If probe_radius is smaller than the B-grid spacing,
# a spherical mask would include ~1 voxel and mean-subtraction would zero it out, producing a flat spectrum.
# We therefore use an effective radius for the B-mask that is at least 2 B-grid spacings.
probe_radius_B_mask_factor = 2.0
trail_tube_radius = 0.05
use_tube_trail = True  # set False if performance is bad

# Slice plane config
# We will build a plane that CONTAINS the probe direction (tangent) and passes through the probe.
# The plane is defined by its normal, which we compute per-frame from the probe tangent.
# `plane_up_ref` selects the second in-plane direction (think "vertical").
plane_up_ref = np.array([1.0, 0.0, 0.0])
plane_roll_deg = 0.0  # rotate plane around the probe direction (degrees). Adjust if you want.

# If you want a fixed plane instead, set this True and use the fixed values below.
use_fixed_plane = False
# fixed_plane_normal = (0, 0, 1)
# fixed_plane_origin = (6.0, 8.0, 10.0)

# Camera config: if True, keep the camera looking face-on to the slice plane
follow_plane_camera = True
camera_distance = 100.0  # in simulation units; increase if the view feels too zoomed

# Glyph density on the slice (bigger => fewer arrows, faster)
show_glyphs = False
show_B_contours = False  # set True to draw |B| contour lines on the plane
n_B_contours = 15         # number of contour levels when show_B_contours=True

# Sampling controls for glyph placement
# Using random sampling avoids points being selected mostly along an edge (which happens with simple stride indexing)
glyph_max_points = 600   # safety cap
rng_seed = 0              # deterministic sampling; change for a different pattern

# Glyph visual scaling

glyph_scale = 0.6

# Species metadata (edit as needed for your run)
SPECIES_META = {
    "0": {"qom": -64, "population": "background"},
    "1": {"qom": 1,   "population": "background"},
    "2": {"qom": -64, "population": "drifting"},
    "3": {"qom": 1,   "population": "drifting"},
}


# =========================
# Energy spectra (final single plot)
# =========================
compute_energy_spectra = True

# We compute isotropic power spectra (|FFT|^2 binned by |k|) in a moving local volume around the probe,
# masked by the same `probe_radius` sphere that defines the synthetic probe volume.
#
# 1) Kinetic energy density spectrum: 0.5 * rho * |V|^2
# 2) Magnetic energy density spectrum: 0.5 * |B|^2
#
# Notes:
# - These are *field* energy-density spectra (not particle distribution spectra).
# - Constants like mu0 are omitted (can be added later as a scale factor).
# - FFT needs a regular cube; we extract a cube of size `spectrum_box_n`^3 around the probe,
#   apply the spherical mask, and compute |FFT|^2 binned by |k|.

spectrum_box_n = 128            # cube edge length (grid points) for FFT (zero-padded if near boundary)
use_spectrum_frame_ids = True  # True: use `frame_ids` (sparser, faster). False: use every sample point.
average_spectrum_over_path = False  # True: average spectra across selected probe positions

plot_kinetic_energy_spectrum = True
plot_magnetic_energy_spectrum = True

kinetic_spectrum_out_png  = "kinetic_energy_spectrum.png"
magnetic_spectrum_out_png = "magnetic_energy_spectrum.png"


# =========================
# MMS-style energy distribution option (moment-based proxy, needs to improve with particle data)
# =========================
# This produces a plot in the same *format* as MMS energy spectra: dN/dE vs E/(m c^2).
# IMPORTANT: This is NOT a true particle distribution unless you have particle data.
# Here we build a *proxy* by treating each grid cell inside the probe sphere as contributing
# weight ~ N (number density) at energy E = 0.5 m |v|^2, with v = J/rho.
compute_mms_style_energy_spectrum = False
mms_energy_out_png = "mms_style_energy_spectrum.png"

# Speed of light in code units (often c=1 in normalized PIC units). Used for x-axis E/(m c^2).
c_light = 1.0

# Log-range and binning for x = E/(m c^2)
# MMS slide spans ~1e-8..1e-4; adjust as needed.
mms_xmin = 1e-6
mms_xmax = 1e-4
mms_nbins = 60

# If available, try to load diagonal pressure components and overlay a Maxwellian fit.
# Fit is approximate and assumes isotropy: P = (Pxx+Pyy+Pzz)/3 and T ~ P/n (kB=1).
overlay_maxwellian_fit = False


# =========================
# Helpers (adapted from your scripts)
# =========================
def open_var(f, name):
    obj = f[f"Step#0/Block/{name}"]
    if isinstance(obj, h5py.Group):
        return obj["0"] if "0" in obj else next(v for v in obj.values() if isinstance(v, h5py.Dataset))
    return obj

def trilerp(arr, xyz, spacing, origin):
    DX, DY, DZ = spacing
    X0, Y0, Z0 = origin
    nx, ny, nz = arr.shape # NOTE: pv.ImageData dims are (nx, ny, nz) | if broken, try nz, ny, nx

    x = (xyz[..., 0] - X0) / DX
    y = (xyz[..., 1] - Y0) / DY
    z = (xyz[..., 2] - Z0) / DZ

    x0 = np.floor(x).astype(int); x1 = x0 + 1
    y0 = np.floor(y).astype(int); y1 = y0 + 1
    z0 = np.floor(z).astype(int); z1 = z0 + 1

    x0 = np.clip(x0, 0, nx-1); x1 = np.clip(x1, 0, nx-1)
    y0 = np.clip(y0, 0, ny-1); y1 = np.clip(y1, 0, ny-1)
    z0 = np.clip(z0, 0, nz-1); z1 = np.clip(z1, 0, nz-1)

    fx = np.clip(x - x0, 0, 1)
    fy = np.clip(y - y0, 0, 1)
    fz = np.clip(z - z0, 0, 1)

    c000 = arr[z0, y0, x0]; c100 = arr[z0, y0, x1]
    c010 = arr[z0, y1, x0]; c110 = arr[z0, y1, x1]
    c001 = arr[z1, y0, x0]; c101 = arr[z1, y0, x1]
    c011 = arr[z1, y1, x0]; c111 = arr[z1, y1, x1]

    c00 = c000*(1-fx) + c100*fx; c01 = c001*(1-fx) + c101*fx
    c10 = c010*(1-fx) + c110*fx; c11 = c011*(1-fx) + c111*fx
    c0  = c00*(1-fy) + c10*fy;   c1  = c01*(1-fy) + c11*fy
    return c0*(1-fz) + c1*fz

def mpl_fig_to_rgb(fig):
    # Renders matplotlib figure into an RGB uint8 array (HxWx3)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    # Agg gives RGBA
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)  # shape (h, w, 4)
    rgba = buf.reshape(h, w, 4)
    rgb = rgba[:, :, :3].copy()
    return rgb

def hstack_images(left_rgb, right_rgb):
    # Pad to same height
    h = max(left_rgb.shape[0], right_rgb.shape[0])
    def pad_to(img, H):
        if img.shape[0] == H:
            return img
        pad = H - img.shape[0]
        top = pad // 2
        bot = pad - top
        return np.pad(img, ((top, bot), (0, 0), (0, 0)), mode="constant")
    L = pad_to(left_rgb, h)
    R = pad_to(right_rgb, h)
    return np.concatenate([L, R], axis=1)

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def _rodrigues_rotate(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    # Rotate vector v about unit axis by angle_rad
    k = _unit(axis)
    v = np.asarray(v, dtype=float)
    return (v * np.cos(angle_rad)
            + np.cross(k, v) * np.sin(angle_rad)
            + k * np.dot(k, v) * (1.0 - np.cos(angle_rad)))

def plane_from_probe(pts: np.ndarray, i: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (origin, normal) for a plane that contains the probe tangent and plane_up_ref."""
    origin = pts[i]

    if use_fixed_plane:
        return np.asarray(fixed_plane_origin, dtype=float), _unit(np.asarray(fixed_plane_normal, dtype=float))

    # Tangent (probe direction)
    if i < len(pts) - 1:
        t = pts[i + 1] - pts[i]
    else:
        t = pts[i] - pts[i - 1]
    t = _unit(t)

    # Choose an up reference that is not parallel to tangent
    up = np.asarray(plane_up_ref, dtype=float)
    if abs(float(np.dot(t, _unit(up)))) > 0.95:
        up = np.array([0.0, 1.0, 0.0])

    # Plane normal so that plane contains both t and up: n = t x up
    n = np.cross(t, up)
    n = _unit(n)

    # Optional roll around the tangent direction
    if plane_roll_deg != 0.0:
        n = _rodrigues_rotate(n, t, np.deg2rad(plane_roll_deg))
        n = _unit(n)

    return origin, n

# Camera face-on helper
def camera_pose_for_plane(origin: np.ndarray, normal: np.ndarray, tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (position, focal_point, view_up) to view the plane face-on."""
    n = _unit(normal)
    t = _unit(tangent)

    # Prefer a view-up that lies in the plane and is as close as possible to the global up reference.
    up = np.asarray(plane_up_ref, dtype=float)
    up_in_plane = up - np.dot(up, n) * n
    if np.linalg.norm(up_in_plane) < 1e-8:
        # Fallback: use a vector perpendicular to both n and t (also lies in plane)
        up_in_plane = np.cross(n, t)
    up_in_plane = _unit(up_in_plane)

    pos = origin + camera_distance * n
    focal = origin
    return pos, focal, up_in_plane


# =========================
# Energy spectrum helpers
# =========================

def _index_center_from_xyz(xyz: np.ndarray, spacing, origin) -> tuple[int, int, int]:
    """Return (iz, iy, ix) integer index of the nearest grid point to xyz."""
    dx, dy, dz = spacing
    x0, y0, z0 = origin
    ix = int(np.rint((float(xyz[0]) - x0) / dx))
    iy = int(np.rint((float(xyz[1]) - y0) / dy))
    iz = int(np.rint((float(xyz[2]) - z0) / dz))
    return iz, iy, ix


def extract_cube_around_point(arr: np.ndarray, center_xyz: np.ndarray, spacing, origin, n: int) -> np.ndarray:
    """Extract an n^3 cube from arr (z,y,x) around center_xyz. Zero-pad if outside bounds."""
    n = int(n)
    if n < 2:
        raise ValueError("spectrum_box_n must be >= 2")

    nz, ny, nx = arr.shape
    izc, iyc, ixc = _index_center_from_xyz(center_xyz, spacing, origin)

    # Use a symmetric window around the center index
    half = n // 2
    z0, z1 = izc - half, izc - half + n
    y0, y1 = iyc - half, iyc - half + n
    x0, x1 = ixc - half, ixc - half + n

    # Allocate output cube (zero-padded)
    cube = np.zeros((n, n, n), dtype=np.float32)

    # Compute valid source ranges
    sz0, sz1 = max(z0, 0), min(z1, nz)
    sy0, sy1 = max(y0, 0), min(y1, ny)
    sx0, sx1 = max(x0, 0), min(x1, nx)

    # Compute destination ranges
    dz0, dz1 = sz0 - z0, (sz0 - z0) + (sz1 - sz0)
    dy0, dy1 = sy0 - y0, (sy0 - y0) + (sy1 - sy0)
    dx0, dx1 = sx0 - x0, (sx0 - x0) + (sx1 - sx0)

    cube[dz0:dz1, dy0:dy1, dx0:dx1] = arr[sz0:sz1, sy0:sy1, sx0:sx1].astype(np.float32, copy=False)
    return cube


def spherical_mask(n: int, spacing, radius: float) -> np.ndarray:
    """Return boolean mask for points within a sphere of given radius centered in the cube."""
    dx, dy, dz = spacing
    # Center index in continuous coordinates
    c = (n - 1) / 2.0
    z, y, x = np.ogrid[:n, :n, :n]
    rx = (x - c) * dx
    ry = (y - c) * dy
    rz = (z - c) * dz
    r2 = rx * rx + ry * ry + rz * rz
    return r2 <= float(radius) ** 2


def isotropic_spectrum_3d(field_cube: np.ndarray, spacing) -> tuple[np.ndarray, np.ndarray]:
    """Compute isotropic |FFT|^2 spectrum binned by |k| for a 3D cube (z,y,x)."""
    n = field_cube.shape[0]
    if field_cube.shape != (n, n, n):
        raise ValueError("field_cube must be cubic")

    dx, dy, dz = spacing

    # FFT and power
    fk = np.fft.fftn(field_cube)
    pk = np.abs(fk) ** 2

    # k grids (radians per unit length)
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(n, d=dy)
    kz = 2.0 * np.pi * np.fft.fftfreq(n, d=dz)
    kz3, ky3, kx3 = np.meshgrid(kz, ky, kx, indexing="ij")
    kmag = np.sqrt(kx3 * kx3 + ky3 * ky3 + kz3 * kz3)

    # Bin by |k|
    km = kmag.ravel()
    pp = pk.ravel()

    # Use linear bins up to k_max; plot on log scale later.
    kmax = float(km.max())
    nbins = max(32, n // 2)
    edges = np.linspace(0.0, kmax, nbins + 1)
    which = np.digitize(km, edges) - 1

    spec = np.zeros(nbins, dtype=np.float64)
    counts = np.zeros(nbins, dtype=np.int64)

    for bi in range(nbins):
        m = which == bi
        if np.any(m):
            spec[bi] = pp[m].sum(dtype=np.float64)
            counts[bi] = int(m.sum())

    # Avoid divide-by-zero; compute average power per shell
    good = counts > 0
    spec[good] /= counts[good]

    # Bin centers
    kcent = 0.5 * (edges[:-1] + edges[1:])

    # Drop the k=0 bin (mean/DC)
    if kcent.size > 1:
        return kcent[1:], spec[1:]
    return kcent, spec


# =========================
# MMS-style energy distribution helpers (moment-based proxy)
# =========================

def try_load_pressure_diagonal(f: h5py.File, species_idx: str, stride: int):
    """Best-effort load of diagonal pressure tensor components for a species.

    Returns (Pxx, Pyy, Pzz) as float32 arrays (z,y,x) with the requested stride,
    or (None, None, None) if not available.
    """
    names = (f"Pxx_{species_idx}", f"Pyy_{species_idx}", f"Pzz_{species_idx}")
    try:
        Pxx = np.asarray(open_var(f, names[0]), dtype=np.float32)[::stride, ::stride, ::stride]
        Pyy = np.asarray(open_var(f, names[1]), dtype=np.float32)[::stride, ::stride, ::stride]
        Pzz = np.asarray(open_var(f, names[2]), dtype=np.float32)[::stride, ::stride, ::stride]
        return Pxx, Pyy, Pzz
    except Exception:
        return None, None, None


def weighted_dNdE_from_cube(vmag: np.ndarray, n_cube: np.ndarray, m: float, c: float,
                           x_edges: np.ndarray, mask=None):
    """Build a weighted proxy dN/dE from local bulk speeds.

    - vmag: speed magnitude per cell (same shape as n_cube)
    - n_cube: number density per cell (weights)
    - m: species mass in code units
    - c: speed of light in code units
    - x_edges: bin edges for x = E/(m c^2)
    - mask: optional boolean mask selecting cells (e.g., spherical probe)

    Returns (E_centers, dNdE) where E_centers are bin centers in energy units.
    """
    vmag = np.asarray(vmag, dtype=np.float64)
    n_cube = np.asarray(n_cube, dtype=np.float64)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        vmag = vmag[mask]
        n_cube = n_cube[mask]

    good = np.isfinite(vmag) & np.isfinite(n_cube) & (n_cube > 0)
    vmag = vmag[good]
    w = n_cube[good]

    x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
    denom = float(m) * (float(c) ** 2)
    E_cent = x_cent * denom

    if vmag.size == 0:
        return E_cent, np.zeros_like(E_cent)

    # Non-relativistic kinetic energy per cell
    E = 0.5 * float(m) * (vmag ** 2)
    if denom <= 0:
        raise ValueError("Invalid m or c for E/(m c^2)")
    x = E / denom

    # print("x stats:", np.nanmin(x), np.nanpercentile(x, [1, 50, 99]), np.nanmax(x)) # test for mms bin statistics

    counts, _ = np.histogram(x, bins=x_edges, weights=w)

    # dN/dE = counts / dE, where dE = (m c^2) dx
    dx = np.diff(x_edges)
    dE = denom * dx
    dNdE = counts / np.where(dE > 0, dE, np.nan)
    dNdE = np.where(np.isfinite(dNdE), dNdE, 0.0)

    return E_cent, dNdE


def maxwellian_dNdE(E: np.ndarray, T: float) -> np.ndarray:
    """Shape-only Maxwellian energy distribution: dN/dE ∝ sqrt(E) exp(-E/T)."""
    E = np.asarray(E, dtype=np.float64)
    T = float(T)
    if T <= 0:
        return np.zeros_like(E)
    return np.sqrt(np.maximum(E, 0.0)) * np.exp(-np.maximum(E, 0.0) / T)


# =========================
# Load fields once
# =========================
with h5py.File(path, "r") as f:
    rho = np.asarray(open_var(f, scalar_name), dtype=np.float32)[::s_stride, ::s_stride, ::s_stride]
    Bx  = np.asarray(open_var(f, bx_name), dtype=np.float32)[::v_stride, ::v_stride, ::v_stride]
    By  = np.asarray(open_var(f, by_name), dtype=np.float32)[::v_stride, ::v_stride, ::v_stride]
    Bz  = np.asarray(open_var(f, bz_name), dtype=np.float32)[::v_stride, ::v_stride, ::v_stride]

    # --- Species moments for kinetic energy density ---
    # In this ECSIM output, bulk velocity components (Vx,Vy,Vz) are not stored directly.
    # Instead we have species current density J and (charge) density rho, plus number density N.
    # Using v = J / rho (valid if rho = q n and J = q n v), we build a kinetic-energy-density proxy:
    #   KE_s = 0.5 * N_s * |J_s / rho_s|^2   (mass factor omitted / assumed 1)
    species_idx = None
    _sn = str(scalar_name)
    if _sn.startswith("rho_") and _sn.split("_")[-1].isdigit():
        species_idx = _sn.split("_")[-1]

    N = Jx = Jy = Jz = None
    if species_idx is not None:
        N  = np.asarray(open_var(f, f"N_{species_idx}"),  dtype=np.float32)[::s_stride, ::s_stride, ::s_stride]
        Jx = np.asarray(open_var(f, f"Jx_{species_idx}"), dtype=np.float32)[::s_stride, ::s_stride, ::s_stride]
        Jy = np.asarray(open_var(f, f"Jy_{species_idx}"), dtype=np.float32)[::s_stride, ::s_stride, ::s_stride]
        Jz = np.asarray(open_var(f, f"Jz_{species_idx}"), dtype=np.float32)[::s_stride, ::s_stride, ::s_stride]
    else:
        print(f"[warn] scalar_name={scalar_name!r} does not look like 'rho_<i>'. Kinetic spectrum will be skipped.")

    # Optional: diagonal pressure components for an approximate Maxwellian overlay
    Pxx = Pyy = Pzz = None
    if (species_idx is not None) and overlay_maxwellian_fit:
        Pxx, Pyy, Pzz = try_load_pressure_diagonal(f, species_idx, s_stride)
        if Pxx is None:
            print(f"[info] Pressure diagonal (Pxx/Pyy/Pzz) not found for species {species_idx}; Maxwellian fit will be skipped.")

# Build PyVista grids
nz, ny, nx = rho.shape
grid = pv.ImageData()
grid.dimensions = (nx, ny, nz)
grid.spacing = (DX*s_stride, DY*s_stride, DZ*s_stride)
grid.origin = (X0, Y0, Z0)
grid.point_data[scalar_name] = rho.ravel(order="C")

nzv, nyv, nxv = Bx.shape
grid_v = pv.ImageData()
grid_v.dimensions = (nxv, nyv, nzv)
grid_v.spacing = (DX*v_stride, DY*v_stride, DZ*v_stride)
grid_v.origin = (X0, Y0, Z0)
B = np.stack([Bx, By, Bz], axis=-1)
grid_v.point_data["B"] = B.reshape(-1, 3, order="C")
grid_v.point_data["Bmag"] = np.linalg.norm(B, axis=-1).ravel(order="C")

# Compute physical bounds for a grid (inclusive min, inclusive max at last index)
def grid_bounds(origin_xyz, spacing_xyz, dims_xyz):
    ox, oy, oz = origin_xyz
    sx, sy, sz = spacing_xyz
    nx, ny, nz = dims_xyz  # NOTE: pv.ImageData dims are (nx, ny, nz)
    xmin, xmax = ox, ox + sx * (nx - 1)
    ymin, ymax = oy, oy + sy * (ny - 1)
    zmin, zmax = oz, oz + sz * (nz - 1)
    return (xmin, xmax), (ymin, ymax), (zmin, zmax)

# Intersection bounds where BOTH rho-grid and B-grid are valid
(bounds_x_rho, bounds_y_rho, bounds_z_rho) = grid_bounds((X0, Y0, Z0), (DX*s_stride, DY*s_stride, DZ*s_stride), (nx, ny, nz))
(bounds_x_B,   bounds_y_B,   bounds_z_B)   = grid_bounds((X0, Y0, Z0), (DX*v_stride, DY*v_stride, DZ*v_stride), (nxv, nyv, nzv))

xmin = max(bounds_x_rho[0], bounds_x_B[0]); xmax = min(bounds_x_rho[1], bounds_x_B[1])
ymin = max(bounds_y_rho[0], bounds_y_B[0]); ymax = min(bounds_y_rho[1], bounds_y_B[1])
zmin = max(bounds_z_rho[0], bounds_z_B[0]); zmax = min(bounds_z_rho[1], bounds_z_B[1])

# Small epsilon so we don't sample exactly on the last index boundary
_eps = 1e-6

xmax -= _eps; ymax -= _eps; zmax -= _eps

# --- Helper: print domain info if requested ---
def _fmt_bounds(b):
    return f"[{b[0]:.6g}, {b[1]:.6g}]"

if print_domain_info:
    print("\n=== Data domain info ===")
    print(f"rho grid dims (nx,ny,nz)=({nx},{ny},{nz}), spacing=({DX*s_stride},{DY*s_stride},{DZ*s_stride}), origin=({X0},{Y0},{Z0})")
    print(f"B   grid dims (nx,ny,nz)=({nxv},{nyv},{nzv}), spacing=({DX*v_stride},{DY*v_stride},{DZ*v_stride}), origin=({X0},{Y0},{Z0})")
    print(f"rho bounds: x={_fmt_bounds(bounds_x_rho)} y={_fmt_bounds(bounds_y_rho)} z={_fmt_bounds(bounds_z_rho)}")
    print(f"B   bounds: x={_fmt_bounds(bounds_x_B)} y={_fmt_bounds(bounds_y_B)} z={_fmt_bounds(bounds_z_B)}")
    print(f"INTERSECTION (valid for both): x={_fmt_bounds((xmin,xmax))} y={_fmt_bounds((ymin,ymax))} z={_fmt_bounds((zmin,zmax))}")
    print("========================\n")

# --- Helper: detect if the probe path is (nearly) in a constant-coordinate plane ---
def detect_out_of_plane_constant(pts: np.ndarray, tol: float = 1e-6):
    """Return (axis, value) if one coordinate is ~constant along the path, else (None, None)."""
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return None, None

    ranges = np.ptp(pts, axis=0)  # peak-to-peak per axis
    axis_idx = int(np.argmin(ranges))
    if float(ranges[axis_idx]) <= float(tol):
        axis = "xyz"[axis_idx]
        val = float(np.mean(pts[:, axis_idx]))
        return axis, val
    return None, None


def make_plane_outline(axis: str, val: float,
                       xb: tuple[float, float], yb: tuple[float, float], zb: tuple[float, float]) -> pv.PolyData:
    """Create a rectangular outline polyline on a constant-x/y/z plane using the provided bounds."""
    xmin, xmax = xb
    ymin, ymax = yb
    zmin, zmax = zb

    axis = (axis or "").lower()
    if axis == "z":
        corners = np.array([
            [xmin, ymin, val],
            [xmax, ymin, val],
            [xmax, ymax, val],
            [xmin, ymax, val],
        ], dtype=float)
    elif axis == "y":
        corners = np.array([
            [xmin, val, zmin],
            [xmax, val, zmin],
            [xmax, val, zmax],
            [xmin, val, zmax],
        ], dtype=float)
    elif axis == "x":
        corners = np.array([
            [val, ymin, zmin],
            [val, ymax, zmin],
            [val, ymax, zmax],
            [val, ymin, zmax],
        ], dtype=float)
    else:
        # Fallback: draw an outline on the mid-z plane of the intersection bounds
        zc = 0.5 * (zmin + zmax)
        corners = np.array([
            [xmin, ymin, zc],
            [xmax, ymin, zc],
            [xmax, ymax, zc],
            [xmin, ymax, zc],
        ], dtype=float)

    outline = pv.PolyData(corners)
    # One closed polyline cell: 0-1-2-3-0
    outline.lines = np.array([5, 0, 1, 2, 3, 0], dtype=np.int64)
    return outline


def plane_outline_labels(axis: str, val: float,
                         xb: tuple[float, float], yb: tuple[float, float], zb: tuple[float, float]) -> tuple[np.ndarray, list[str]]:
    """Return (points, labels) for the outline corners (used by add_point_labels)."""
    xmin, xmax = xb
    ymin, ymax = yb
    zmin, zmax = zb

    axis = (axis or "").lower()
    if axis == "z":
        pts_lab = np.array([
            [xmin, ymin, val],
            [xmax, ymin, val],
            [xmax, ymax, val],
            [xmin, ymax, val],
        ], dtype=float)
        labels = [
            f"x={xmin:.3g}, y={ymin:.3g}",
            f"x={xmax:.3g}, y={ymin:.3g}",
            f"x={xmax:.3g}, y={ymax:.3g}",
            f"x={xmin:.3g}, y={ymax:.3g}",
        ]
    elif axis == "y":
        pts_lab = np.array([
            [xmin, val, zmin],
            [xmax, val, zmin],
            [xmax, val, zmax],
            [xmin, val, zmax],
        ], dtype=float)
        labels = [
            f"x={xmin:.3g}, z={zmin:.3g}",
            f"x={xmax:.3g}, z={zmin:.3g}",
            f"x={xmax:.3g}, z={zmax:.3g}",
            f"x={xmin:.3g}, z={zmax:.3g}",
        ]
    elif axis == "x":
        pts_lab = np.array([
            [val, ymin, zmin],
            [val, ymax, zmin],
            [val, ymax, zmax],
            [val, ymin, zmax],
        ], dtype=float)
        labels = [
            f"y={ymin:.3g}, z={zmin:.3g}",
            f"y={ymax:.3g}, z={zmin:.3g}",
            f"y={ymax:.3g}, z={zmax:.3g}",
            f"y={ymin:.3g}, z={zmax:.3g}",
        ]
    else:
        zc = 0.5 * (zmin + zmax)
        pts_lab = np.array([
            [xmin, ymin, zc],
            [xmax, ymin, zc],
            [xmax, ymax, zc],
            [xmin, ymax, zc],
        ], dtype=float)
        labels = [
            f"x={xmin:.3g}, y={ymin:.3g}",
            f"x={xmax:.3g}, y={ymin:.3g}",
            f"x={xmax:.3g}, y={ymax:.3g}",
            f"x={xmin:.3g}, y={ymax:.3g}",
        ]

    return pts_lab, labels


# =========================
# Probe path and time series (sample once)
# =========================
t = np.linspace(0.0, 1.0, n_samples)
pts = (1.0 - t)[:, None] * p0[None, :] + t[:, None] * p1[None, :]

dist = np.linalg.norm(pts - pts[0], axis=1)
x_axis = dist

# Truncate the path once it leaves the valid domain (prevents trilerp() clamping from producing fake values)
if stop_at_domain_exit:
    inside = (
        (pts[:, 0] >= xmin + 1e-6) & (pts[:, 0] <= xmax + 1e-6) &
        (pts[:, 1] >= ymin + 1e-6) & (pts[:, 1] <= ymax + 1e-6) &
        (pts[:, 2] >= zmin + 1e-6) & (pts[:, 2] <= zmax + 1e-6)
    )
    if inside.any():
        first_oob = np.where(~inside)[0]
        if first_oob.size > 0:
            cut = int(first_oob[0])
            if cut > 2:  # keep at least a couple of points
                pts = pts[:cut]
                t = t[:cut]
                x_axis = x_axis[:cut]
                n_samples = int(cut)
            else:
                # Exits almost immediately; provide helpful diagnostics
                msg_lines = [
                    "Probe path leaves the data domain almost immediately (within first 2 samples).",
                    f"Current p0={p0.tolist()} p1={p1.tolist()}",
                    "Valid coordinate ranges (intersection of rho and B grids):",
                    f"  x in [{xmin:.6g}, {xmax:.6g}]",
                    f"  y in [{ymin:.6g}, {ymax:.6g}]",
                    f"  z in [{zmin:.6g}, {zmax:.6g}]",
                ]
                raise RuntimeError("\n".join(msg_lines))
    else:
        # Provide a detailed error with valid bounds and current endpoints
        msg_lines = [
            "Probe path starts outside the data domain.",
            f"Current p0={p0.tolist()} p1={p1.tolist()}",
            "Valid coordinate ranges (intersection of rho and B grids):",
            f"  x in [{xmin:.6g}, {xmax:.6g}]",
            f"  y in [{ymin:.6g}, {ymax:.6g}]",
            f"  z in [{zmin:.6g}, {zmax:.6g}]",
            "Tip: choose p0/p1 inside these ranges (or set stop_at_domain_exit=False for debugging).",
        ]
        raise RuntimeError("\n".join(msg_lines))

# --- Report out-of-plane constant axis/value (useful when the flyby lies in a plane) ---
fixed_axis, fixed_val = detect_out_of_plane_constant(pts, tol=1e-6)
if fixed_axis is not None:
    print(f"Out-of-plane constant: {fixed_axis.upper()} = {fixed_val:.6g}")
else:
    print("Out-of-plane constant: none (probe is not confined to a single constant-coordinate plane)")

spacing_rho = (DX*s_stride, DY*s_stride, DZ*s_stride)
spacing_B   = (DX*v_stride, DY*v_stride, DZ*v_stride)
origin = (X0, Y0, Z0)

# Spacing for the rho cube extraction (matches rho array)
spacing_rho_cube = spacing_rho

# Precompute spherical masks for the FFT cubes (used in spectra)
_spectrum_mask_rho = None
_spectrum_mask_B = None
if compute_energy_spectra:
    _spectrum_mask_rho = spherical_mask(int(spectrum_box_n), spacing_rho_cube, probe_radius)

    # B is on a coarser grid; ensure the mask contains enough voxels.
    dxB, dyB, dzB = spacing_B
    min_B_radius = probe_radius_B_mask_factor * max(dxB, dyB, dzB)
    probe_radius_B = max(float(probe_radius), float(min_B_radius))
    _spectrum_mask_B = spherical_mask(int(spectrum_box_n), spacing_B, probe_radius_B)

    # If still too few points (e.g., extremely small cube or unusual spacing), fall back to unmasked.
    if int(_spectrum_mask_B.sum()) < 10:
        _spectrum_mask_B = None
        print(f"[warn] B-mask had <10 voxels (probe_radius={probe_radius}, spacing_B={spacing_B}); using unmasked B spectrum.")


rho_ts = trilerp(rho, pts, spacing_rho, origin)
Bx_ts  = trilerp(Bx,  pts, spacing_B, origin)
By_ts  = trilerp(By,  pts, spacing_B, origin)
Bz_ts  = trilerp(Bz,  pts, spacing_B, origin)
Bmag_ts = np.sqrt(Bx_ts**2 + By_ts**2 + Bz_ts**2)

frame_ids = np.arange(0, len(pts), frame_stride, dtype=int)
if max_frames is not None:
    frame_ids = frame_ids[:max_frames]

# =========================
# Final: MMS-style energy distribution (moment-based proxy)
# =========================
if compute_mms_style_energy_spectrum:
    if species_idx is None:
        print(f"[mms-style] scalar_name={scalar_name!r} does not look like 'rho_<i>'; cannot infer species index. Skipping.")
    elif (N is None) or (Jx is None) or (Jy is None) or (Jz is None):
        print(f"[mms-style] Missing N/J moments for species {species_idx}; cannot build proxy dN/dE. Skipping.")
    else:
        # Infer mass from QOM assuming |q|=1: q/m = QOM => m = 1/|QOM|
        sp = SPECIES_META.get(str(species_idx), None)
        if sp is None:
            print(f"[mms-style] No SPECIES_META for species {species_idx}; assuming m=1.")
            m_s = 1.0
        else:
            qom = float(sp.get("qom", 1.0))
            m_s = 1.0 / max(abs(qom), 1e-12)

        # x bins for x = E/(m c^2)
        x_edges = np.logspace(np.log10(float(mms_xmin)), np.log10(float(mms_xmax)), int(mms_nbins) + 1)

        # Choose which probe centers to accumulate over
        if use_spectrum_frame_ids:
            mms_ids = frame_ids
        else:
            mms_ids = np.arange(len(pts), dtype=int)

        E_cent = None
        dNdE_accum = None
        n_used = 0

        for idx in mms_ids:
            center = pts[idx]

            # Extract cubes on rho/N/J grid
            rho_cube = extract_cube_around_point(rho, center, spacing_rho_cube, origin, int(spectrum_box_n))
            n_cube   = extract_cube_around_point(N,   center, spacing_rho_cube, origin, int(spectrum_box_n))
            jx_cube  = extract_cube_around_point(Jx,  center, spacing_rho_cube, origin, int(spectrum_box_n))
            jy_cube  = extract_cube_around_point(Jy,  center, spacing_rho_cube, origin, int(spectrum_box_n))
            jz_cube  = extract_cube_around_point(Jz,  center, spacing_rho_cube, origin, int(spectrum_box_n))

            # Bulk velocity proxy v = J/rho
            eps = 1e-12
            den = np.where(np.abs(rho_cube) > eps, rho_cube, np.sign(rho_cube) * eps + (rho_cube == 0) * eps)
            vx = jx_cube / den
            vy = jy_cube / den
            vz = jz_cube / den
            vmag = np.sqrt(vx * vx + vy * vy + vz * vz)

            Ei, dNi = weighted_dNdE_from_cube(
                vmag,
                n_cube,
                m=m_s,
                c=float(c_light),
                x_edges=x_edges,
                mask=_spectrum_mask_rho,
            )

            if E_cent is None:
                E_cent = Ei
                dNdE_accum = np.zeros_like(dNi, dtype=np.float64)

            if Ei.shape == E_cent.shape:
                dNdE_accum += dNi.astype(np.float64, copy=False)
                n_used += 1

            if not average_spectrum_over_path:
                break

        if n_used == 0:
            print("[mms-style] No valid cubes for dN/dE accumulation; skipping plot.")
        else:
            dNdE_mean = dNdE_accum / float(n_used)
            denom = float(m_s) * (float(c_light) ** 2)
            x_cent = np.where(denom > 0, E_cent / denom, np.nan)
            # after dNdE_mean is computed
            mx = np.nanmax(dNdE_mean)
            if mx > 0:
                dNdE_mean /= mx

            figE, axE = plt.subplots(1, 1, figsize=(7.5, 5.0))
            m = np.isfinite(x_cent) & np.isfinite(dNdE_mean) & (x_cent > 0) & (dNdE_mean > 0)
            if m.any():
                axE.loglog(x_cent[m], dNdE_mean[m], label="proxy: dN/dE")
            else:
                axE.plot(x_cent, dNdE_mean, label="proxy: dN/dE")

            # Optional Maxwellian overlay using isotropic pressure estimate
            if overlay_maxwellian_fit and ('Pxx' in globals()) and (Pxx is not None) and (Pyy is not None) and (Pzz is not None):
                center0 = pts[int(mms_ids[0])]
                pxx_cube = extract_cube_around_point(Pxx, center0, spacing_rho_cube, origin, int(spectrum_box_n))
                pyy_cube = extract_cube_around_point(Pyy, center0, spacing_rho_cube, origin, int(spectrum_box_n))
                pzz_cube = extract_cube_around_point(Pzz, center0, spacing_rho_cube, origin, int(spectrum_box_n))
                n0_cube  = extract_cube_around_point(N,   center0, spacing_rho_cube, origin, int(spectrum_box_n))

                P_iso = (pxx_cube + pyy_cube + pzz_cube) / 3.0
                if _spectrum_mask_rho is not None:
                    P_iso = P_iso[_spectrum_mask_rho]
                    n0_cube = n0_cube[_spectrum_mask_rho]

                goodT = np.isfinite(P_iso) & np.isfinite(n0_cube) & (n0_cube > 0) & (P_iso > 0)
                if np.any(goodT):
                    T_est = float(np.median(P_iso[goodT] / n0_cube[goodT]))
                    ffit = maxwellian_dNdE(E_cent, T_est)

                    # Scale fit to match mid-range of the measured curve
                    mm = m.copy()
                    if np.any(mm):
                        mid = int(np.clip(np.sum(mm) // 2, 0, np.sum(mm) - 1))
                        idx_mid = np.where(mm)[0][mid]
                        scale = dNdE_mean[idx_mid] / max(ffit[idx_mid], 1e-300)
                        axE.loglog(x_cent[m], (scale * ffit)[m], linestyle="--", label=f"Maxwellian (T≈{T_est:.3g})")

            title_bits = ["MMS-style energy spectrum (moment-based proxy)"]
            if sp is not None:
                title_bits.append(f"species {species_idx} ({sp['population']}, QOM={sp['qom']}, m≈{m_s:.3g})")
            else:
                title_bits.append(f"species {species_idx} (m≈{m_s:.3g})")
            title_bits.append(r"x = E/(m c^2),  E = 0.5 m |J/\rho|^2")
            if average_spectrum_over_path:
                title_bits.append(f"avg over {n_used} probe centers")
            axE.set_title("\n".join(title_bits))

            axE.set_xlabel(r"$E/(m c^2)$")
            axE.set_ylabel(r"$dN/dE$ (normalized)")
            axE.grid(True, which="both", alpha=0.3)
            axE.legend(frameon=False)
            figE.tight_layout()
            figE.savefig(mms_energy_out_png, dpi=200)
            plt.close(figE)
            print(f"[mms-style] Wrote {mms_energy_out_png} (avg over {n_used}).")


# =========================
# Matplotlib figure (progressive reveal)
# =========================
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax0, ax1 = axes

# Pre-set limits (prevents rescaling flicker)
ax0.set_xlim(x_axis[0], x_axis[-1])
ax0.set_ylim(min(Bx_ts.min(), By_ts.min(), Bz_ts.min())*1.1, max(Bx_ts.max(), By_ts.max(), Bz_ts.max())*1.1)
ax1.set_xlim(x_axis[0], x_axis[-1])
ax1.set_ylim(min(Bmag_ts.min(), rho_ts.min())*1.1, max(Bmag_ts.max(), rho_ts.max())*1.1)

# Lines that will grow
(line_bx,) = ax0.plot([], [], label="Bx")
(line_by,) = ax0.plot([], [], label="By")
(line_bz,) = ax0.plot([], [], label="Bz")
cursor0 = ax0.axvline(x_axis[0], linewidth=1)
ax0.set_ylabel("B comps")
ax0.legend(frameon=False, ncol=3, loc="upper right")

(line_bmag,) = ax1.plot([], [], label="|B|")
(line_rho,)  = ax1.plot([], [], label=scalar_name)
cursor1 = ax1.axvline(x_axis[0], linewidth=1)
ax1.set_ylabel(f"|B|, {scalar_name}")
ax1.set_xlabel("Distance along path")
ax1.legend(frameon=False, ncol=2, loc="upper right")

fig.tight_layout()


# =========================
# PyVista scene (single canvas: slice + glyphs + probe)
# =========================
pv.global_theme.multi_samples = 0  # helps on some systems for screenshots
pl = pv.Plotter(window_size=(1100, 700), off_screen=True)
_title = f"Slice ({scalar_name}) + B glyphs + probe"
if fixed_axis is not None:
    _title += f"  |  {fixed_axis.upper()} constant = {fixed_val:.6g}"
pl.add_text(_title, font_size=12)
pl.show_axes()

# --- Add a 2D outline around the valid domain on the (nearly) constant plane, if detected ---
# This is intended as a visual frame-of-reference for start/end locations without obscuring data.
if fixed_axis is not None:
    outline = make_plane_outline(fixed_axis, fixed_val, (xmin, xmax), (ymin, ymax), (zmin, zmax))
    pl.add_mesh(outline, line_width=3)

    # Corner labels (kept short); these sit on the outline, not over the data interior.
    lab_pts, lab_txt = plane_outline_labels(fixed_axis, fixed_val, (xmin, xmax), (ymin, ymax), (zmin, zmax))
    pl.add_point_labels(
        lab_pts,
        lab_txt,
        always_visible=True,
        shape=None,
        font_size=12,
        point_size=0,
    )

# Create slice and glyphs (sample B onto slice points so glyph orientation works)
def make_slice(origin_xyz, normal_vec):
    # Slice scalar grid for the cross-section
    sl = grid.slice(normal=normal_vec, origin=origin_xyz)

    # Sample vectors from grid_v onto the slice points
    sl_v = sl.sample(grid_v)

    if sl_v.n_points == 0:
        return sl, None, None

    # Optionally compute |B| on the slice for contouring
    if "Bmag" not in sl_v.point_data:
        try:
            btmp = np.asarray(sl_v.get_array("B", preference="point"))
            if btmp.ndim == 2 and btmp.shape[1] == 3 and btmp.shape[0] == sl_v.n_points:
                sl_v.point_data["Bmag"] = np.linalg.norm(btmp, axis=1)
        except Exception:
            pass

    # --- Contours of |B| on the plane (optional) ---
    contours = None
    if show_B_contours and "Bmag" in sl_v.point_data and sl_v.n_points > 0:
        try:
            contours = sl_v.contour(isosurfaces=int(n_B_contours), scalars="Bmag")
        except Exception:
            contours = None

    # --- Glyphs on the plane (optional) ---
    glyphs = None
    if show_glyphs and sl_v.n_points > 0:
        # IMPORTANT: Avoid VTK association issues by creating a fresh PolyData
        # that has ONLY point-associated vectors.
        b_arr = None
        try:
            b_arr = sl_v.get_array("B", preference="point")
        except Exception:
            b_arr = None
        if b_arr is None:
            try:
                b_arr = sl_v.get_array("B", preference="cell")
            except Exception:
                b_arr = None
        if b_arr is None:
            try:
                b_arr = sl_v.get_array("B", preference="field")
            except Exception:
                b_arr = None

        if b_arr is not None:
            b_arr = np.asarray(b_arr)

            # Prefer point-associated vectors
            if b_arr.ndim == 2 and b_arr.shape[1] == 3 and b_arr.shape[0] == sl_v.n_points:
                pts_for_glyph = sl_v.points
                b_for_glyph = b_arr
            # Fallback: if vectors are per-cell, place glyphs at cell centers
            elif b_arr.ndim == 2 and b_arr.shape[1] == 3 and b_arr.shape[0] == getattr(sl_v, "n_cells", -1):
                cc = sl_v.cell_centers()
                pts_for_glyph = cc.points
                b_for_glyph = b_arr
            else:
                pts_for_glyph = None
                b_for_glyph = None

            if pts_for_glyph is not None:
                # Randomly sample points across the plane so glyphs don't concentrate on an edge.
                rng = np.random.default_rng(rng_seed)
                n_pts = pts_for_glyph.shape[0]
                n_take = min(int(glyph_max_points), int(n_pts))
                if n_take > 0:
                    pick = rng.choice(n_pts, size=n_take, replace=False)
                    glyph_pts = pv.PolyData(pts_for_glyph[pick])
                    glyph_pts.point_data["Bvec"] = b_for_glyph[pick]
                    glyphs = glyph_pts.glyph(orient="Bvec", scale=False, factor=glyph_scale)

    return sl, glyphs, contours

# Initial slice aligned with the probe direction (like the sketch)
origin0, normal0 = plane_from_probe(pts, 0)
slice_mesh, glyph_mesh, contour_mesh = make_slice(origin0, normal0)

# Add slice colored by rho_0 if available
if scalar_name in slice_mesh.point_data:
    slice_actor = pl.add_mesh(slice_mesh, scalars=scalar_name, cmap="Purples", opacity=1.0)
else:
    slice_actor = pl.add_mesh(slice_mesh, opacity=1.0)

glyph_actor = None
if glyph_mesh is not None and glyph_mesh.n_points > 0:
    glyph_actor = pl.add_mesh(glyph_mesh, opacity=1.0)

contour_actor = None
if contour_mesh is not None and contour_mesh.n_points > 0:
    # draw contours as lines
    contour_actor = pl.add_mesh(contour_mesh, line_width=2)

# Probe + trail
probe_center0 = pts[0]

sphere = pv.Sphere(radius=probe_radius, center=(0, 0, 0))
probe_actor = pl.add_mesh(sphere, opacity=1.0)
probe_actor.SetPosition(*probe_center0)

trail = pv.PolyData(pts[:1].copy())
trail.lines = np.array([1, 0], dtype=np.int64)  # dummy
trail_actor = pl.add_mesh(trail, opacity=1.0)

pl.camera.zoom(1.2)
pl.render()


# =========================
# Frame rendering loop
# =========================
os.makedirs(out_dir, exist_ok=True)
if clear_existing_frames:
    for fn in os.listdir(out_dir):
        if fn.startswith("frame_") and fn.endswith(".png"):
            try:
                os.remove(os.path.join(out_dir, fn))
            except OSError:
                pass

for frame_i, idx in enumerate(frame_ids):
    x, y, z = pts[idx]

    # Move probe via actor.SetPosition
    probe_actor.SetPosition(x, y, z)

    # Update slice plane to contain the probe direction and pass through the probe
    origin_i, normal_i = plane_from_probe(pts, idx)

    if follow_plane_camera:
        # Tangent for this frame (match the one used in plane_from_probe)
        if idx < len(pts) - 1:
            t_i = pts[idx + 1] - pts[idx]
        else:
            t_i = pts[idx] - pts[idx - 1]
        pos, focal, vup = camera_pose_for_plane(origin_i, normal_i, t_i)
        pl.camera.position = tuple(pos)
        pl.camera.focal_point = tuple(focal)
        pl.camera.up = tuple(vup)

    # Rebuild slice + glyphs and replace the actors
    new_slice, new_glyphs, new_contours = make_slice(origin_i, normal_i)

    # Replace slice actor
    pl.remove_actor(slice_actor, reset_camera=False)
    if scalar_name in new_slice.point_data:
        slice_actor = pl.add_mesh(new_slice, scalars=scalar_name, cmap="Purples", opacity=1.0)
    else:
        slice_actor = pl.add_mesh(new_slice, opacity=1.0)

    # Replace glyph actor
    if glyph_actor is not None:
        pl.remove_actor(glyph_actor, reset_camera=False)
        glyph_actor = None
    if new_glyphs is not None and new_glyphs.n_points > 0:
        glyph_actor = pl.add_mesh(new_glyphs, opacity=1.0)

    # Replace contour actor
    if contour_actor is not None:
        pl.remove_actor(contour_actor, reset_camera=False)
        contour_actor = None
    if new_contours is not None and new_contours.n_points > 0:
        contour_actor = pl.add_mesh(new_contours, line_width=2)

    # Update trail geometry
    seg_pts = pts[:idx+1]
    if seg_pts.shape[0] >= 2:
        cells = np.hstack([[seg_pts.shape[0]], np.arange(seg_pts.shape[0])]).astype(np.int64)
        trail.points = seg_pts
        trail.lines = cells
        trail.Modified()

    # Optional nicer trail (tube) — if too slow, disable use_tube_trail
    if use_tube_trail and seg_pts.shape[0] >= 2:
        pl.remove_actor(trail_actor, reset_camera=False)
        trail_actor = pl.add_mesh(trail.tube(radius=trail_tube_radius), opacity=1.0)

    # Update time series reveal up to idx
    xs = x_axis[:idx+1]
    line_bx.set_data(xs, Bx_ts[:idx+1])
    line_by.set_data(xs, By_ts[:idx+1])
    line_bz.set_data(xs, Bz_ts[:idx+1])

    line_bmag.set_data(xs, Bmag_ts[:idx+1])
    line_rho.set_data(xs, rho_ts[:idx+1])

    xcur = x_axis[idx]
    cursor0.set_xdata([xcur, xcur])
    cursor1.set_xdata([xcur, xcur])

    # Render both to images and combine
    pl.render()
    pv_rgb = pl.screenshot(return_img=True)  # HxWx3 uint8

    ts_rgb = mpl_fig_to_rgb(fig)

    combo = hstack_images(pv_rgb, ts_rgb)

    out_path = os.path.join(out_dir, f"frame_{frame_i:06d}.png")
    # save with matplotlib imsave (no extra deps)
    plt.imsave(out_path, combo)

    if (frame_i + 1) % 50 == 0:
        print(f"Wrote {frame_i+1}/{len(frame_ids)} frames...")

print("Done frames.")
print(f"Now run ffmpeg on the output directory, e.g.:\n"
      f"ffmpeg -r {fps} -i frame_%06d.png -pix_fmt yuv420p -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' out.mp4")


# =========================
# Final: energy spectra plot (single figure)
# =========================
if compute_energy_spectra:
    # Choose which probe samples to use
    if use_spectrum_frame_ids:
        spec_ids = frame_ids
    else:
        spec_ids = np.arange(len(pts), dtype=int)

    # Accumulators
    kK_accum = None
    sK_accum = None
    kM_accum = None
    sM_accum = None
    n_used_K = 0
    n_used_M = 0

    for idx in spec_ids:
        center = pts[idx]

        # -------------------------
        # Kinetic energy spectrum
        # -------------------------
        if plot_kinetic_energy_spectrum:
            if (N is None) or (Jx is None) or (Jy is None) or (Jz is None):
                # Required species moments not available
                pass
            else:
                # Extract cubes on rho/J/N grid
                rho_cube = extract_cube_around_point(rho, center, spacing_rho_cube, origin, int(spectrum_box_n))
                n_cube = extract_cube_around_point(N, center, spacing_rho_cube, origin, int(spectrum_box_n))
                jx_cube = extract_cube_around_point(Jx, center, spacing_rho_cube, origin, int(spectrum_box_n))
                jy_cube = extract_cube_around_point(Jy, center, spacing_rho_cube, origin, int(spectrum_box_n))
                jz_cube = extract_cube_around_point(Jz, center, spacing_rho_cube, origin, int(spectrum_box_n))

                # Velocity proxy: v = J / rho (valid if rho is charge density q n)
                eps = 1e-12
                den = np.where(np.abs(rho_cube) > eps, rho_cube, np.sign(rho_cube) * eps + (rho_cube == 0) * eps)
                vx_cube = jx_cube / den
                vy_cube = jy_cube / den
                vz_cube = jz_cube / den

                # Kinetic energy density proxy (mass factor omitted / assumed 1): 0.5 * N * |v|^2
                ked = 0.5 * n_cube * (vx_cube * vx_cube + vy_cube * vy_cube + vz_cube * vz_cube)

                # Apply spherical mask
                if _spectrum_mask_rho is not None:
                    ked = np.where(_spectrum_mask_rho, ked, 0.0).astype(np.float32, copy=False)

                # Remove DC inside mask
                if _spectrum_mask_rho is not None and _spectrum_mask_rho.any():
                    mvals = ked[_spectrum_mask_rho]
                    if mvals.size > 0:
                        ked[_spectrum_mask_rho] = ked[_spectrum_mask_rho] - float(mvals.mean())
                else:
                    ked = ked - float(ked.mean())

                kK, sK = isotropic_spectrum_3d(ked, spacing_rho_cube)

                if kK_accum is None:
                    kK_accum = kK
                    sK_accum = np.zeros_like(sK, dtype=np.float64)

                if kK.shape == kK_accum.shape and np.allclose(kK, kK_accum):
                    sK_accum += sK.astype(np.float64, copy=False)
                    n_used_K += 1

        # -------------------------
        # Magnetic energy spectrum
        # -------------------------
        if plot_magnetic_energy_spectrum:
            # Extract cubes on B grid (note: B arrays were loaded with v_stride)
            bx_cube = extract_cube_around_point(Bx, center, spacing_B, origin, int(spectrum_box_n))
            by_cube = extract_cube_around_point(By, center, spacing_B, origin, int(spectrum_box_n))
            bz_cube = extract_cube_around_point(Bz, center, spacing_B, origin, int(spectrum_box_n))

            # Magnetic energy density: 0.5 * |B|^2  (mu0 omitted)
            med = 0.5 * (bx_cube * bx_cube + by_cube * by_cube + bz_cube * bz_cube)

            # Apply spherical mask
            if _spectrum_mask_B is not None:
                med = np.where(_spectrum_mask_B, med, 0.0).astype(np.float32, copy=False)

            # Remove DC inside mask
            if _spectrum_mask_B is not None and _spectrum_mask_B.any():
                mvals = med[_spectrum_mask_B]
                if mvals.size > 0:
                    med[_spectrum_mask_B] = med[_spectrum_mask_B] - float(mvals.mean())
            else:
                med = med - float(med.mean())

            kM, sM = isotropic_spectrum_3d(med, spacing_B)

            if kM_accum is None:
                kM_accum = kM
                sM_accum = np.zeros_like(sM, dtype=np.float64)

            if kM.shape == kM_accum.shape and np.allclose(kM, kM_accum):
                sM_accum += sM.astype(np.float64, copy=False)
                n_used_M += 1

        if not average_spectrum_over_path:
            break

    # Finalize means
    sK_mean = None
    sM_mean = None
    if plot_kinetic_energy_spectrum and n_used_K > 0:
        sK_mean = sK_accum / float(n_used_K)
    if plot_magnetic_energy_spectrum and n_used_M > 0:
        sM_mean = sM_accum / float(n_used_M)

    # Build subtitle with probe/averaging info
    subtitle_bits = [f"sphere mask r={probe_radius} (rho-grid)"]
    if _spectrum_mask_B is not None:
        subtitle_bits.append(f"B-mask r={probe_radius_B:.3g} (B-grid)")
    else:
        subtitle_bits.append("B-mask: none")
    if average_spectrum_over_path:
        subtitle_bits.append(f"avg over K:{n_used_K}  M:{n_used_M}")
    subtitle = "\n".join(subtitle_bits)

    sp = None
    if species_idx is not None:
        sp = SPECIES_META.get(str(species_idx), None)

    # ---- Kinetic figure ----
    if sK_mean is not None:
        figK, axK = plt.subplots(1, 1, figsize=(7, 5))
        m = (kK_accum > 0) & (sK_mean > 0)
        if m.any():
            axK.loglog(kK_accum[m], sK_mean[m])
        else:
            axK.plot(kK_accum, sK_mean)

        axK.set_xlabel(r"$|k|$ (rad / unit length)")
        axK.set_ylabel(r"$\langle |\hat{\varepsilon}_K(k)|^2 \rangle$ (arb.)")

        title_bits = ["Local isotropic kinetic-energy spectrum"]
        if sp is not None:
            title_bits.append(f"species {species_idx} ({sp['population']}, QOM={sp['qom']})")
        else:
            title_bits.append(f"species {species_idx} (from {scalar_name})")
        title_bits.append(r"$\varepsilon_K = 0.5\,N\,|J/\rho|^2$ (m=1)")
        axK.set_title("\n".join(title_bits) + "\n" + subtitle)

        axK.grid(True, which="both", alpha=0.3)
        figK.tight_layout()
        figK.savefig(kinetic_spectrum_out_png, dpi=200)
        plt.close(figK)
        print(f"[energy spectra] Wrote {kinetic_spectrum_out_png} (K:{n_used_K}).")

    # ---- Magnetic figure ----
    if sM_mean is not None:
        figM, axM = plt.subplots(1, 1, figsize=(7, 5))
        m = (kM_accum > 0) & (sM_mean > 0)
        if m.any():
            axM.loglog(kM_accum[m], sM_mean[m])
        else:
            axM.plot(kM_accum, sM_mean)

        axM.set_xlabel(r"$|k|$ (rad / unit length)")
        axM.set_ylabel(r"$\langle |\hat{\varepsilon}_B(k)|^2 \rangle$ (arb.)")

        title_bits = ["Local isotropic magnetic-energy spectrum"]
        title_bits.append(r"$\varepsilon_B = 0.5\,|B|^2$ ($\mu_0$ omitted)")
        axM.set_title("\n".join(title_bits) + "\n" + subtitle)

        axM.grid(True, which="both", alpha=0.3)
        figM.tight_layout()
        figM.savefig(magnetic_spectrum_out_png, dpi=200)
        plt.close(figM)
        print(f"[energy spectra] Wrote {magnetic_spectrum_out_png} (M:{n_used_M}).")