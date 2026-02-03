import h5py, numpy as np, pyvista as pv, vtk

# path = "../T3D-Fields_005500.h5"
path = "../../DoubleGEM-Fields_300000.h5" # "../T3D-Fields_005500.h5"
scalar_name = "rho_0"
bx_name, by_name, bz_name = "Bx", "By", "Bz"

# geometry from SimulationData.txt
DX, DY, DZ = 0.125, 0.125, 0.125
X0, Y0, Z0 = 0.0, 0.0, 0.0

# strides (coarser if slow)
s_stride = 1
v_stride = 4

def open_var(f, name):
    obj = f[f"Step#0/Block/{name}"]
    if isinstance(obj, h5py.Group):
        return obj["0"] if "0" in obj else next(v for v in obj.values() if isinstance(v, h5py.Dataset))
    return obj

# ---- load density ----
with h5py.File(path, "r") as f:
    rho_ds = open_var(f, scalar_name)
    rho = rho_ds[::s_stride, ::s_stride, ::s_stride].astype(np.float32)

nz, ny, nx = rho.shape
grid = pv.ImageData()
grid.dimensions = (nx, ny, nz)
grid.spacing    = (DX*s_stride, DY*s_stride, DZ*s_stride)
grid.origin     = (X0, Y0, Z0)

if rho.size == grid.n_points:
    grid.point_data[scalar_name] = rho.ravel(order="C")
    grid_point = grid
else:
    grid.cell_data[scalar_name] = rho.ravel(order="C")
    grid_point = grid.cell_data_to_point_data()

# ranges (print to terminal)
rmin, rmax = float(rho.min()), float(rho.max())
p = np.percentile(rho, [1,5,50,95,99])
print("rho stats:", rmin, rmax, p)

# ---- load B (coarser) ----
with h5py.File(path, "r") as f:
    Bx = open_var(f, bx_name)[::v_stride, ::v_stride, ::v_stride].astype(np.float32)
    By = open_var(f, by_name)[::v_stride, ::v_stride, ::v_stride].astype(np.float32)
    Bz = open_var(f, bz_name)[::v_stride, ::v_stride, ::v_stride].astype(np.float32)

nxv, nyv, nzv = Bx.shape # | if broken, try nzv, nyv, nxv
grid_v = pv.ImageData()
grid_v.dimensions = (nxv, nyv, nzv)
grid_v.spacing    = (DX*v_stride, DY*v_stride, DZ*v_stride)
grid_v.origin     = (X0, Y0, Z0)

B = np.stack([Bx, By, Bz], axis=-1)
Bmag = np.linalg.norm(B, axis=-1)
grid_v.point_data["B"]    = B.reshape(-1, 3, order="C")
grid_v.point_data["Bmag"] = Bmag.ravel(order="C")

bmin, bmax = float(Bmag.min()), float(Bmag.max())
bp = np.percentile(Bmag, [1,5,50,95,99])
print("|B| stats:", bmin, bmax, bp)

# Use broad clims first so we SEE something; tighten later
rho_clim = (rmin, rmax)
B_clim   = (bmin, bmax)

pl = pv.Plotter(window_size=(1300, 950))
pl.enable_depth_peeling()
pl.show_grid(location="outer", ticks="both")
pl.add_axes(interactive=True)
pl.add_box_axes()



# --- density volume with safe preset opacity ---
vol_actor = pl.add_volume(
    grid_point, scalars=scalar_name,
    cmap="Purples", clim=rho_clim,
    opacity='sigmoid_6',  # robust default#
    shade=False,
    scalar_bar_args={'title': scalar_name}
)

#rho_lo, rho_hi = np.percentile(rho, [5, 99.5])
#tf = np.linspace(0, 0.8, 256)**2.2   # more transparent at low values
#vol_actor = pl.add_volume(
#    grid_point, scalars=scalar_name, cmap="Purples",
#    clim=(rho_lo, rho_hi), opacity=tf, shade=False,
#    scalar_bar_args={'title': scalar_name},
#)

# --- set up a single VTK clipping plane attached to the volume mapper ---
clip_plane = vtk.vtkPlane()
origin0 = list(grid_v.center)
normal0 = [0.0, 0.0, 1.0]
clip_plane.SetOrigin(*origin0)
clip_plane.SetNormal(*normal0)
# keep the back half initially
vol_actor.mapper.RemoveAllClippingPlanes()
vol_actor.mapper.AddClippingPlane(clip_plane)
KEEP_BACK = True  # flip with widget or key if needed

# --- interactive plane + glyphs ---
current_plane = {"actor": None}
current_arrows = {"actor": None}

# Make arrows visibly larger and never vanish: add a baseline to scaling
BASE_FACTOR = 2.0 * v_stride * min(DX, DY, DZ)  # bump if still small
BASELINE_SCALE = 0.5                            # ensures nonzero length

def plane_and_glyphs(normal, origin):
    if current_plane["actor"] is not None:
        pl.remove_actor(current_plane["actor"], reset_camera=False)
    if current_arrows["actor"] is not None:
        pl.remove_actor(current_arrows["actor"], reset_camera=False)

    # update the single clipping plane (fast): keep one half-space relative to the plane
    nn = np.array(normal, float)
    if not KEEP_BACK:
        nn = -nn
    clip_plane.SetOrigin(*(float(x) for x in origin))
    clip_plane.SetNormal(*(float(x) for x in nn))

    slc = grid_v.slice(normal=normal, origin=origin)
    plane_actor = pl.add_mesh(slc, color=(0.2, 0.6, 0.3), opacity=0.25)
    current_plane["actor"] = plane_actor

    # Build a custom scale array: normalized |B| + baseline
    b = slc["Bmag"]
    if b is None or len(b) == 0:
        return
    # normalize to 0..1 using wide bounds, then add baseline
    bnorm = (b - B_clim[0]) / max(B_clim[1] - B_clim[0], 1e-12)
    bnorm = np.clip(bnorm, 0.0, 1.0)
    slc.point_data["Bscale"] = BASELINE_SCALE + bnorm

    # sparsify if too dense
    if slc.n_points > 150_000:
        slc = slc.extract_points(range(0, slc.n_points, 3))

    arrows = slc.glyph(
        orient="B",
        scale="Bscale",      # uses our baseline-boosted scale
        factor=BASE_FACTOR
    )
    arrow_actor = pl.add_mesh(
        arrows,
        scalars="Bmag", cmap="viridis", clim=B_clim,
        lighting=True,
        scalar_bar_args={'title': '|B|'}
    )
    current_arrows["actor"] = arrow_actor

pl.add_plane_widget(
    plane_and_glyphs,
    normal=(0, 0, 1),
    origin=grid_v.center,
    normal_rotation=True
)

# print("dims:", grid_point.dimensions)
# print("spacing:", grid_point.spacing)
# print("bounds:", grid_point.bounds)  # expect (x0,x1,y0,y1,z0,z1) ≈ (0,16, 0,16, 0,24)

# The only slice shown is the one driven by the plane widget (added in plane_and_glyphs)

# initialize once so glyphs & clipping are visible immediately
plane_and_glyphs((0, 0, 1), grid_v.center)

# press 'f' to flip kept side
def flip_side():
    global KEEP_BACK
    KEEP_BACK = not KEEP_BACK
    plane_and_glyphs((0, 0, 1), grid_v.center)
pl.add_key_event('f', flip_side)  # press 'f' to flip kept side

pl.camera.zoom(1.2)
pl.show()
