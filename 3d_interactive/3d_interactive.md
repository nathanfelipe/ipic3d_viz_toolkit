# 3D interactive visualization for iPIC3D / ECSIM HDF5 fields (`3d_interactive.py`)

This script provides an **interactive 3D viewer** for scalar and vector fields stored in iPIC3D/ECSIM-style HDF5 output.  
It uses **PyVista/VTK** to render:

- a **volume rendering** of a scalar field (default: `rho_0`)
- an **interactive slicing plane** through a vector field (default: **B** = `Bx, By, Bz`)
- **arrow glyphs** on the plane, oriented with **B** and colored/scaled by **|B|**
- a **clipping plane** applied to the volume so you can “see inside” and correlate structures with the slice

---

## Dependencies

Install (typical options):

```bash
pip install numpy h5py pyvista vtk
```

Notes:
- On some systems, `pyvista` will pull in `vtk`; if not, install `vtk` explicitly.
- For best interactivity, use a machine with a decent GPU and working OpenGL drivers.

---

## Input data assumptions (HDF5 layout)

The script expects datasets under paths like:

- `Step#0/Block/<name>`

It also supports the case where `<name>` is an HDF5 **group** containing one or more datasets (common in some outputs).  
In that case, it chooses:
- the dataset named `"0"` if it exists, otherwise
- the first dataset found inside the group.

This logic is implemented in:

```python
def open_var(f, name):
    obj = f[f"Step#0/Block/{name}"]
    if isinstance(obj, h5py.Group):
        return obj["0"] if "0" in obj else next(v for v in obj.values() if isinstance(v, h5py.Dataset))
    return obj
```

---

## What the script does

### 1) Load a scalar field and render it as a volume

- Opens `scalar_name` (default: `rho_0`)
- Applies a stride (`s_stride`) to decimate the grid for speed
- Creates a `pyvista.ImageData` grid
- Attaches the scalar to either **point data** or **cell data**, depending on array size
- Converts cell-data scalars to point data if needed (volume rendering expects point data in many workflows)

Key steps:

- **Striding (decimation)**:
  - `rho = rho_ds[::s_stride, ::s_stride, ::s_stride]`
- **Grid geometry** (from `SimulationData.txt` in your simulation):
  - spacing: `(DX*s_stride, DY*s_stride, DZ*s_stride)`
  - origin: `(X0, Y0, Z0)`
- **Point vs cell data** decision:
  - If `rho.size == grid.n_points`: store in `grid.point_data`
  - Else: store in `grid.cell_data`, then convert to points with `cell_data_to_point_data()`

It prints basic statistics for sanity checking:

- min/max
- selected percentiles (1, 5, 50, 95, 99)

### 2) Load a vector field (B) on a coarser grid

- Loads `Bx`, `By`, `Bz` using a separate stride (`v_stride`) for speed
- Builds a second `pyvista.ImageData` grid (`grid_v`)
- Stacks the components into an `N×3` array called `B`
- Computes magnitude:

$
|B| = \sqrt{B_x^2 + B_y^2 + B_z^2}
$

Implementation:

```python
B    = np.stack([Bx, By, Bz], axis=-1)
Bmag = np.linalg.norm(B, axis=-1)
```

It stores:
- `grid_v.point_data["B"]` (vector)
- `grid_v.point_data["Bmag"]` (scalar)

Again, prints `|B|` statistics and percentiles.

### 3) Volume rendering

The scalar field is rendered with:

```python
pl.add_volume(
    grid_point, scalars=scalar_name,
    cmap="Purples", clim=(rmin, rmax),
    opacity="sigmoid_6",
    shade=False,
    scalar_bar_args={"title": scalar_name},
)
```

**Opacity**: `sigmoid_6` is a robust preset that tends to make low values more transparent and high values more opaque.  
You can replace this with a custom transfer function (an example is included but commented out).

### 4) A single clipping plane applied to the volume (VTK plane)

A `vtkPlane` is created once and attached to the volume mapper:

```python
clip_plane = vtk.vtkPlane()
vol_actor.mapper.RemoveAllClippingPlanes()
vol_actor.mapper.AddClippingPlane(clip_plane)
```

When you move/rotate the interactive plane widget, the callback updates the **same** `clip_plane` in-place (fast).

The boolean `KEEP_BACK` controls which half-space is kept:
- `KEEP_BACK = True` keeps one side
- flipping it negates the plane normal

### 5) Interactive slice plane + B-field arrows (glyphs)

The plane widget triggers `plane_and_glyphs(normal, origin)`:

1. **Update clipping plane**: (also affected by `KEEP_BACK`)
2. **Slice** the coarse vector grid: `slc = grid_v.slice(normal=normal, origin=origin)`
3. Render the slice surface (semi-transparent)
4. Add arrow glyphs oriented by `B`

#### Glyph scaling (important detail)

The script creates a **custom scale array** called `Bscale` so arrows never vanish:

1. Normalize `|B|` into `[0, 1]` using the *global* `B_clim = (min, max)`:

$$
b_\mathrm{norm} = \mathrm{clip}\left(\frac{b - b_{min}}{b_{max} - b_{min}}, 0, 1 \right)
$$

2. Add a constant baseline so scale is never zero:

$$
Bscale = \mathrm{BASELINE\_SCALE} + b_\mathrm{norm}
$$

3. Multiply by a global `factor` (units roughly in *grid spacing*), to control arrow length:

```python
BASE_FACTOR = 2.0 * v_stride * min(DX, DY, DZ)
BASELINE_SCALE = 0.5
```

Then:

```python
arrows = slc.glyph(
    orient="B",
    scale="Bscale",
    factor=BASE_FACTOR
)
```

Arrows are colored by `Bmag` using `viridis`.

#### Downsampling the slice if it’s too dense

If the slice contains more than 150k points, it keeps every 3rd point:

```python
if slc.n_points > 150_000:
    slc = slc.extract_points(range(0, slc.n_points, 3))
```

---

## Controls

- **Plane widget**:
  - drag to translate
  - rotate to change normal (when `normal_rotation=True`)

- **Key bindings**:
  - Press **`f`** to flip which side of the clipping plane is kept.

---

## Configuration (edit at top of file)

```python
path = "../../DoubleGEM-Fields_300000.h5"
scalar_name = "rho_0"
bx_name, by_name, bz_name = "Bx", "By", "Bz"

DX, DY, DZ = 0.125, 0.125, 0.125
X0, Y0, Z0 = 0.0, 0.0, 0.0

s_stride = 1
v_stride = 4
```

Recommended workflow:
- start with larger strides (e.g., `s_stride=2`, `v_stride=8`) to confirm everything loads and renders
- reduce stride for higher fidelity once performance is acceptable

---

## Minimal “how to run”

```bash
python 3d_interactive.py
```

If you bundle this into a package, you can expose it as a console entry point, e.g.:

```bash
ipic3d-viewer --file DoubleGEM-Fields_300000.h5 --scalar rho_0 --vector Bx By Bz
```

---

## Output

When the script runs successfully, you get an interactive VTK window:
- purple volume = scalar field
- green transparent plane = slice
- arrows on plane = B vectors (colored by |B|)
- scalar bars for both scalar and |B|
