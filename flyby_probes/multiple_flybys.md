# Multiple synthetic flybys with MMS-like tetrahedron sampling (`multiple_flybys.py`)

This script generates an **ensemble of synthetic spacecraft flybys** through a single 3D iPIC3D/ECSIM snapshot and writes **per-flyby CSVs** plus **per-flyby figures**.

Compared to `live_flyby_slice_glyphs.py` (one animated flyby), this tool focuses on **batch sampling + comparison**:

- many random straight-line paths (`n_flybys`)
- an **MMS-like 4-probe tetrahedron** (4 nearby trajectories per flyby)
- sampling of **many fields** along those paths (density, **B**, optional **E**, currents, pressure tensors, derived velocities)
- exporting time series to CSV for downstream analysis

---

## Dependencies

```bash
pip install numpy h5py matplotlib
```

---

## Input data assumptions (HDF5 layout)

The script expects fields under:

- `Step#0/Block/<name>`

It supports the case where `<name>` is an HDF5 **group** containing datasets (it prefers dataset `"0"` if present).

At startup it prints *every dataset path in the file* via `list_all_fields(path)`. This is intentional: iPIC3D/ECSIM naming conventions differ between runs, and the printout helps you map aliases quickly.

---

## Configuration you’ll edit

At the top of the script:

- File: `path = "../../DoubleGEM-Fields_300000.h5"`
- Density-like field: `rho_name = "rho_0"`
- Magnetic field: `bx_name, by_name, bz_name = "Bx", "By", "Bz"`
- Electric field (optional): `ex_name, ey_name, ez_name = "Ex", "Ey", "Ez"` (skipped if not found)

Flyby sampling:
- `n_flybys = 5`
- `n_samples = 2001`

MMS-like tetrahedron:
- `MMS_separation` sets typical inter-spacecraft spacing (simulation units)
- `probe_to_plot` selects which probe (1..4) is emphasized in plots

Output directories:
- CSV: `flyby_csvs/`
- Figures: `flyby_figs/`

---

## What fields are loaded

### Always loaded
- Charge densities: `rho_0`, `rho_1`
- Magnetic field: `Bx, By, Bz`
- Currents for both species:
  - `Jx_0, Jy_0, Jz_0`
  - `Jx_1, Jy_1, Jz_1`
- Full pressure tensor for both species:
  - `Pxx_s, Pyy_s, Pzz_s, Pxy_s, Pxz_s, Pyz_s` for `s ∈ {0,1}`

### Loaded if present (alias search)
- Electric field components (`Ex, Ey, Ez`) using multiple common aliases
- Bulk velocity components (`Vx, Vy, Vz`) using multiple common aliases

The helper `find_first_available(f, aliases)` tries a list of candidate names and returns the first dataset that exists. This makes the script robust across different output conventions.

---

## Core geometry: many flybys + a 4-probe tetrahedron

### 1) Random straight-line flyby definition
For each flyby `k`, the script draws two random endpoints inside the domain:

- `p0 ~ Uniform([X0, BOX_X] × [Y0, BOX_Y] × [Z0, BOX_Z])`
- `p1 ~ Uniform(...)`

Then it samples points along the line:

$$
\mathbf{x}(t) = (1-t)\,\mathbf{p}_0 + t\,\mathbf{p}_1,\qquad t\in[0,1]
$$

with `n_samples` evenly spaced values of (t).

It constructs a distance-like coordinate for plotting:

$$
d_i = \|\mathbf{x}_i - \mathbf{x}_0\|.
$$

Optionally, if you provide `ion_inertial_length_di`, it normalizes the x-axis to $(x/d_i)$.

### 2) MMS-like tetrahedron (4 probes)
A centered tetrahedron is created via fixed ±1 vertex offsets, then scaled so edge lengths are approximately `MMS_separation`:

- `base_offsets`: 4 vertices
- `scale = MMS_separation / (2\sqrt{2})`
- `offsets = base_offsets * scale`

Each probe trajectory is:

$$
\mathbf{x}_p(t) = \mathbf{x}(t) + \Delta \mathbf{x}_p,\qquad p=1..4.
$$

A fixed RNG seed (`default_rng(12345)`) makes random flybys reproducible.

---

## Tri-linear interpolation and how it’s used

The simulation fields live on a regular Cartesian grid, but probe coordinates are continuous. Sampling uses a vectorized **tri-linear interpolation** function:

```python
trilerp(arr, xyz, spacing=(DX,DY,DZ), origin=(X0,Y0,Z0))
```

### Coordinate transform
For each sample point $((x,y,z))$, convert to fractional grid coordinates:

$$
u = \frac{x - X_0}{\Delta x},\quad
v = \frac{y - Y_0}{\Delta y},\quad
w = \frac{z - Z_0}{\Delta z}.
$$

Let $(i=\lfloor u\rfloor), (j=\lfloor v\rfloor), (k=\lfloor w\rfloor)$ and fractions
$(f_x=u-i), (f_y=v-j), (f_z=w-k)$.

### Weighted sum of 8 corners
The value is a convex combination of the 8 surrounding grid nodes:

$$
f(u,v,w)=\sum_{a,b,c\in\{0,1\}}
f_{k+c,\,j+b,\,i+a}\,(1-f_x)^{1-a}f_x^a\,(1-f_y)^{1-b}f_y^b\,(1-f_z)^{1-c}f_z^c.
$$

### Boundary behavior
Indices are **clipped** to valid bounds (`np.clip`). If a probe point lies outside the domain, it will effectively **clamp** to the nearest boundary cell (rather than throwing an error). This is robust for batch runs.

### What gets interpolated
For each probe and each sample position, the script uses `trilerp` to sample:

- densities (`rho_0`, `rho_1`) and derived number densities
- `Bx, By, Bz` and derived $(|B|)$
- `Ex, Ey, Ez` (if present) and derived $(|E|)$
- derived velocities (computed on-grid, then sampled)
- any additional diagnostic fields you add later (the interpolation is generic)

Because `trilerp` is vectorized, you can pass all sample coordinates for one probe at once (fast and simple).

---

## Derived quantities computed on the grid

Before sampling, the script derives several useful plasma quantities on the full grid.

### Species velocities from current and charge density
With a small guard `eps`:

- electrons (species 1): `rho0 = q_e n_e` is negative (assuming $(q_e=-1)$)
  $$
  \mathbf{v}_e = \frac{\mathbf{J}_0}{\rho_0+\varepsilon}
  $$
- ions (species 0): `rho1 = q_i n_i` is positive (assuming $(q_i=+1)$)
  $$
  \mathbf{v}_i = \frac{\mathbf{J}_1}{\rho_1+\varepsilon}
  $$

### Number densities (code units)
Assuming $(|q|=1)$:

$$
n_e = -\rho_0,\qquad n_i = \rho_1.
$$

### Barycentric (bulk) velocity
Number-density-weighted average:

$$
\mathbf{v}_b =
\frac{n_i\mathbf{v}_i + n_e\mathbf{v}_e}{n_i + n_e + \varepsilon}.
$$

These are useful for comparing flow signatures across many flybys.

---

## Outputs

### CSVs
For each flyby, the script writes a CSV in `flyby_csvs/` containing the sampled time series (distance coordinate plus multiple fields). These CSVs are meant for quick reuse in notebooks and post-processing.

### Figures
For each flyby, the script writes figures to `flyby_figs/` showing selected diagnostics (typically for `probe_to_plot`, while still computing all probes).

The `highlight_window` option allows marking a region of interest along the x-axis.

---

## Performance notes

Cost scales like:

$$
O(n_{flybys} \times n_{probes} \times n_{samples}).
$$

If you need it faster:
- reduce `n_samples` and/or `n_flybys`
- compute fewer derived fields
- parallelize over flybys (each flyby is independent)

---

## How to run

```bash
python multiple_flybys.py
```

Recommended first run:
1. Set `path` and geometry parameters.
2. Check the printed dataset list and adjust aliases if needed.
3. Start with small `n_flybys` (e.g. 1–2) and smaller `n_samples` to validate.

