# Live flyby with slice + glyphs + synthetic probe diagnostics (`live_flyby_slice_glyphs.py`)

This script builds a **“synthetic spacecraft flyby”** through an iPIC3D/ECSIM 3D field snapshot and generates a **frame-by-frame animation** that combines:

1) a **PyVista/VTK 3D view** (volume + a moving slice plane, optional glyphs/contours, probe sphere + trail), and  
2) a **Matplotlib time-series panel** showing the sampled quantities along the flyby path.

It can also compute **local isotropic energy spectra** (FFT-based, binned by \(|k|\)) inside a moving spherical probe volume and write **one final spectrum plot per energy type**.

---

## What you get

### Main output (animation frames)
- `frames/frame_000000.png`, `frames/frame_000001.png`, …  
  Each PNG is a **side-by-side** composite:
  - left: PyVista render (3D view)
  - right: Matplotlib time series (revealed up to current frame)

At the end, the script prints an `ffmpeg` command you can run to assemble the video.

### Optional output (energy spectra)
If enabled, the script writes:
- `kinetic_energy_spectrum.png`
- `magnetic_energy_spectrum.png`

---

## Dependencies

```bash
pip install numpy h5py pyvista vtk matplotlib
```

Notes:
- `matplotlib.use("Agg")` is set so frames can be rendered headlessly (e.g., on clusters).
- `pyvista` uses VTK/OpenGL for the 3D render; you’ll typically want an environment where offscreen rendering works (or a display).

---

## Inputs and configuration

The configuration is at the top of the file. The key parameters are:

### File and variable names
- HDF5 file: `path = "../../DoubleGEM-Fields_300000.h5" # "../T3D-Fields_005500.h5"`
- Scalar volume field: `scalar_name = "rho_0"` (default: density-like)
- Vector field components: `Bx`, `By`, `Bz` (used for slice sampling and optional glyphs/contours)

### Grid geometry
You must provide (from `SimulationData.txt`):
- spacing: `(DX, DY, DZ)`
- origin: `(X0, Y0, Z0)`

### Strides
To reduce load/rendering cost:
- `s_stride = 1` for the scalar volume grid
- `v_stride = 4` for the vector grid (B)

These strides directly affect spatial resolution and speed.

### Probe path
The flyby is a straight line between endpoints:
- `p0`: start point (simulation coordinates)
- `p1`: end point
- `n_samples`: number of sampled points along the line

Two safety controls:
- `stop_at_domain_exit`: truncates the path when it leaves the domain (prevents “clamped” samples)
- `print_domain_info`: prints coordinate bounds if errors occur

### Frame generation
- `out_dir = "frames"`
- `clear_existing_frames`: delete old `frame_*.png`
- `fps`: video frame rate hint
- `frame_stride`: render every Nth sample (controls speed/length)
- `max_frames`: hard cap if you want quick tests

### Slice plane and camera
- The slice plane **moves with the probe** and is oriented using the probe tangent plus an “up reference”.
- `follow_plane_camera=True` keeps the camera roughly face-on to the slice plane.

### Slice overlays
- `show_B_contours`: draw contour lines of \(|B|\) on the slice plane
- `n_B_contours`: number of contour levels
- `show_glyphs`: render arrow glyphs on the slice (off by default)
- `glyph_max_points`: cap glyph count (random sampling)
- `glyph_scale`: arrow size factor

### Energy spectra (optional)
- `compute_energy_spectra`: master switch
- `spectrum_box_n`: cube edge length in grid points for the FFT box
- `average_spectrum_over_path`: average spectra across selected probe positions
- `use_spectrum_frame_ids`: use only the rendered frames’ indices (sparser, faster)

---

## Data model assumptions (HDF5 layout)

Like your other scripts, this one expects datasets under:

- `Step#0/Block/<name>`

It also supports the case where `<name>` is an HDF5 **group** containing datasets (common in some outputs). In that case it selects:
- dataset `"0"` if it exists, otherwise
- the first dataset found in the group.

---

## Core ideas and calculations

## 1) Synthetic probe sampling along the path

The probe samples scalar and vector components along a continuous path in physical coordinates. Because the data are stored on a regular grid, sampling uses **tri-linear interpolation** (`trilerp`).

Given a point $(\mathbf{x}=(x,y,z)),$ define grid coordinates:

$$
u = \frac{x - X_0}{\Delta x},\quad
v = \frac{y - Y_0}{\Delta y},\quad
w = \frac{z - Z_0}{\Delta z}.
$$

Let $(i=\lfloor u\rfloor ), (j=\lfloor v\rfloor ), (k=\lfloor w\rfloor )$ and fractions $(f_x=u-i), (f_y=v-j), (f_z=w-k).$  
The interpolated value is the weighted sum of the 8 surrounding grid nodes:

$$
f(u,v,w)=\sum_{a\in\{0,1\}}\sum_{b\in\{0,1\}}\sum_{c\in\{0,1\}}
f_{(i+a,\,j+b,\,k+c)}\,(1-f_x)^{1-a}f_x^{a}\,(1-f_y)^{1-b}f_y^{b}\,(1-f_z)^{1-c}f_z^{c}.
$$

The script applies this to:
- `rho` (the scalar field)
- `Bx, By, Bz` (vector components)

Then it computes:

$$
|B| = \sqrt{B_x^2 + B_y^2 + B_z^2}.
$$

These values are plotted in the Matplotlib time-series panel as the flyby advances.

### Domain exit handling
To avoid silently sampling invalid regions, the script can truncate the path when the probe leaves the domain (`stop_at_domain_exit=True`).

---

## 2) Building the moving slice plane

By default the script constructs a plane that:
- passes through the current probe position \(\mathbf{x}_p\)
- contains the probe tangent direction \(\mathbf{t}\)
- contains a user-specified “up reference” direction \(\mathbf{u}\) (as long as it is not nearly parallel to \(\mathbf{t}\))

The plane **normal** is:

$$
\mathbf{n} = \frac{\mathbf{t} \times \mathbf{u}}{\|\mathbf{t} \times \mathbf{u}\|}.
$$

Optional roll (rotation of the plane around the tangent) is applied using Rodrigues’ rotation formula.

### Fixed plane mode
If `use_fixed_plane=True`, the plane origin/normal are taken from the fixed values and do not follow the probe.

---

## 3) Slice extraction and B sampling onto the slice

For each frame, the script:
1. Slices the scalar volume grid:
   - `sl = grid.slice(normal=n, origin=xp)`
2. Samples the vector grid onto the slice points:
   - `sl_v = sl.sample(grid_v)`

This ensures \(\mathbf{B}\) values are available **on the slice geometry**, even though \(\mathbf{B}\) is stored on a potentially coarser grid (`v_stride`).

---

## 4) Contours of \(|B|\) on the slice (optional)

If `show_B_contours=True` and `Bmag` is available on the slice, the script computes contour lines:

- `contours = sl_v.contour(isosurfaces=n_B_contours, scalars="Bmag")`

These are drawn as line geometry on top of the slice.

---

## 5) Glyph arrows on the slice (optional)

If `show_glyphs=True`, the script renders arrow glyphs oriented by \(\mathbf{B}\) on the slice.

Key design choices:

- **Association safety:** VTK can be picky about whether arrays are point- or cell-associated. The script builds a fresh `PolyData` containing only the chosen points and a point-associated vector array `Bvec`.
- **Random sampling:** rather than striding indices (which can cluster glyphs), it randomly selects up to `glyph_max_points` points uniformly across the slice.
- **Fixed arrow size:** glyphs use `scale=False` and a global `factor=glyph_scale` so arrows represent **direction** more than magnitude (you can change this later if you want magnitude scaling).

---

## 6) The “probe” geometry in the 3D view

The 3D panel includes:
- a sphere centered at the probe position (`probe_radius`)
- a line trail (optionally tubed with `use_tube_trail=True` for nicer visuals)

The trail updates each frame to show where the probe has been.

---

## 7) Frame compositing

Each frame is created by:
1. rendering PyVista to an RGB image (`pl.screenshot(return_img=True)`)
2. rendering the Matplotlib figure canvas to RGB (`mpl_fig_to_rgb`)
3. padding and horizontally concatenating the images (`hstack_images`)
4. saving a PNG using `plt.imsave`

This avoids adding extra dependencies for image I/O.

---

## Local isotropic energy spectra (optional)

If `compute_energy_spectra=True`, the script computes isotropic spectra inside a moving local volume around the probe.

### 1) Local extraction box
At selected probe indices, it extracts a cube of size:

- `spectrum_box_n × spectrum_box_n × spectrum_box_n`

around the probe location, on:
- the scalar grid for density/moments (if available)
- the B grid for magnetic fields

Near boundaries it **zero-pads** as needed (instead of wrapping periodically).

### 2) Spherical probe mask
Within the cube, it applies a spherical mask of radius `probe_radius` (and an “effective” radius on the B grid to avoid tiny masks that collapse to ~1 voxel). This defines the local measurement volume.

### 3) DC removal
Before FFT, it subtracts the mean (DC component) inside the masked region to reduce the zero-frequency spike.

### 4) FFT and isotropic binning
It computes a 3D FFT, forms $(|\hat{f}(\mathbf{k})|^2)$, and bins it by $(|k|)$ into a 1D spectrum:

$$
S(|k|) = \langle |\hat{f}(\mathbf{k})|^2 \rangle_{|\mathbf{k}|\in\text{bin}}.
$$

The script returns arrays `(k, S(k))` where:
- `k` is the radial wavenumber (in rad / unit length)
- `S(k)` is the binned power

### Kinetic energy proxy
If the required moments exist, it uses a **proxy** based on moments:

- velocity proxy: $(\mathbf{v} = \mathbf{J}/\rho)$
- kinetic energy density proxy (mass factor omitted):  
  $(\varepsilon_K = 0.5\, N \, |\mathbf{v}|^2)$

### Magnetic energy density
Uses:
$(\varepsilon_B = 0.5\,|\mathbf{B}|^2), where (\mu_0)$ is omitted.

### Averaging over the flyby
If `average_spectrum_over_path=True`, it averages the spectra over multiple probe locations (either every rendered frame index or every probe sample, depending on `use_spectrum_frame_ids`).

---

## How to run

From the directory containing the script:

```bash
python live_flyby_slice_glyphs.py
```

Then assemble the movie using the printed `ffmpeg` command (run it inside `out_dir`, or adapt the input path as needed).

---

## Practical tips

- Start with **coarser** settings: larger `frame_stride`, smaller `max_frames`, bigger `v_stride`.
- Turn on heavy visuals incrementally:
  - contours are usually cheaper than glyphs
  - tubed trails can be expensive (`use_tube_trail=False` if slow)
- If you want magnitude-scaled arrows, modify the glyph call to scale by \(|B|\) (and apply robust normalization as in your `3d_interactive.py`).

---

## Notes on conventions

- Array ordering is assumed to be compatible with your other tools (often HDF5 arrays shaped as `[z, y, x]`), with PyVista grid dimensions set as `(nx, ny, nz)` accordingly.
- All coordinates are in **simulation units** consistent with `(DX, DY, DZ)` and `(X0, Y0, Z0)`.
