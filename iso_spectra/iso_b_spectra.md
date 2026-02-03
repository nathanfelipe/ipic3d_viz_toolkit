# 3D magnetic-field spectra (`3d_b_spectra.py`)

This script computes **magnetic-field power spectra** from a single 3D iPIC3D/ECSIM HDF5 snapshot:

- an **isotropic 1D spectrum** $(E_B(k))$ obtained by shell-averaging in $(|\mathbf{k}|)$
- **reduced 1D spectra** along the coordinate axes: $(E(k_x))$, $(E(k_y))$, $(E(k_z))$
- optionally, a **2D spectrum** $(E(k_\perp, k_\parallel))$ relative to the mean-field direction

It’s designed mainly for **slope comparisons** (e.g. $(k^{-5/3})$, $(k^{-3/2})$) and quick anisotropy checks—exact absolute normalization depends on conventions and windowing.

---

## Dependencies

```bash
pip install numpy h5py matplotlib
```

---

## Inputs and HDF5 layout

The script expects (for a single time step):

- `Step#0/Block/Bx`
- `Step#0/Block/By`
- `Step#0/Block/Bz`

If one of these is an HDF5 **group**, it uses dataset `"0"` if present (otherwise the first dataset in the group).

---

## Grid ordering and why `nz, ny, nx` makes sense

The code assumes that the arrays are stored in memory as:

- `Bx.shape == (nz, ny, nx)` meaning **(z, y, x)** ordering.

This is consistent with the comment in the script (“C-order: (z,y,x)”) and with many iPIC3D/ECSIM field outputs.

Why this is important:

- The FFT is executed over axes `(0,1,2)`, so axis 2 is treated as the **x** direction.
- The wavenumber arrays are built as:
  - `kx = rfftfreq(nx, d=dx)`
  - `ky = fftfreq(ny, d=dy)`
  - `kz = fftfreq(nz, d=dz)`

So the mapping **only makes sense** if axis 2 corresponds to x, axis 1 to y, axis 0 to z.

### Quick sanity checks you can do
- If your system is physically anisotropic (e.g. current sheet oriented along y), you should see that reflected in `E(ky)` vs `E(kx)` vs `E(kz)`—your attached plot shows distinct reduced spectra, consistent with this assumption.
- If you ever suspect the array order is `(x,y,z)`, swapping axes before the FFT would be required.

---

## What the script computes

### 1) Downsampling (`stride`)
It loads:

```python
Bx = Bx_full[::stride, ::stride, ::stride]
```

This reduces memory and speeds FFTs. The physical spacing is updated:

$$
dx = DX\,\text{stride},\quad dy = DY\,\text{stride},\quad dz = DZ\,\text{stride}.
$$

### 2) Mean removal + Hann window
Each component is preprocessed as:

1. Remove global mean:
   $$
   B_i \leftarrow B_i - \langle B_i \rangle
   $$
2. Apply separable Hann window:
   $$
   W(z,y,x) = w_z(z)\,w_y(y)\,w_x(x)
   $$
   $$
   B_i^{(w)} = B_i\,W
   $$

This reduces spectral leakage due to non-periodic boundaries.

### 3) Orthonormal FFT
The script uses an orthonormal real FFT:

```python
Fb_x = rfftn(Bxw, norm="ortho")
```

With `norm="ortho"`, Parseval holds for the **windowed** data:

$$
\sum |B|^2 \approx \sum |\hat B|^2
$$

### 4) Total magnetic power per mode
It forms the 3D power per Fourier mode:

$$
P_{3D}(\mathbf{k}) = |\hat B_x|^2 + |\hat B_y|^2 + |\hat B_z|^2.
$$

Then it compensates for Hann window RMS loss via:

$$
W_2 = \langle W^2 \rangle,
\qquad
P_{3D} \leftarrow P_{3D}/W_2.
$$

### 5) Physical wavenumbers (rad / unit length)
Wavenumbers are built as:

$$
k_x = 2\pi\,\mathrm{rfftfreq}(n_x, d=dx),\quad
k_y = 2\pi\,\mathrm{fftfreq}(n_y, d=dy),\quad
k_z = 2\pi\,\mathrm{fftfreq}(n_z, d=dz).
$$

Then $(|\mathbf{k}|)$ is computed from the meshgrid.

The **k-space cell volume** is:

$$
\Delta k_x\Delta k_y\Delta k_z =
\left(\frac{2\pi}{L_x}\right)
\left(\frac{2\pi}{L_y}\right)
\left(\frac{2\pi}{L_z}\right),
\quad
L_x = n_x dx,\;\text{etc.}
$$

---

## Isotropic spectrum $(E_B(k))$

Modes are shell-binned by $(|\mathbf{k}|)$:

1. Choose log-spaced bins `kbins`
2. Sum per shell:
   $$
   S(k) = \sum_{\mathbf{k}\in\text{shell}} P_{3D}(\mathbf{k})
   $$
3. Convert to an isotropic **spectral density** by dividing by shell volume:
   $$
   E_B(k) \approx
   \frac{S(k)\,\Delta k_x\Delta k_y\Delta k_z}
        {4\pi k^2\,\Delta k}.
   $$

This produces a curve suitable for inertial-range slope checks.

---

## Reduced spectra $(E(k_x), E(k_y), E(k_z))$

Reduced spectra are computed by summing $(P_{3D})$ over planes:

- $(E(k_x) \propto \sum_{k_y,k_z} P_{3D}\,\Delta k_y\Delta k_z)$
- $(E(k_y) \propto \sum_{k_x,k_z} P_{3D}\,\Delta k_x\Delta k_z)$
- $(E(k_z) \propto \sum_{k_x,k_y} P_{3D}\,\Delta k_x\Delta k_y)$

The script folds `ky` and `kz` to positive values to avoid double-counting in a one-sided plot.

---

## Optional $(E(k_\perp, k_\parallel))$

If `compute_kperp_kpar=True`, it estimates a mean-field direction $(\hat b)$ and projects:

$$
k_\parallel = \mathbf{k}\cdot\hat b,\quad
k_\perp = \sqrt{|\mathbf{k}|^2 - k_\parallel^2}.
$$

Then it bins a 2D histogram in $((k_\perp, |k_\parallel|))$.

---

## How to run

```bash
python 3d_b_spectra.py
```

Outputs are displayed interactively with Matplotlib:

- left: isotropic $(E_B(k))$ + slope guides
- right: reduced $(E(k_x))$, $(E(k_y))$, $(E(k_z))$
- optional: 2D $(E(k_\perp, k_\parallel))$

---

## Notes for your stride=2 run (attached figure)

Your plot shows:

- a smooth, monotonic isotropic spectrum with a dissipative roll-off at high $(k)$
- reduced spectra that differ between axes (expected in structured reconnection/turbulence setups)

These are qualitatively consistent with the assumed `(z,y,x)` array ordering and the updated spacings $(dx=DX\,\text{stride})$, etc.

