# Beta–anisotropy diagram from pressure tensor and magnetic field (`beta_aniso.py`)

This script computes the **beta–anisotropy distribution** from a 3D iPIC3D/ECSIM snapshot and plots it as a **2D log-density histogram** in the $(\beta_\parallel,\; T_\perp/T_\parallel)$ plane.

It overlays common **linear-instability threshold curves** (ion cyclotron, mirror mode, firehose; or electron whistler/firehose) and can optionally show **two growth-rate panels** side-by-side (nominally $(10^{-3})$ vs $(10^{-1})$).

---

## Dependencies

```bash
pip install numpy h5py matplotlib
```

---

## Configuration

At the top of the script:

- File: `path = "../../DoubleGEM-Fields_300000.h5"   # or any of your snapshots # T3D-Fields_005500.h5`
- `species = 1                         # 1 = ions, 0 = electrons`
  - `0` → electrons
  - `1` → ions
- `combine_species = False             # if True, use P_total = P(species) + P(other)`
  - If `True`, uses the sum of both species pressure tensors:
    $$
    \mathbf{P}_\mathrm{total} = \mathbf{P}_s + \mathbf{P}_{1-s}
    $$
- `stride = 2                          # downsample factor (adjust 2–6)` downsampling factor (reduces memory and speeds plotting)
- Guards:
  - `eps_B2` prevents division by tiny \(B^2\)
  - `eps_Ppar` prevents division by tiny \(P_\parallel\)
- `compare_growth_rates = True   # if True, show side-by-side panels: left=1e-3, right=1e-1 (for any species)`
  - If `True`, produces **two panels** with the same histogram normalization (useful for comparing threshold curves).

---

## Input data assumptions (HDF5 layout)

The script expects datasets under:

- `Step#0/Block/<name>`

It supports the case where `<name>` is a group containing datasets (prefers dataset `"0"` if present).

Required fields:

- Magnetic field components: `Bx, By, Bz`
- Pressure tensor components for the chosen species `s`:
  - `Pxx_s, Pyy_s, Pzz_s, Pxy_s, Pxz_s, Pyz_s`

---

## What the script computes

### 1) Magnetic-field unit vector \(\hat{\mathbf{b}}\)

First compute:

$$
B^2 = B_x^2 + B_y^2 + B_z^2
$$

Then the unit vector:

$$
\hat{\mathbf{b}} = \frac{\mathbf{B}}{\sqrt{\max(B^2,\varepsilon_{B^2})}}.
$$

This defines the local parallel/perpendicular directions.

---

### 2) Parallel pressure from the full pressure tensor

Given the (symmetric) pressure tensor $(\mathbf{P})$, the parallel pressure is:

$$
P_\parallel = \hat{\mathbf{b}}^T \mathbf{P}\,\hat{\mathbf{b}}.
$$

Expanded in components (including off-diagonal terms):

$$
\begin{aligned}
P_\parallel =\,& b_x^2 P_{xx} + b_y^2 P_{yy} + b_z^2 P_{zz} \\
&+ 2\,(b_x b_y P_{xy} + b_x b_z P_{xz} + b_y b_z P_{yz}).
\end{aligned}
$$

---

### 3) Perpendicular pressure

Using the trace:

$$
\mathrm{Tr}(\mathbf{P}) = P_{xx} + P_{yy} + P_{zz},
$$

the perpendicular pressure is:

$$
P_\perp = \frac{\mathrm{Tr}(\mathbf{P}) - P_\parallel}{2}.
$$

---

### 4) Numerical safety: sign correction and clipping

#### Sign correction
Some outputs store *electron* pressure components with an overall negative sign. The script checks the mean trace and if it is negative, flips the sign of **all** tensor components.

#### Clipping small negatives
Due to noise/roundoff, $(P_\parallel)$ or $(P_\perp)$ can become slightly negative. The script clips them to $(\ge 0)$.

---

### 5) Beta and anisotropy

#### Parallel beta
In the code’s normalized units (magnetic pressure $(\propto B^2/2)$):

$$
\beta_\parallel = \frac{2 P_\parallel}{\max(B^2,\varepsilon_{B^2})}.
$$

#### Temperature anisotropy proxy
The plotted anisotropy is the pressure ratio:

$$
A \equiv \frac{T_\perp}{T_\parallel} \approx \frac{P_\perp}{\max(P_\parallel,\varepsilon_{P_\parallel})}.
$$

(Using pressures is standard when density cancels between perpendicular and parallel temperatures for the same species.)

---

### 6) Masking (valid-point selection)

Before building the histogram, the script keeps only points satisfying:

- finite $(\beta_\parallel)$ and $(A)$
- $(B^2 > \varepsilon_{B^2})$
- $(P_\parallel > 0)$

It prints how many points survive and raises a clear error if none do.

---

## Plot: 2D log-density histogram

### Binning and normalization
- $(\beta_\parallel)$: log-spaced bins (`logspace`)
- $(A)$: linear bins (`linspace`), with robust percentile-based limits and a cap around $(A\le 5)$ for readability
- histogram counts use `LogNorm` so both dense and sparse regions remain visible

A reference line $(A=1)$ is drawn for isotropy.

### Two-panel mode
If `compare_growth_rates=True`, the script creates two panels with the **same** histogram normalization, so differences you see are due to the overlaid thresholds, not color rescaling.

---

## Instability threshold curves overlaid

The script overlays parametric fits of the form:

$
A(\beta_\parallel) = 1 + \frac{C}{\beta_\parallel^\alpha}
\quad\text{or}\quad
A(\beta_\parallel) = 1 - \frac{C}{\beta_\parallel^\alpha}.
$

### Ions (`species=1`)
When two-panel mode is enabled:
- left: nominal growth $(\gamma/\Omega_p = 10^{-3})$
- right: nominal growth $(\gamma/\Omega_p = 10^{-1})$

Curves shown:
- Ion cyclotron
- Mirror mode
- Firehose

### Electrons (`species=0`)
The script plots electron whistler and electron firehose fits (Lazar+2018). In two-panel mode, it repeats them for visual consistency.

---

## How to run

```bash
python beta_aniso.py
```

Recommended workflow:
1. Start with `stride` around 4–6 to validate quickly.
2. Reduce `stride` for publication-quality statistics.
3. Toggle `species` (0/1) and `combine_species` depending on your physics goal.

---

## Interpretation tips

- $(A>1$) corresponds to $(T_\perp>T_\parallel)$; $(A<1)$ corresponds to $(T_\perp<T_\parallel)$.
- Threshold curves indicate where linear theory predicts anisotropy-driven instabilities may grow for a given $(\beta_\parallel)$.
- Accumulation of points near a curve is often interpreted as “marginal stability,” but interpretation depends on the full dynamics (driving, relaxation, collisions, finite box effects, etc.).
