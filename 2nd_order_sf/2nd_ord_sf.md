# Second-order structure function analysis (`2nd_ord_sf.py`)

This script computes and visualizes the **second-order structure function** of simulation fields (commonly used in turbulence and plasma physics).

The second-order structure function is defined as:

S₂(ℓ) = ⟨ | f(x + ℓ) − f(x) |² ⟩

where:

- f is the field of interest (density, magnetic field component, etc.)
- ℓ is the spatial separation (lag)
- ⟨ ⟩ indicates spatial averaging

---

## Dependencies

```bash
pip install numpy matplotlib h5py
```

---

## What the script does

1. Loads a field from file (usually HDF5 output)
2. Chooses maximum spatial lag
3. Computes field increments for each lag
4. Squares and averages differences
5. Plots S₂ vs separation

---

## Core algorithm

For each separation ℓ:

```python
delta = field_shifted - field_original
S2[l] = np.mean(delta**2)
```

This gives the mean squared increment at scale ℓ.

---

## Physical interpretation

Power-law scaling:

S₂(ℓ) ∝ ℓ^α

Typical exponents:

- Kolmogorov turbulence → α ≈ 2/3
- MHD turbulence → α ≈ 0.5–0.7
- Dissipation range → steeper

---

## Assumptions

- Uniform grid
- Often periodic boundaries
- Isotropic averaging unless modified

---

## Performance

Brute-force method scales poorly for large 3D arrays.

Future improvements:

- FFT-based structure functions
- Numba acceleration
- Directional structure functions
- Save outputs
- Add power-law fitting
- Support vector magnitudes



