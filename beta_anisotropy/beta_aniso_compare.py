import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ---------------- CONFIG ----------------

path = "../../DoubleGEM-Fields_300000.h5"
stride = 2
order = "C"

# Choose which pair to compare:
# pair = (0,1)      # rho_0 vs rho_1 (background electrons vs ions)
pair = (2,3)    # rho_2 vs rho_3 (drifting / trapped populations)

eps_B2   = 1e-10
eps_Ppar = 1e-12

# ---------------------------------------


def open_var(f, name):
    obj = f[f"Step#0/Block/{name}"]
    if isinstance(obj, h5py.Group):
        if "0" in obj:
            return obj["0"]
        for k in obj.keys():
            if isinstance(obj[k], h5py.Dataset):
                return obj[k]
        raise KeyError(name)
    return obj


def load_pressure(f, sp):

    suf = f"_{sp}"

    Pxx = open_var(f, "Pxx"+suf)[::stride,::stride,::stride].astype(np.float32)
    Pyy = open_var(f, "Pyy"+suf)[::stride,::stride,::stride].astype(np.float32)
    Pzz = open_var(f, "Pzz"+suf)[::stride,::stride,::stride].astype(np.float32)
    Pxy = open_var(f, "Pxy"+suf)[::stride,::stride,::stride].astype(np.float32)
    Pxz = open_var(f, "Pxz"+suf)[::stride,::stride,::stride].astype(np.float32)
    Pyz = open_var(f, "Pyz"+suf)[::stride,::stride,::stride].astype(np.float32)

    trace_mean = np.nanmean(Pxx + Pyy + Pzz)

    if trace_mean < 0:
        print(f"[note] species {sp}: flipping pressure sign")
        Pxx *= -1; Pyy *= -1; Pzz *= -1
        Pxy *= -1; Pxz *= -1; Pyz *= -1

    return Pxx,Pyy,Pzz,Pxy,Pxz,Pyz


def compute_beta_aniso(Bx,By,Bz,P):

    Pxx,Pyy,Pzz,Pxy,Pxz,Pyz = P

    B2 = Bx*Bx + By*By + Bz*Bz

    invB = 1.0 / np.sqrt(np.maximum(B2, eps_B2))

    bx = Bx*invB
    by = By*invB
    bz = Bz*invB

    # P_parallel = b^T P b

    Ppar = (
        bx*bx*Pxx + by*by*Pyy + bz*bz*Pzz
        + 2*(bx*by*Pxy + bx*bz*Pxz + by*bz*Pyz)
    )

    traceP = Pxx + Pyy + Pzz
    Pperp = 0.5*(traceP - Ppar)

    Ppar  = np.maximum(Ppar,0)
    Pperp = np.maximum(Pperp,0)

    beta_par = 2*Ppar / np.maximum(B2, eps_B2)
    A = Pperp / np.maximum(Ppar, eps_Ppar)

    return B2, beta_par, A


def stats(name, arr):
    arr = arr[np.isfinite(arr)]
    if arr.size:
        print(f"{name}: min={arr.min():.3e}, max={arr.max():.3e}, mean={arr.mean():.3e}")


# ---------------- LOAD DATA ----------------

with h5py.File(path,"r") as f:

    Bx = open_var(f,"Bx")[::stride,::stride,::stride].astype(np.float32)
    By = open_var(f,"By")[::stride,::stride,::stride].astype(np.float32)
    Bz = open_var(f,"Bz")[::stride,::stride,::stride].astype(np.float32)

    P1 = load_pressure(f, pair[0])
    P2 = load_pressure(f, pair[1])


# ---------------- COMPUTE ----------------

B2_1, beta1, A1 = compute_beta_aniso(Bx,By,Bz,P1)
B2_2, beta2, A2 = compute_beta_aniso(Bx,By,Bz,P2)

stats("beta_1",beta1)
stats("A_1",A1)
stats("beta_2",beta2)
stats("A_2",A2)


mask1 = np.isfinite(beta1)&np.isfinite(A1)&(beta1>0)&(A1>0)&(B2_1>eps_B2)
mask2 = np.isfinite(beta2)&np.isfinite(A2)&(beta2>0)&(A2>0)&(B2_2>eps_B2)

x1 = beta1[mask1].ravel(order=order)
y1 = A1[mask1].ravel(order=order)

x2 = beta2[mask2].ravel(order=order)
y2 = A2[mask2].ravel(order=order)

print(f"kept species {pair[0]}: {x1.size}")
print(f"kept species {pair[1]}: {x2.size}")


# ---------------- HISTOGRAM BINS ----------------

x_all = np.concatenate([x1,x2])
y_all = np.concatenate([y1,y2])

xmin = max(1e-4, x_all.min())
xmax = x_all.max()

xbins = np.logspace(np.log10(xmin), np.log10(xmax), 80)

ymin = max(0.1, np.nanpercentile(y_all,0.1))
ymax = min(7.0, np.nanpercentile(y_all,99.9))

ybins = np.linspace(ymin,ymax,80)

H1,_,_ = np.histogram2d(x1,y1,bins=[xbins,ybins])
H2,_,_ = np.histogram2d(x2,y2,bins=[xbins,ybins])

H = H1 + H2

vmin = max(1, H[H>0].min())
vmax = H.max()

norm = LogNorm(vmin=vmin, vmax=vmax)


# ---------------- PLOT ----------------

fig,(axL,axR) = plt.subplots(1,2,figsize=(14,6),sharex=True,sharey=True,constrained_layout=True)


# ---- LEFT PANEL ----

hL = axL.hist2d(x1,y1,bins=[xbins,ybins],norm=norm,cmap="jet")

axL.set_xscale("log")
axL.set_xlabel(r"$\beta_\parallel$")
axL.set_ylabel(r"$T_\perp/T_\parallel$")
axL.axhline(1,color="w",lw=1,alpha=0.6)
axL.grid(alpha=0.3)

axL.set_title(f"species {pair[0]}")


# ---- RIGHT PANEL ----

hR = axR.hist2d(x2,y2,bins=[xbins,ybins],norm=norm,cmap="jet")

axR.set_xscale("log")
axR.set_xlabel(r"$\beta_\parallel$")
axR.axhline(1,color="w",lw=1,alpha=0.6)
axR.grid(alpha=0.3)

axR.set_title(f"species {pair[1]}")


# ---------------- THRESHOLDS ----------------

beta_vals = np.logspace(-2,2.5,300)

# Ion fits (Hellinger-like, gamma=1e-3)

A_ic_3 = 1 + 0.43 / beta_vals**0.42
A_mm_3 = 1 + 0.77 / beta_vals**0.76
A_fh_3 = 1 - 1.40 / beta_vals**0.47

# Electron fits (Lazar)

A_ew  = 1 + 0.25 / beta_vals**0.5
A_efh = 1 - 1.29 / beta_vals**0.98


def overlay(ax, species):

    # electrons assumed QOM negative → even species (0,2)
    is_electron = (species % 2 == 0)

    if is_electron:
        ax.plot(beta_vals, A_ew,  "r--", lw=1.6, label="Whistler")
        ax.plot(beta_vals, A_efh, "b:",  lw=1.6, label="Firehose")

    else:
        ax.plot(beta_vals, A_ic_3, "r--", lw=1.6, label="Ion cyclotron")
        ax.plot(beta_vals, A_mm_3, "m-.", lw=1.6, label="Mirror mode")
        ax.plot(beta_vals, A_fh_3, "b:",  lw=1.6, label="Firehose")

    ax.legend(frameon=False)


overlay(axL, pair[0])
overlay(axR, pair[1])


plt.colorbar(hL[3], ax=axL, label="counts (log)")
plt.colorbar(hR[3], ax=axR, label="counts (log)")

plt.show()