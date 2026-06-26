# Kasus 4 - Distribusi medan magnet aktuator VCM (magnetostatik silinder, nabla^2 A = -mu J)

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
from numba import njit

# Parameter fisik
mu0 = 4 * np.pi * 1e-7
Rmax = 15e-3
Zmax = 20e-3
Nr = 100
Nz = 120
dr = Rmax / (Nr - 1)
dz = Zmax / (Nz - 1)
r = np.linspace(0, Rmax, Nr)
z = np.linspace(0, Zmax, Nz)

# Geometri kumparan
r1 = 1.5e-3
r2 = 2.0e-3
coil_h = 3e-3
Icoil = 1.0
J0 = Icoil / ((r2 - r1) * coil_h)


# Perakitan matriks FDFD (Numba). Stensil pers. 90 termasuk suku -Aphi/r^2.
@njit
def rakit(Nr, Nz, r, dr, dz, J, mu0):
    N = Nr * Nz
    rows = np.empty(N * 5, np.int64)
    cols = np.empty(N * 5, np.int64)
    vals = np.empty(N * 5)
    rhs = np.zeros(N)
    c = 0
    for i in range(Nr):
        for j in range(Nz):
            p = i * Nz + j

            # batas luar (Dirichlet A = 0)
            if i == Nr - 1 or j == 0 or j == Nz - 1:
                rows[c] = p
                cols[c] = p
                vals[c] = 1.0
                c += 1
                continue

            # sumbu r = 0 (regularitas dA/dr = 0)
            if i == 0:
                rows[c] = p
                cols[c] = p
                vals[c] = 1.0
                c += 1
                rows[c] = p
                cols[c] = p + Nz
                vals[c] = -1.0
                c += 1
                continue

            rp = r[i]
            ae = 1 / dr**2 + 1 / (2 * rp * dr)
            aw = 1 / dr**2 - 1 / (2 * rp * dr)
            an = 1 / dz**2
            ass = 1 / dz**2
            ap = -(ae + aw + an + ass) - 1 / rp**2      # KOREKSI: suku -Aphi/r^2

            rows[c] = p
            cols[c] = (i + 1) * Nz + j
            vals[c] = ae
            c += 1
            rows[c] = p
            cols[c] = (i - 1) * Nz + j
            vals[c] = aw
            c += 1
            rows[c] = p
            cols[c] = p + 1
            vals[c] = an
            c += 1
            rows[c] = p
            cols[c] = p - 1
            vals[c] = ass
            c += 1
            rows[c] = p
            cols[c] = p
            vals[c] = ap
            c += 1
            rhs[p] = -mu0 * J[i, j]

    return rows[:c], cols[:c], vals[:c], rhs


# Selesaikan medan untuk satu posisi koil (pusat zc)
def solve(zc):
    J = np.zeros((Nr, Nz))
    za = zc - coil_h / 2
    zb = zc + coil_h / 2
    mask = (r[:, None] >= r1) & (r[:, None] <= r2) & (z[None, :] >= za) & (z[None, :] <= zb)
    J[mask] = J0

    rows, cols, vals, rhs = rakit(Nr, Nz, r, dr, dz, J, mu0)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(Nr * Nz, Nr * Nz))
    Aphi = spla.spsolve(A, rhs).reshape((Nr, Nz))

    Br = np.zeros_like(Aphi)
    Bz = np.zeros_like(Aphi)
    Br[:, 1:-1] = -(Aphi[:, 2:] - Aphi[:, :-2]) / (2 * dz)
    for i in range(1, Nr - 1):
        # KOREKSI Bz = (1/r) d(r Aphi)/dr
        Bz[i, :] = (r[i + 1] * Aphi[i + 1, :] - r[i - 1] * Aphi[i - 1, :]) / (2 * dr * r[i])
    return np.sqrt(Br**2 + Bz**2), Br, Bz


# Sapuan posisi koil sepanjang sumbu (gerak autofokus)
zs = np.linspace(7e-3, 13e-3, 30)
FB = []
FR = []
FZ = []
print("Menyelesaikan medan magnet untuk tiap posisi koil ...")
for zc in zs:
    b, br, bz = solve(zc)
    FB.append(b)
    FR.append(br)
    FZ.append(bz)
print("Selesai.")

Bmx = max(b.max() for b in FB)
ic = np.argmin(np.abs(r - 2e-3))
ext = [0, Rmax * 1e3, 0, Zmax * 1e3]
Rg, Zg = np.meshgrid(r * 1e3, z * 1e3, indexing="ij")


# Vektor satuan supaya panah pendek dan hanya menunjukkan arah
def arah(U, V):
    M = np.hypot(U, V) + 1e-30
    return U / M, V / M


# Dashboard animasi: |B| + arah medan + profil sumbu
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.6))
fig.suptitle("DISTRIBUSI MEDAN MAGNET AKTUATOR VCM (STROKE AUTOFOKUS)",
             fontsize=12, fontweight="bold", color="darkgreen")

im1 = ax1.imshow(FB[0].T, origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=Bmx, aspect="auto")
ax1.set_title("Distribusi Medan Magnet VCM")
ax1.set_xlabel("r (mm)")
ax1.set_ylabel("z (mm)")
fig.colorbar(im1, ax=ax1, label="|B| (Tesla)")

im2 = ax2.imshow(FB[0].T, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=Bmx, aspect="auto")
u0, v0 = arah(FR[0][::5, ::5], FZ[0][::5, ::5])
qv = ax2.quiver(Rg[::5, ::5], Zg[::5, ::5], u0, v0, color="white", scale=30, width=0.004)
ax2.set_title("Arah Medan Magnet")
ax2.set_xlabel("r (mm)")
ax2.set_ylabel("z (mm)")

ln, = ax3.plot(z * 1e3, FB[0][ic, :], lw=2)
ax3.set_ylim(0, Bmx * 1.05)
ax3.set_title("Profil |B| Sepanjang Sumbu Aktuator")
ax3.set_xlabel("z (mm)")
ax3.set_ylabel("|B| (Tesla)")
ax3.grid(True)


def update(f):
    im1.set_data(FB[f].T)
    im2.set_data(FB[f].T)
    qv.set_UVC(*arah(FR[f][::5, ::5], FZ[f][::5, ::5]))
    ln.set_ydata(FB[f][ic, :])
    ax1.set_title("Distribusi Medan Magnet VCM  (koil z = %.1f mm)" % (zs[f] * 1e3))
    return [im1, im2, qv, ln]


ani = FuncAnimation(fig, update, frames=len(FB), interval=120, blit=False)
plt.tight_layout()
plt.close()

# Plot statis (posisi koil di tengah)
fmid = len(FB) // 2
figs, (b1, b2, b3) = plt.subplots(1, 3, figsize=(15, 4.6))
figs.suptitle("DISTRIBUSI MEDAN MAGNET VCM (koil z = %.1f mm)" % (zs[fmid] * 1e3), fontweight="bold")

s1 = b1.imshow(FB[fmid].T, origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=Bmx, aspect="auto")
b1.set_title("Distribusi Medan Magnet VCM")
b1.set_xlabel("r (mm)")
b1.set_ylabel("z (mm)")
figs.colorbar(s1, ax=b1, label="|B| (Tesla)")

b2.imshow(FB[fmid].T, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=Bmx, aspect="auto")
us, vs = arah(FR[fmid][::5, ::5], FZ[fmid][::5, ::5])
b2.quiver(Rg[::5, ::5], Zg[::5, ::5], us, vs, color="white", scale=30, width=0.004)
b2.set_title("Arah Medan Magnet")
b2.set_xlabel("r (mm)")
b2.set_ylabel("z (mm)")

b3.plot(z * 1e3, FB[fmid][ic, :], lw=2)
b3.set_title("Profil |B| Sepanjang Sumbu Aktuator")
b3.set_xlabel("z (mm)")
b3.set_ylabel("|B| (Tesla)")
b3.grid(True)

figs.tight_layout()

# Output: animasi video + plot statis
# display(HTML(ani.to_html5_video()))
figs.savefig('../figures/kasus4.png', dpi=200, bbox_inches='tight')
# plt.show()
