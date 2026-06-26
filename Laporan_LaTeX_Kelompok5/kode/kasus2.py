# Kasus 2 - Elektrostatik koaksial barrel-housing (Laplace, FDFD), dashboard 3 panel
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from IPython.display import HTML, display
from numba import njit

a, b = 4e-3, 10e-3          # jari-jari barrel dan rumah
Lz, V0 = 30e-3, 100.0
Nr, Nz = 70, 150
r = np.linspace(0, b, Nr)
z = np.linspace(0, Lz, Nz)
dr, dz = r[1] - r[0], z[1] - z[0]
half = 7e-3                 # setengah panjang barrel


# Perakitan matriks Laplace silinder (Numba). Indeks node p = i*Nz + j
@njit
def rakit(zc, r, z, dr, dz, a, V0):
    Nr = r.size
    Nz = z.size
    N = Nr * Nz
    R = np.empty(N * 5, np.int64)
    C = np.empty(N * 5, np.int64)
    Vv = np.empty(N * 5, np.float64)
    rhs = np.zeros(N)
    c = 0

    for i in range(Nr):
        for j in range(Nz):
            p = i * Nz + j
            barrel = (r[i] <= a) and (z[j] >= zc - half) and (z[j] <= zc + half)

            # Dirichlet: barrel (V0), rumah dan ujung (ground)
            if barrel or i == Nr - 1 or j == 0 or j == Nz - 1:
                R[c] = p; C[c] = p; Vv[c] = 1.0; c += 1
                rhs[p] = V0 if barrel else 0.0
                continue

            # sumbu r = 0, regularitas dV/dr = 0
            if i == 0:
                R[c] = p; C[c] = p; Vv[c] = 1.0; c += 1
                R[c] = p; C[c] = p + Nz; Vv[c] = -1.0; c += 1
                continue

            crp = 1 / dr**2 + 1 / (2 * r[i] * dr)
            crm = 1 / dr**2 - 1 / (2 * r[i] * dr)
            czz = 1 / dz**2
            R[c] = p; C[c] = p;      Vv[c] = -(2 / dr**2 + 2 / dz**2); c += 1
            R[c] = p; C[c] = p + Nz; Vv[c] = crp; c += 1
            R[c] = p; C[c] = p - Nz; Vv[c] = crm; c += 1
            R[c] = p; C[c] = p + 1;  Vv[c] = czz; c += 1
            R[c] = p; C[c] = p - 1;  Vv[c] = czz; c += 1

    return R[:c], C[:c], Vv[:c], rhs


# Selesaikan potensial dan medan untuk satu posisi barrel
def solve(zc):
    R, C, Vv, rhs = rakit(zc, r, z, dr, dz, a, V0)
    M = sp.csr_matrix((Vv, (R, C)), shape=(Nr * Nz, Nr * Nz))
    V = spla.spsolve(M, rhs).reshape(Nr, Nz)
    Er = np.zeros_like(V)
    Ez = np.zeros_like(V)
    Er[1:-1, :] = -(V[2:, :] - V[:-2, :]) / (2 * dr)
    Ez[:, 1:-1] = -(V[:, 2:] - V[:, :-2]) / (2 * dz)
    return V, Er, Ez


# Validasi koaksial pada bidang tengah
Vmid, _, _ = solve(Lz / 2)
mask = (r >= a) & (r <= b)
Vana = V0 * np.log(b / r[mask]) / np.log(b / a)
err = np.max(np.abs(Vmid[mask, Nz // 2] - Vana)) / V0 * 100
print("error validasi koaksial =", round(float(err), 2), "%")

# Precompute medan tiap posisi barrel (durasi animasi sekitar 5 detik)
zcs = np.linspace(0.30 * Lz, 0.70 * Lz, 50)
hasil = [solve(zc) for zc in zcs]

# Subgrid panah arah medan
zi = np.arange(0, Nz, 8)
ri = np.arange(2, Nr, 6)
fmid = len(hasil) // 2
Ecap = np.percentile(np.hypot(hasil[fmid][1], hasil[fmid][2]), 98)


# Vektor satuan supaya panah pendek dan hanya menunjukkan arah
def arah(U, V):
    M = np.hypot(U, V) + 1e-30
    return U / M, V / M


# Dashboard 3 panel: potensial, |E| dengan arah, validasi
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
fig.suptitle("DISTRIBUSI POTENSIAL DAN MEDAN LISTRIK BARREL-HOUSING",
             fontsize=12, fontweight="bold", color="navy")

V0_, Er0, Ez0 = hasil[0]
im1 = ax1.imshow(V0_, origin="lower", cmap="viridis", extent=[0, Lz * 1e3, 0, b * 1e3],
                 aspect="auto", vmin=0, vmax=V0, interpolation="bilinear")
barrel = Rectangle(((zcs[0] - half) * 1e3, 0), 2 * half * 1e3, a * 1e3,
                   fc="0.15", ec="orange", lw=1.6)
ax1.add_patch(barrel)
ax1.axhline(b * 1e3, color="red", lw=2)
ax1.set_title("Potensial V(r,z)")
ax1.set_xlabel("z (mm)")
ax1.set_ylabel("r (mm)")
fig.colorbar(im1, ax=ax1, label="V (Volt)")

im2 = ax2.imshow(np.hypot(Er0, Ez0), origin="lower", cmap="magma", extent=[0, Lz * 1e3, 0, b * 1e3],
                 aspect="auto", vmin=0, vmax=Ecap, interpolation="bilinear")
u0, v0 = arah(Ez0[np.ix_(ri, zi)], Er0[np.ix_(ri, zi)])
quiv = ax2.quiver(z[zi] * 1e3, r[ri] * 1e3, u0, v0, color="white", scale=28, width=0.004)
ax2.set_title("|E| dan arah medan")
ax2.set_xlabel("z (mm)")
ax2.set_ylabel("r (mm)")
fig.colorbar(im2, ax=ax2, label="|E| (V/m)")

jm0 = np.argmin(np.abs(z - zcs[0]))
ln, = ax3.plot(r[mask] * 1e3, hasil[0][0][mask, jm0], "bo", ms=3, label="numerik (tengah)")
ax3.plot(r[mask] * 1e3, Vana, "r-", lw=1.5, label="analitik koaksial")
ax3.set_ylim(0, V0 * 1.05)
ax3.set_title("Validasi V(r) bidang tengah")
ax3.set_xlabel("r (mm)")
ax3.set_ylabel("V (Volt)")
ax3.legend()
ax3.grid(alpha=0.3)


def update(f):
    V, Er, Ez = hasil[f]
    im1.set_data(V)
    im2.set_data(np.hypot(Er, Ez))
    quiv.set_UVC(*arah(Ez[np.ix_(ri, zi)], Er[np.ix_(ri, zi)]))
    barrel.set_x((zcs[f] - half) * 1e3)
    jmf = np.argmin(np.abs(z - zcs[f]))
    ln.set_ydata(V[mask, jmf])
    ax1.set_title("Potensial V(r,z)  (barrel z = %.1f mm)" % (zcs[f] * 1e3))
    return [im1, im2, quiv, barrel, ln]


ani = FuncAnimation(fig, update, frames=len(hasil), interval=100, blit=False)
plt.close(fig)

# Plot statis (barrel posisi tengah)
Vf, Erf, Ezf = hasil[fmid]
jmf = np.argmin(np.abs(z - zcs[fmid]))
figs, (b1, b2, b3) = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
figs.suptitle("DISTRIBUSI POTENSIAL DAN MEDAN LISTRIK BARREL-HOUSING (barrel z = %.1f mm)" % (zcs[fmid] * 1e3),
              fontweight="bold")

s1 = b1.imshow(Vf, origin="lower", cmap="viridis", extent=[0, Lz * 1e3, 0, b * 1e3],
               aspect="auto", vmin=0, vmax=V0, interpolation="bilinear")
b1.add_patch(Rectangle(((zcs[fmid] - half) * 1e3, 0), 2 * half * 1e3, a * 1e3,
                       fc="0.15", ec="orange", lw=1.6))
b1.axhline(b * 1e3, color="red", lw=2)
b1.set_title("Potensial V(r,z)")
b1.set_xlabel("z (mm)")
b1.set_ylabel("r (mm)")
figs.colorbar(s1, ax=b1, label="V (Volt)")

s2 = b2.imshow(np.hypot(Erf, Ezf), origin="lower", cmap="magma", extent=[0, Lz * 1e3, 0, b * 1e3],
               aspect="auto", vmin=0, vmax=Ecap, interpolation="bilinear")
us, vs = arah(Ezf[np.ix_(ri, zi)], Erf[np.ix_(ri, zi)])
b2.quiver(z[zi] * 1e3, r[ri] * 1e3, us, vs, color="white", scale=28, width=0.004)
b2.set_title("|E| dan arah medan")
b2.set_xlabel("z (mm)")
b2.set_ylabel("r (mm)")
figs.colorbar(s2, ax=b2, label="|E| (V/m)")

b3.plot(r[mask] * 1e3, Vf[mask, jmf], "bo", ms=3, label="numerik (tengah)")
b3.plot(r[mask] * 1e3, Vana, "r-", lw=1.5, label="analitik koaksial")
b3.set_title("Validasi V(r)  (err %.2f %%)" % err)
b3.set_xlabel("r (mm)")
b3.set_ylabel("V (Volt)")
b3.legend()
b3.grid(alpha=0.3)

# Output: animasi video + plot statis
# display(HTML(ani.to_html5_video()))
figs.savefig('../figures/kasus2.png', dpi=200, bbox_inches='tight')
# plt.show()
