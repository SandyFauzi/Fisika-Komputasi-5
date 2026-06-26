# Kasus 3 - Bola dielektrik (dome) dalam medan seragam (Laplace, FV), dashboard 3 panel
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
from numba import njit

A_dome = 50e-3
E0 = 1.0e3
Rmax = 300e-3
Nr, Nt = 120, 100
dr, dth = Rmax / Nr, np.pi / Nt
r = (np.arange(Nr) + 0.5) * dr
th = (np.arange(Nt) + 0.5) * dth
cth = np.cos(th)
sthc = np.sin(th)
rf = np.arange(Nr + 1) * dr
sthf = np.sin(np.arange(Nt + 1) * dth)


# Perakitan matriks finite volume div(eps grad V) = 0 (Numba). Node p = i*Nt + j
@njit
def rakit(eps, r, rf, sthf, sthc, cth, dr, dth, E0):
    Nr = r.size
    Nt = cth.size
    N = Nr * Nt
    R = np.empty(N * 5, np.int64)
    C = np.empty(N * 5, np.int64)
    Vv = np.empty(N * 5, np.float64)
    rhs = np.zeros(N)
    c = 0

    for i in range(Nr):
        for j in range(Nt):
            p = i * Nt + j

            # batas luar, medan seragam V = -E0 r cos(theta)
            if i == Nr - 1:
                R[c] = p; C[c] = p; Vv[c] = 1.0; c += 1
                rhs[p] = -E0 * r[i] * cth[j]
                continue

            diag = 0.0
            ef = 0.5 * (eps[i, j] + eps[i + 1, j])
            aR = ef * rf[i + 1]**2 * sthc[j] * dth / dr
            R[c] = p; C[c] = (i + 1) * Nt + j; Vv[c] = aR; c += 1
            diag -= aR

            if i > 0:
                ef = 0.5 * (eps[i, j] + eps[i - 1, j])
                aRm = ef * rf[i]**2 * sthc[j] * dth / dr
                R[c] = p; C[c] = (i - 1) * Nt + j; Vv[c] = aRm; c += 1
                diag -= aRm

            if j < Nt - 1:
                ef = 0.5 * (eps[i, j] + eps[i, j + 1])
                aT = ef * sthf[j + 1] * dr / dth
                R[c] = p; C[c] = p + 1; Vv[c] = aT; c += 1
                diag -= aT

            if j > 0:
                ef = 0.5 * (eps[i, j] + eps[i, j - 1])
                aTm = ef * sthf[j] * dr / dth
                R[c] = p; C[c] = p - 1; Vv[c] = aTm; c += 1
                diag -= aTm

            R[c] = p; C[c] = p; Vv[c] = diag; c += 1

    return R[:c], C[:c], Vv[:c], rhs


# Selesaikan potensial dan medan untuk satu permitivitas dome
def solve(eps_in):
    eps = np.where(r[:, None] <= A_dome, eps_in, 1.0) * np.ones((Nr, Nt))
    R, C, Vv, rhs = rakit(eps, r, rf, sthf, sthc, cth, dr, dth, E0)
    M = sp.csr_matrix((Vv, (R, C)), shape=(Nr * Nt, Nr * Nt))
    V = spla.spsolve(M, rhs).reshape(Nr, Nt)
    Er = np.zeros_like(V)
    Et = np.zeros_like(V)
    Er[1:-1, :] = -(V[2:, :] - V[:-2, :]) / (2 * dr)
    Et[:, 1:-1] = -(V[:, 2:] - V[:, :-2]) / (2 * dth) / r[:, None]
    Emag = np.hypot(Er, Et)
    Ex = Er * sthc[None, :] + Et * cth[None, :]
    Ez = Er * cth[None, :] - Et * sthc[None, :]
    return V, Emag, Ex, Ez


# Validasi medan dalam bola
Vd, Emd, Exd, Ezd = solve(3.4)
inti = r[:, None] <= 0.6 * A_dome
Ez_num = float(np.mean(Ezd[np.broadcast_to(inti, Ezd.shape)]))
Ez_ana = 3 / (3.4 + 2) * E0
print("E dalam numerik =", round(Ez_num, 1), "V/m ; analitik =", round(Ez_ana, 1), "V/m")

# Ramp permitivitas dome 1.0 sampai 3.4 (durasi animasi sekitar 5 detik)
eps_list = np.linspace(1.0, 3.4, 50)
hasil = [solve(e) for e in eps_list]

# Koordinat meridian dan subgrid panah
TH, RR = np.meshgrid(th, r)
Xc = RR * np.sin(TH) * 1e3
Zc = RR * np.cos(TH) * 1e3
ii = np.arange(3, Nr, 7)
jj = np.arange(2, Nt, 6)
Xq = (r[ii][:, None] * np.sin(th[jj])[None, :]) * 1e3
Zq = (r[ii][:, None] * np.cos(th[jj])[None, :]) * 1e3
tt = np.linspace(0, np.pi, 120)
xc = A_dome * np.sin(tt) * 1e3
zc = A_dome * np.cos(tt) * 1e3
vmax = np.percentile(hasil[-1][1], 99)
lim = 2.2 * A_dome * 1e3


# Vektor satuan supaya panah pendek dan hanya menunjukkan arah
def arah(U, V):
    M = np.hypot(U, V) + 1e-30
    return U / M, V / M


# Dashboard 3 panel: potensial, |E| dengan arah, profil E_z di sumbu
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
fig.suptitle("DISTRIBUSI MEDAN LISTRIK DOME PORT DIELEKTRIK (MEDAN SERAGAM)",
             fontsize=11, fontweight="bold", color="teal")

V0_, Em0, Ex0, Ez0 = hasil[0]
m1 = ax1.pcolormesh(Xc, Zc, V0_, shading="gouraud", cmap="RdBu_r")
ax1.plot(xc, zc, "g--", lw=1.6, label="dome")
ax1.set_aspect("equal")
ax1.set_xlim(0, lim)
ax1.set_ylim(-lim, lim)
ax1.set_title("Potensial V")
ax1.set_xlabel("x (mm)")
ax1.set_ylabel("z (mm)")
ax1.legend(loc="upper right", fontsize=8)
fig.colorbar(m1, ax=ax1, label="V (Volt)")

m2 = ax2.pcolormesh(Xc, Zc, Em0, shading="gouraud", cmap="inferno", vmin=0, vmax=vmax)
u0, v0 = arah(Ex0[np.ix_(ii, jj)], Ez0[np.ix_(ii, jj)])
quiv = ax2.quiver(Xq, Zq, u0, v0, color="white", scale=26, width=0.005)
ax2.plot(xc, zc, "c--", lw=1.6)
ax2.set_aspect("equal")
ax2.set_xlim(0, lim)
ax2.set_ylim(-lim, lim)
ax2.set_title("|E| dan arah medan")
ax2.set_xlabel("x (mm)")
ax2.set_ylabel("z (mm)")
fig.colorbar(m2, ax=ax2, label="|E| (V/m)")

ln, = ax3.plot(r * 1e3, Ez0[:, 0], "b-", label="E_z numerik")
ref, = ax3.plot([0, Rmax * 1e3], [E0, E0], "r--", label="medan dalam (analitik)")
ax3.axvline(A_dome * 1e3, color="green", ls=":")
ax3.set_xlim(0, 3 * A_dome * 1e3)
ax3.set_ylim(0, E0 * 1.15)
ax3.set_title("Validasi E_z di sumbu")
ax3.set_xlabel("r (mm)")
ax3.set_ylabel("E_z (V/m)")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)


def update(f):
    V, Em, Ex, Ez = hasil[f]
    m1.set_array(V.ravel())
    m2.set_array(Em.ravel())
    quiv.set_UVC(*arah(Ex[np.ix_(ii, jj)], Ez[np.ix_(ii, jj)]))
    ln.set_ydata(Ez[:, 0])
    ez_ana = 3 / (eps_list[f] + 2) * E0
    ref.set_ydata([ez_ana, ez_ana])
    ax1.set_title("Potensial V  (eps_r = %.2f)" % eps_list[f])
    return [m1, m2, quiv, ln, ref]


ani = FuncAnimation(fig, update, frames=len(hasil), interval=100, blit=False)
plt.close(fig)

# Plot statis (akrilik penuh eps_r = 3.4)
Vf, Emf, Exf, Ezf = hasil[-1]
figs, (b1, b2, b3) = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
figs.suptitle("DISTRIBUSI MEDAN LISTRIK DOME PORT DIELEKTRIK (eps_r = 3.4)", fontweight="bold")

c1 = b1.pcolormesh(Xc, Zc, Vf, shading="gouraud", cmap="RdBu_r")
b1.plot(xc, zc, "g--", lw=1.6)
b1.set_aspect("equal")
b1.set_xlim(0, lim)
b1.set_ylim(-lim, lim)
b1.set_title("Potensial V")
b1.set_xlabel("x (mm)")
b1.set_ylabel("z (mm)")
figs.colorbar(c1, ax=b1, label="V (Volt)")

c2 = b2.pcolormesh(Xc, Zc, Emf, shading="gouraud", cmap="inferno", vmin=0, vmax=vmax)
us, vs = arah(Exf[np.ix_(ii, jj)], Ezf[np.ix_(ii, jj)])
b2.quiver(Xq, Zq, us, vs, color="white", scale=26, width=0.005)
b2.plot(xc, zc, "c--", lw=1.6)
b2.set_aspect("equal")
b2.set_xlim(0, lim)
b2.set_ylim(-lim, lim)
b2.set_title("|E| dan arah medan")
b2.set_xlabel("x (mm)")
b2.set_ylabel("z (mm)")
figs.colorbar(c2, ax=b2, label="|E| (V/m)")

b3.plot(r * 1e3, Ezf[:, 0], "b-", label="E_z numerik")
b3.axhline(Ez_ana, color="r", ls="--", label="analitik dalam = %.0f V/m" % Ez_ana)
b3.axhline(E0, color="gray", ls=":", label="E0 = %.0f V/m" % E0)
b3.axvline(A_dome * 1e3, color="green", ls=":")
b3.set_xlim(0, 3 * A_dome * 1e3)
b3.set_title("Validasi E_z di dalam vs luar dome")
b3.set_xlabel("r (mm)")
b3.set_ylabel("E_z (V/m)")
b3.legend(fontsize=8)
b3.grid(alpha=0.3)

# Output: animasi video + plot statis
# display(HTML(ani.to_html5_video()))
figs.savefig('../figures/kasus3.png', dpi=200, bbox_inches='tight')
# plt.show()
