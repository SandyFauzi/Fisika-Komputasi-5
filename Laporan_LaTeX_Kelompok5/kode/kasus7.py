# Kasus 7 - Schrodinger 2D TDSE Crank-Nicolson, citra dari probabilitas
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
from numba import njit, prange

Lx, Ly = 60.0, 40.0
Nx, Ny = 200, 130
dx, dy = Lx / Nx, Ly / Ny
x = (np.arange(Nx) + 0.5) * dx - Lx / 2
y = (np.arange(Ny) + 0.5) * dy - Ly / 2
X, Y = np.meshgrid(x, y, indexing="ij")
dt = 0.02
nstep = 1000

# potensial: sumur deplesi + penghalang STI dengan celah
U = np.zeros((Nx, Ny))
U[(X > 8) & (X < 22) & (np.abs(Y) < 10)] = -1.2
U[(np.abs(X - 2.0) < 0.8) & ~(np.abs(Y) < 2.2)] = 4.0

# CAP penyerap tepi
cap = np.zeros((Nx, Ny))
for arr, L in ((np.abs(X), Lx / 2), (np.abs(Y), Ly / 2)):
    s = np.clip((arr - (L - 5.0)) / 5.0, 0, 1)
    cap += 3.0 * s**2
Uc = U - 1j * cap

# Hamiltonian 5 titik dan operator Crank-Nicolson
N = Nx * Ny
Dx = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(Nx, Nx)) / dx**2
Dy = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(Ny, Ny)) / dy**2
H = -0.5 * (sp.kron(Dx, sp.identity(Ny)) + sp.kron(sp.identity(Nx), Dy)) + sp.diags(Uc.ravel())
A_cn = (sp.identity(N, dtype=complex) + 0.5j * dt * H).tocsc()
B_cn = (sp.identity(N, dtype=complex) - 0.5j * dt * H).tocsc()
lu = spla.splu(A_cn)

# paket gelombang elektron foto-eksitasi
x0, y0, sigma, k0 = -18.0, 0.0, 3.0, 2.2
psi = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2)) * np.exp(1j * k0 * X)
psi = (psi / np.sqrt(np.sum(np.abs(psi)**2) * dx * dy)).ravel().astype(complex)
well = ((X > 8) & (X < 22) & (np.abs(Y) < 10)).ravel()


@njit(parallel=True)
def akumulasi(citra, prob, dt):
    n = prob.shape[0]
    for p in prange(n):
        citra[p] += prob[p] * dt


# integrasi waktu
citra = np.zeros(N)                       # peta muatan terkumpul (dari probabilitas)
frames_psi, frames_citra = [], []
pw, ts = [], []
for n in range(nstep + 1):
    prob = np.abs(psi)**2
    akumulasi(citra, prob, dt)
    pw.append(float(np.sum(prob[well]) * dx * dy))
    ts.append(n * dt)
    if n % 20 == 0:
        frames_psi.append(prob.reshape(Nx, Ny).T.copy())
        frames_citra.append(citra.reshape(Nx, Ny).T.copy())
    if n < nstep:
        psi = lu.solve(B_cn @ psi)

print("peluang tangkap puncak =", round(max(pw) * 100, 1), "%  pada t =", round(ts[int(np.argmax(pw))], 1))
print("peluang tangkap akhir =", round(pw[-1] * 100, 1), "%")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
fig.suptitle("DINAMIKA ELEKTRON FOTO-EKSITASI PADA SENSOR CMOS (SCHRODINGER 2D)", fontsize=11, fontweight="bold")

im1 = ax1.imshow(frames_psi[0], origin="lower", cmap="magma",
                 extent=[x[0], x[-1], y[0], y[-1]], aspect="auto", interpolation="bilinear")
ax1.contour(x, y, (U.T > 2), levels=[0.5], colors="cyan", linewidths=1.2)
ax1.contour(x, y, (U.T < -0.5), levels=[0.5], colors="lime", linewidths=1.2)
ax1.set_title("Rapat peluang |psi|^2 (elektron bergerak)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

im2 = ax2.imshow(frames_citra[0], origin="lower", cmap="cividis",
                 extent=[x[0], x[-1], y[0], y[-1]], aspect="auto", interpolation="bilinear")
fig.colorbar(im2, ax=ax2, label="muatan terkumpul (a.u.)")
ax2.contour(x, y, (U.T < -0.5), levels=[0.5], colors="lime", linewidths=1.2)
ax2.set_title("Citra dari probabilitas (sinyal piksel)")
ax2.set_xlabel("x")
ax2.set_ylabel("y")

garis, = ax3.plot([], [], "g-", lw=2, label="P(dalam sumur)")
ax3.set_xlim(0, nstep * dt)
ax3.set_ylim(0, 1)
ax3.grid(alpha=0.3)
ax3.set_title("Efisiensi penangkapan elektron")
ax3.set_xlabel("t")
ax3.set_ylabel("probabilitas")
ax3.legend(loc="upper left", fontsize=8)

ft = ts[::20]
fp = pw[::20]


def update(f):
    im1.set_data(frames_psi[f])
    im1.set_clim(0, max(frames_psi[f].max(), 1e-9))
    im2.set_data(frames_citra[f])
    im2.set_clim(0, max(frames_citra[f].max(), 1e-9))
    garis.set_data(ft[:f + 1], fp[:f + 1])
    return im1, im2, garis


ani = FuncAnimation(fig, update, frames=len(frames_psi), interval=100, blit=False)
plt.close(fig)

# Plot statis kondisi akhir
figs, (cx1, cx2, cx3) = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
figs.suptitle("DINAMIKA ELEKTRON FOTO-EKSITASI PADA SENSOR CMOS (KONDISI AKHIR)", fontweight="bold")
cx1.imshow(frames_psi[-1], origin="lower", cmap="magma", extent=[x[0], x[-1], y[0], y[-1]],
           aspect="auto", interpolation="bilinear")
cx1.contour(x, y, (U.T > 2), levels=[0.5], colors="cyan", linewidths=1.2)
cx1.contour(x, y, (U.T < -0.5), levels=[0.5], colors="lime", linewidths=1.2)
cx1.set_title("Rapat peluang akhir")
cx1.set_xlabel("x")
cx1.set_ylabel("y")
sm = cx2.imshow(frames_citra[-1], origin="lower", cmap="cividis", extent=[x[0], x[-1], y[0], y[-1]],
                aspect="auto", interpolation="bilinear")
figs.colorbar(sm, ax=cx2, label="muatan terkumpul (a.u.)")
cx2.contour(x, y, (U.T < -0.5), levels=[0.5], colors="lime", linewidths=1.2)
cx2.set_title("Citra dari probabilitas (sinyal piksel)")
cx2.set_xlabel("x")
cx2.set_ylabel("y")
cx3.plot(ft, fp, "g-", lw=2)
cx3.set_xlim(0, nstep * dt)
cx3.set_ylim(0, 1)
cx3.grid(alpha=0.3)
cx3.set_title("Efisiensi penangkapan elektron")
cx3.set_xlabel("t")
cx3.set_ylabel("probabilitas")

# Output: animasi video + plot statis
# display(HTML(ani.to_html5_video()))
figs.savefig('../figures/kasus7.png', dpi=200, bbox_inches='tight')
# plt.show()
