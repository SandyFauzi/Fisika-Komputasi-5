# Kasus 6 - Propagasi cahaya dome port + lensa menuju sensor CMOS

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
from numba import njit, prange

# Konstanta fisis
c0 = 299792458.0
eps0 = 8.8541878128e-12
mu0 = 4e-7 * np.pi
lam = 550e-9
omega = 2 * np.pi * c0 / lam

# Grid (konvensi Kakai: i = Grid X, j = arah rambat = Grid Y, k = kedalaman)
dx = dy = dz = 25e-9
NX, NY, NK = 200, 200, 80
kmid = NK // 2
dt = 0.99 / (c0 * np.sqrt(1 / dx**2 + 1 / dy**2 + 1 / dz**2))   # syarat CFL 3D
nstep = 800                                                     # iterasi penuh agar citra terbentuk jelas
simpan = 6

# Bidang penting
jsrc = 15           # bidang injeksi gelombang
j_sensor = 185      # bidang sensor CMOS (bidang fokus, dari pemindaian)
acc_start = 300     # mulai akumulasi citra setelah gelombang mencapai sensor

# Peta indeks bias: geometri 2D Kakai diekstrusi jadi 3D (revolusi terhadap sumbu optik)
i = np.arange(NX)[:, None, None]
j = np.arange(NY)[None, :, None]
k = np.arange(NK)[None, None, :]
center_x, center_y = 100, 170
r_luar, r_dalam = 135, 120

n = np.ones((NX, NY, NK)) * 1.33                    # air (n = 1.33)
rr = np.sqrt((i - center_x)**2 + (j - center_y)**2 + (k - kmid)**2)
n[(rr >= r_dalam) & (rr <= r_luar)] = 1.52          # cangkang dome (kaca BK7)
n[rr < r_dalam] = 1.00                              # udara dalam kamera
lensa = ((i - 100) / 40.0)**2 + ((k - kmid) / 40.0)**2 + ((j - 140) / 18.0)**2
n[lensa <= 1] = 1.50                                # lensa biconvex (kaca)

De = dt / (eps0 * n**2)
Dh = dt / mu0
tap = np.hanning(NX)[:, None] * np.hanning(NK)[None, :]


# Kernel Yee 3D (Numba paralel)
@njit(parallel=True, fastmath=True)
def upd_H(Ex, Ey, Ez, Hx, Hy, Hz, Dh, dx, dy, dz):
    NX, NY, NK = Ex.shape
    for a in prange(NX):
        for b in range(NY - 1):
            for c in range(NK - 1):
                Hx[a, b, c] += Dh * ((Ey[a, b, c + 1] - Ey[a, b, c]) / dz - (Ez[a, b + 1, c] - Ez[a, b, c]) / dy)
    for a in prange(NX - 1):
        for b in range(NY):
            for c in range(NK - 1):
                Hy[a, b, c] += Dh * ((Ez[a + 1, b, c] - Ez[a, b, c]) / dx - (Ex[a, b, c + 1] - Ex[a, b, c]) / dz)
    for a in prange(NX - 1):
        for b in range(NY - 1):
            for c in range(NK):
                Hz[a, b, c] += Dh * ((Ex[a, b + 1, c] - Ex[a, b, c]) / dy - (Ey[a + 1, b, c] - Ey[a, b, c]) / dx)


@njit(parallel=True, fastmath=True)
def upd_E(Ex, Ey, Ez, Hx, Hy, Hz, De, dx, dy, dz):
    NX, NY, NK = Ex.shape
    for a in prange(NX):
        for b in range(1, NY):
            for c in range(1, NK):
                Ex[a, b, c] += De[a, b, c] * ((Hz[a, b, c] - Hz[a, b - 1, c]) / dy - (Hy[a, b, c] - Hy[a, b, c - 1]) / dz)
    for a in prange(1, NX):
        for b in range(NY):
            for c in range(1, NK):
                Ey[a, b, c] += De[a, b, c] * ((Hx[a, b, c] - Hx[a, b, c - 1]) / dz - (Hz[a, b, c] - Hz[a - 1, b, c]) / dx)
    for a in prange(1, NX):
        for b in range(1, NY):
            for c in range(NK):
                Ez[a, b, c] += De[a, b, c] * ((Hy[a, b, c] - Hy[a - 1, b, c]) / dx - (Hx[a, b, c] - Hx[a, b - 1, c]) / dy)


# Inisialisasi medan (sumber terpolarisasi Ez, seperti mode TMz Kakai)
Ex = np.zeros((NX, NY, NK))
Ey = np.zeros((NX, NY, NK))
Ez = np.zeros((NX, NY, NK))
Hx = np.zeros((NX, NY, NK))
Hy = np.zeros((NX, NY, NK))
Hz = np.zeros((NX, NY, NK))
acc = np.zeros((NX, NK))

data_Ez = []
data_prof = []
data_cit = []
data_t = []

print("Menjalankan FDTD Yee 3D ...")
for s in range(nstep):
    upd_H(Ex, Ey, Ez, Hx, Hy, Hz, Dh, dx, dy, dz)
    upd_E(Ex, Ey, Ez, Hx, Hy, Hz, De, dx, dy, dz)

    # Sumber gelombang bidang
    Ez[:, jsrc, :] += np.sin(omega * s * dt) * tap * 0.5

    # Batas serap sederhana (ABC) di tepi domain
    ab = 0.98
    for F in (Ex, Ey, Ez):
        F[:, 0, :] *= ab
        F[:, -1, :] *= ab
        F[0, :, :] *= ab
        F[-1, :, :] *= ab
        F[:, :, 0] *= ab
        F[:, :, -1] *= ab

    # Akumulasi intensitas pada bidang sensor untuk membentuk citra
    if s > acc_start:
        acc += Ex[:, j_sensor, :]**2 + Ey[:, j_sensor, :]**2 + Ez[:, j_sensor, :]**2

    # Simpan frame
    if s % simpan == 0:
        data_Ez.append(Ez[:, :, kmid].copy())
        prof = Ez[:, j_sensor, kmid]**2 + Ex[:, j_sensor, kmid]**2 + Ey[:, j_sensor, kmid]**2
        data_prof.append(prof / (prof.max() + 1e-12))
        data_cit.append(acc.copy())
        data_t.append(s * dt * 1e15)

print("Selesai.")

nsl = n[:, :, kmid]
cropi = slice(60, 140)                      # crop citra jadi kotak di sekitar fokus
cit_max = data_cit[-1][cropi, :].max() + 1e-12   # skala tetap agar citra terlihat tumbuh

# Dashboard animasi: peta Ez (Kakai) + profil intensitas (Kakai) + citra sensor (baru)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.6))
fig.suptitle("PROPAGASI GELOMBANG CAHAYA MELALUI DOME PORT DAN LENSA (FDTD YEE 3D)",
             fontsize=11, fontweight="bold", color="darkblue")

im = ax1.imshow(data_Ez[0].T, cmap="RdBu", origin="lower", vmin=-0.2, vmax=0.2)
ax1.contour(nsl.T, levels=[1.1, 1.4], colors=["white", "cyan"], alpha=0.5, linestyles="--", linewidths=0.9)
ax1.axhline(j_sensor, color="lime", ls=":", label="Sensor CMOS")
ax1.set_title("Distribusi Medan Listrik (Ez)")
ax1.set_xlabel("Grid X")
ax1.set_ylabel("Grid Y")
ax1.legend(loc="upper right", fontsize=8)
fig.colorbar(im, ax=ax1, label="Amplitudo Ez")

line, = ax2.plot(np.arange(NX), data_prof[0], "g-", lw=2)
ax2.set_title("Profil Intensitas pada Sensor CMOS")
ax2.set_xlabel("Posisi Piksel Sensor")
ax2.set_ylabel("Intensitas Relatif")
ax2.set_xlim(0, NX)
ax2.set_ylim(0, 1.05)
ax2.grid(alpha=0.4)

ic = ax3.imshow(data_cit[0][cropi, :].T, cmap="inferno", origin="lower",
                interpolation="bilinear", aspect="auto", vmin=0, vmax=cit_max)
ax3.set_title("Citra Terbentuk di Sensor CMOS")
ax3.set_xlabel("Piksel X")
ax3.set_ylabel("Piksel Y")
fig.colorbar(ic, ax=ax3, label="Intensitas terkumpul (a.u.)")


def update(f):
    im.set_data(data_Ez[f].T)
    line.set_ydata(data_prof[f])
    ic.set_data(data_cit[f][cropi, :].T)
    ax1.set_xlabel("Grid X   |   t = %.1f fs" % data_t[f])
    return [im, line, ic]


ani = FuncAnimation(fig, update, frames=len(data_Ez), interval=80, blit=False)
plt.tight_layout()
plt.close()

# Plot statis kondisi akhir (2x2 Grid dengan 3D Surface)
figs = plt.figure(figsize=(12, 10))
figs.suptitle("PROPAGASI GELOMBANG CAHAYA MENUJU SENSOR CMOS (FDTD YEE 3D)", fontweight="bold", fontsize=14)

# Panel 1: 3D Surface Plot
b0 = figs.add_subplot(2, 2, 1, projection='3d')
X_mesh, Y_mesh = np.meshgrid(np.arange(NX), np.arange(NY), indexing='ij')
surf = b0.plot_surface(X_mesh, Y_mesh, data_Ez[-1], cmap="RdBu", rstride=2, cstride=2, alpha=0.9, antialiased=True, vmin=-0.2, vmax=0.2)
b0.set_title("Proyeksi Permukaan 3D Medan Listrik (Ez)")
b0.set_xlabel("Grid X")
b0.set_ylabel("Grid Y")
b0.set_zlabel("Amplitudo Ez")
b0.view_init(elev=45, azim=-60)
figs.colorbar(surf, ax=b0, shrink=0.5, pad=0.1, label="Ez")

# Panel 2: Peta 2D (sebelumnya b1)
b1 = figs.add_subplot(2, 2, 2)
p1 = b1.imshow(data_Ez[-1].T, cmap="RdBu", origin="lower", vmin=-0.2, vmax=0.2)
b1.contour(nsl.T, levels=[1.1, 1.4], colors=["white", "cyan"], alpha=0.5, linestyles="--", linewidths=0.9)
b1.axhline(j_sensor, color="lime", ls=":")
b1.set_title("Peta 2D Distribusi Medan Listrik (Ez)")
b1.set_xlabel("Grid X")
b1.set_ylabel("Grid Y")
figs.colorbar(p1, ax=b1, label="Ez")

# Panel 3: Profil Intensitas
b2 = figs.add_subplot(2, 2, 3)
b2.plot(np.arange(NX), data_prof[-1], "g-", lw=2)
b2.set_title("Profil Intensitas pada Sensor CMOS")
b2.set_xlabel("Posisi Piksel Sensor")
b2.set_ylabel("Intensitas Relatif")
b2.grid(alpha=0.4)

# Panel 4: Citra
b3 = figs.add_subplot(2, 2, 4)
icf = b3.imshow(acc[cropi, :].T, cmap="inferno", origin="lower", interpolation="bilinear", aspect="auto")
b3.set_title("Citra di Sensor CMOS")
b3.set_xlabel("Piksel X")
b3.set_ylabel("Piksel Y")
figs.colorbar(icf, ax=b3, label="Intensitas (a.u.)")

figs.tight_layout()

# Output: animasi video + plot statis
# display(HTML(ani.to_html5_video()))
figs.savefig('../figures/kasus6.png', dpi=200, bbox_inches='tight')
# plt.show()
