# Kasus 1 - Distribusi panas sensor CMOS (FTCS 3D)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display

# Parameter fisis silikon
k = 148.0            # konduktivitas termal (W/m.K)
rho = 2330.0         # massa jenis (kg/m^3)
cp = 700.0           # kapasitas panas spesifik (J/kg.K)
alpha = k / (rho * cp)

T_amb = 25.0         # suhu lingkungan (C)
T_awal = 25.0        # suhu awal sensor (C)
Daya = 1.2           # daya disipasi 4K (W), tanpa x150
h_eff = 330.0        # koef konveksi efektif ke lingkungan (W/m^2.K)

Lx, Ly, Lz = 6.2e-3, 4.6e-3, 1.5e-3
Nx, Ny, Nz = 20, 20, 10
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
dz = Lz / (Nz - 1)

# Langkah waktu dari syarat stabilitas Von Neumann
fac = 1 / dx**2 + 1 / dy**2 + 1 / dz**2
dt = 0.9 * 0.5 / (alpha * fac)
t_total = 4.0
nstep = int(t_total / dt)

# Sumber panas internal di tengah die
T = np.ones((Nx, Ny, Nz)) * T_awal
q = np.zeros((Nx, Ny, Nz))
x1, x2 = int(Nx * 0.3), int(Nx * 0.7)
y1, y2 = int(Ny * 0.3), int(Ny * 0.7)
z1, z2 = int(Nz * 0.4), Nz
V_src = (x2 - x1) * dx * (y2 - y1) * dy * (z2 - z1) * dz
q[x1:x2, y1:y2, z1:z2] = Daya / V_src

# Koefisien pendinginan konveksi tiap muka (Robin eksplisit)
cx = h_eff * dt / (rho * cp * (dx / 2))
cy = h_eff * dt / (rho * cp * (dy / 2))
cz = h_eff * dt / (rho * cp * (dz / 2))

# Jadwal simpan frame rapat di awal agar penyebaran panas terlihat detail
simpan_set = set(np.unique(np.geomspace(1, nstep - 1, 90).astype(int)))
jc = Ny // 2

# Iterasi FTCS 3D
top_frames = []
side_frames = []
t_list = []
max_list = []

print("Menjalankan simulasi termal FTCS 3D ...")
for n in range(nstep):
    lap = (np.roll(T, -1, 0) - 2 * T + np.roll(T, 1, 0)) / dx**2 \
        + (np.roll(T, -1, 1) - 2 * T + np.roll(T, 1, 1)) / dy**2 \
        + (np.roll(T, -1, 2) - 2 * T + np.roll(T, 1, 2)) / dz**2
    Tn = T + alpha * dt * lap + q * dt / (rho * cp)

    # Batas Neumann untuk menghapus efek wrap np.roll
    Tn[0, :, :] = Tn[1, :, :]
    Tn[-1, :, :] = Tn[-2, :, :]
    Tn[:, 0, :] = Tn[:, 1, :]
    Tn[:, -1, :] = Tn[:, -2, :]
    Tn[:, :, 0] = Tn[:, :, 1]
    Tn[:, :, -1] = Tn[:, :, -2]

    # Batas konveksi Robin ke lingkungan
    Tn[0, :, :] -= cx * (Tn[0, :, :] - T_amb)
    Tn[-1, :, :] -= cx * (Tn[-1, :, :] - T_amb)
    Tn[:, 0, :] -= cy * (Tn[:, 0, :] - T_amb)
    Tn[:, -1, :] -= cy * (Tn[:, -1, :] - T_amb)
    Tn[:, :, 0] -= cz * (Tn[:, :, 0] - T_amb)
    Tn[:, :, -1] -= cz * (Tn[:, :, -1] - T_amb)

    T = Tn

    if n in simpan_set:
        top_frames.append(T[:, :, -1].copy())     # permukaan atas (citra kamera termal)
        side_frames.append(T[:, jc, :].copy())    # tampang samping X-Z (penyebaran ke dalam)
        t_list.append(n * dt)
        max_list.append(T.max())

print("Selesai. T_max = %.1f C (kenaikan %.1f C)" % (T.max(), T.max() - T_amb))

Thi = max(max_list)   # untuk batas sumbu telemetri

# Dashboard animasi: peta atas + tampang samping + telemetri
# Skala warna peta diatur adaptif tiap frame supaya pola penyebaran selalu terlihat (tidak blank)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.6))
fig.suptitle("DISTRIBUSI PANAS SENSOR CMOS (PEREKAMAN 4K)",
             fontsize=12, fontweight="bold", color="darkred")

im1 = ax1.imshow(top_frames[0].T, cmap="inferno", origin="lower", extent=[0, Lx * 1e3, 0, Ly * 1e3],
                 interpolation="bicubic")
ax1.set_title("Citra termal permukaan atas")
ax1.set_xlabel("X (mm)")
ax1.set_ylabel("Y (mm)")
fig.colorbar(im1, ax=ax1, label="T (C)")

im2 = ax2.imshow(side_frames[0].T, cmap="inferno", origin="lower", extent=[0, Lx * 1e3, 0, Lz * 1e3],
                 interpolation="bicubic", aspect="auto")
ax2.set_title("Penyebaran panas (tampang X-Z)")
ax2.set_xlabel("X (mm)")
ax2.set_ylabel("Z (mm)")
fig.colorbar(im2, ax=ax2, label="T (C)")


def update(f):
    top = top_frames[f]
    side = side_frames[f]
    im1.set_data(top.T)
    im1.set_clim(top.min(), top.max())     # skala adaptif: kontras pola panas selalu jelas
    im2.set_data(side.T)
    im2.set_clim(side.min(), side.max())
    ax3.clear()
    ax3.grid(True, ls=":", alpha=0.6)
    ax3.plot(t_list[:f + 1], max_list[:f + 1], "r-", lw=2, label="Suhu hotspot")
    ax3.axhline(80, color="darkorange", ls="--", label="Batas overheat (80 C)")
    ax3.set_xlim(0, t_total)
    ax3.set_ylim(T_amb - 2, Thi + 8)
    ax3.set_xlabel("Durasi perekaman (s)")
    ax3.set_ylabel("T maksimum (C)")
    ax3.legend(loc="lower right")
    suhu = max_list[f]
    if suhu < 60:
        stat, warna = "AMAN", "green"
    elif suhu < 78:
        stat, warna = "PERINGATAN: PANAS", "orange"
    else:
        stat, warna = "CRITICAL: OVERHEAT", "red"
    ax3.set_title("Telemetri  t=%.2f s  T=%.1f C  [%s]" % (t_list[f], suhu, stat), color=warna)
    return [im1, im2]


ani = FuncAnimation(fig, update, frames=len(top_frames), interval=120, blit=False)
plt.tight_layout()
plt.close()

# Plot statis kondisi akhir (2x2 Grid dengan 3D Surface)
figs = plt.figure(figsize=(12, 10))
figs.suptitle("DISTRIBUSI TERMAL SENSOR CMOS (KONDISI AKHIR)", fontweight="bold", fontsize=14)

# Panel 1: 3D Surface Plot
c1 = figs.add_subplot(2, 2, 1, projection='3d')
X_mesh, Y_mesh = np.meshgrid(np.linspace(0, Lx * 1e3, Nx), np.linspace(0, Ly * 1e3, Ny), indexing='ij')
surf = c1.plot_surface(X_mesh, Y_mesh, top_frames[-1], cmap="inferno", rstride=1, cstride=1, alpha=0.9, antialiased=True)
c1.set_title("Proyeksi Permukaan 3D Suhu Sensor")
c1.set_xlabel("X (mm)")
c1.set_ylabel("Y (mm)")
c1.set_zlabel("T (C)")
figs.colorbar(surf, ax=c1, shrink=0.5, pad=0.1, label="T (C)")
c1.view_init(elev=30, azim=-45)

# Panel 2: Tampak Atas
c2 = figs.add_subplot(2, 2, 2)
p1 = c2.imshow(top_frames[-1].T, cmap="inferno", origin="lower", extent=[0, Lx * 1e3, 0, Ly * 1e3],
               interpolation="bicubic")
c2.set_title("Suhu permukaan atas (X-Y)")
c2.set_xlabel("X (mm)")
c2.set_ylabel("Y (mm)")
figs.colorbar(p1, ax=c2, label="T (C)")

# Panel 3: Tampak Samping
c3 = figs.add_subplot(2, 2, 3)
p2 = c3.imshow(side_frames[-1].T, cmap="inferno", origin="lower", extent=[0, Lx * 1e3, 0, Lz * 1e3],
               interpolation="bicubic", aspect="auto")
c3.set_title("Tampang Penetrasi Panas (X-Z)")
c3.set_xlabel("X (mm)")
c3.set_ylabel("Z (mm)")
figs.colorbar(p2, ax=c3, label="T (C)")

# Panel 4: Telemetri
c4 = figs.add_subplot(2, 2, 4)
c4.plot(t_list, max_list, "r-", lw=2)
c4.axhline(80, color="darkorange", ls="--", label="80 C (Overheat)")
c4.set_title("Kenaikan suhu vs waktu")
c4.set_xlabel("Waktu (s)")
c4.set_ylabel("T_max (C)")
c4.grid(alpha=0.3)
c4.legend()

figs.tight_layout()

# Output: animasi video + plot statis
# display(HTML(ani.to_html5_video()))
figs.savefig('../figures/kasus1.png', dpi=200, bbox_inches='tight')
# plt.show()
