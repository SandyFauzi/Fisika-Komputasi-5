# Uji sensitivitas CAP untuk Kasus 7 (jawaban empiris pertanyaan reviewer Q1)
# Inti Crank-Nicolson sama persis dengan notebook, hanya visualisasi dibuang.
# Yang divariasikan: kekuatan (A_cap) dan lebar (w_cap) penyerap di tepi.

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Grid dan parameter (identik notebook)
Lx, Ly = 60.0, 40.0
Nx, Ny = 200, 130
dx, dy = Lx / Nx, Ly / Ny
x = (np.arange(Nx) + 0.5) * dx - Lx / 2
y = (np.arange(Ny) + 0.5) * dy - Ly / 2
X, Y = np.meshgrid(x, y, indexing="ij")
dt = 0.02
nstep = 1000

# Potensial: sumur deplesi + penghalang STI dengan celah (identik notebook)
U = np.zeros((Nx, Ny))
U[(X > 8) & (X < 22) & (np.abs(Y) < 10)] = -1.2
U[(np.abs(X - 2.0) < 0.8) & ~(np.abs(Y) < 2.2)] = 4.0
well = ((X > 8) & (X < 22) & (np.abs(Y) < 10)).ravel()

# Paket gelombang awal (identik notebook)
x0, y0, sigma, k0 = -18.0, 0.0, 3.0, 2.2
psi0 = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2)) * np.exp(1j * k0 * X)
psi0 = (psi0 / np.sqrt(np.sum(np.abs(psi0)**2) * dx * dy)).ravel().astype(complex)

N = Nx * Ny
Dx = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(Nx, Nx)) / dx**2
Dy = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(Ny, Ny)) / dy**2
Lap = sp.kron(Dx, sp.identity(Ny)) + sp.kron(sp.identity(Nx), Dy)


# Satu run lengkap untuk satu setelan CAP, kembalikan kurva P_sumur(t)
def run(A_cap, w_cap):
    cap = np.zeros((Nx, Ny))
    if A_cap > 0:
        for arr, L in ((np.abs(X), Lx / 2), (np.abs(Y), Ly / 2)):
            s = np.clip((arr - (L - w_cap)) / w_cap, 0, 1)
            cap += A_cap * s**2
    Uc = U - 1j * cap
    H = -0.5 * Lap + sp.diags(Uc.ravel())
    A_cn = (sp.identity(N, dtype=complex) + 0.5j * dt * H).tocsc()
    B_cn = (sp.identity(N, dtype=complex) - 0.5j * dt * H).tocsc()
    lu = spla.splu(A_cn)

    psi = psi0.copy()
    pw = []
    for n in range(nstep + 1):
        prob = np.abs(psi)**2
        pw.append(float(np.sum(prob[well]) * dx * dy))
        if n < nstep:
            psi = lu.solve(B_cn @ psi)
    return np.array(pw)


# Setelan yang diuji: baseline notebook (A=3, w=5), variasi kekuatan, variasi lebar,
# dan tanpa CAP sebagai pembanding (tepi memantul)
setelan = [
    ("A=1.5 w=5", 1.5, 5.0),
    ("A=3   w=5 (baseline)", 3.0, 5.0),
    ("A=6   w=5", 6.0, 5.0),
    ("A=12  w=5", 12.0, 5.0),
    ("A=3   w=3", 3.0, 3.0),
    ("A=3   w=8", 3.0, 8.0),
    ("tanpa CAP (mantul)", 0.0, 5.0),
]

t = np.arange(nstep + 1) * dt
hasil = {}
print("%-24s %10s %10s %10s" % ("setelan CAP", "P_puncak", "t_puncak", "P_akhir"))
for nama, A, w in setelan:
    pw = run(A, w)
    hasil[nama] = pw
    print("%-24s %9.1f%% %10.2f %9.1f%%" % (nama, pw.max() * 100, t[pw.argmax()], pw[-1] * 100))

# Plot kurva P_sumur(t) semua setelan
fig, ax = plt.subplots(figsize=(9, 5.5))
for nama, A, w in setelan:
    gaya = "--" if "tanpa" in nama else "-"
    ax.plot(t, hasil[nama] * 100, gaya, lw=1.8, label=nama)
ax.set_xlabel("waktu (a.u.)")
ax.set_ylabel("P di dalam sumur (%)")
ax.set_title("Uji sensitivitas CAP: peluang tangkap vs waktu untuk berbagai setelan penyerap")
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("uji_sensitivitas_CAP.png", dpi=130)
print("plot tersimpan: uji_sensitivitas_CAP.png")
