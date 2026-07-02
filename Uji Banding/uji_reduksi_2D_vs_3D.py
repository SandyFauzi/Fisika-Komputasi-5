# Uji reduksi simetri 2D vs 3D penuh untuk Kasus 2 (jawaban empiris pertanyaan reviewer Q3)
# (a) geometri simetris: solusi 3D Kartesius harus cocok dengan rumus koaksial analitik
# (b) geometri tak simetris: elektroda digeser dari sumbu, lihat seberapa besar medan berubah

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit

# Geometri sama dengan Kasus 2 (satuan mm)
a = 4.0
b = 10.0
Lz = 30.0
V0 = 330.0
half = 7.0
zc = Lz / 2

# Grid Kartesius seragam, dx = dy = dz = 1/3 mm
h = 1.0 / 3.0
xs = np.arange(-b, b + h / 2, h)
zs = np.arange(0, Lz + h / 2, h)
Nx = Ny = xs.size
Nz = zs.size
X, Y, Z = np.meshgrid(xs, xs, zs, indexing="ij")
print("grid 3D:", Nx, "x", Ny, "x", Nz, "=", Nx * Ny * Nz, "titik")


# SOR Gauss-Seidel (Numba). Titik fixed tidak diubah.
@njit(cache=True)
def sor(V, fixed, nsweep, omega):
    Nx, Ny, Nz = V.shape
    for it in range(nsweep):
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                for k in range(1, Nz - 1):
                    if fixed[i, j, k]:
                        continue
                    vnew = (V[i + 1, j, k] + V[i - 1, j, k] + V[i, j + 1, k] +
                            V[i, j - 1, k] + V[i, j, k + 1] + V[i, j, k - 1]) / 6.0
                    V[i, j, k] += omega * (vnew - V[i, j, k])


# Bangun dan selesaikan satu konfigurasi; x0 = geseran pusat elektroda dari sumbu
def solve3d(x0):
    rho_cas = np.sqrt(X**2 + Y**2)
    rho_ele = np.sqrt((X - x0)**2 + Y**2)
    fixed = np.zeros(X.shape, np.bool_)
    V = np.zeros(X.shape)

    fixed[rho_cas >= b] = True                       # casing silinder, V = 0
    ele = (rho_ele <= a) & (np.abs(Z - zc) <= half)  # elektroda dalam, V = V0
    fixed[ele] = True
    V[ele] = V0
    fixed[:, :, 0] = True                            # tutup ujung tabung, V = 0
    fixed[:, :, -1] = True

    # sapu bertahap sampai konvergen
    for tahap in range(20):
        Vlama = V.copy()
        sor(V, fixed, 1000, 1.9)
        delta = np.abs(V - Vlama).max()
        if delta < 1e-3:
            break
    print("  konvergen: delta akhir %.2e V setelah %d sapuan" % (delta, (tahap + 1) * 1000))
    return V, fixed


def emag(V):
    E = np.zeros(V.shape)
    Ex = np.zeros(V.shape)
    Ey = np.zeros(V.shape)
    Ez = np.zeros(V.shape)
    Ex[1:-1, :, :] = -(V[2:, :, :] - V[:-2, :, :]) / (2 * h)
    Ey[:, 1:-1, :] = -(V[:, 2:, :] - V[:, :-2, :]) / (2 * h)
    Ez[:, :, 1:-1] = -(V[:, :, 2:] - V[:, :, :-2]) / (2 * h)
    return np.sqrt(Ex**2 + Ey**2 + Ez**2)


ic = Nx // 2
kc = Nz // 2

# (a) Geometri simetris: bandingkan profil V(r) bidang tengah dengan rumus koaksial
print("menjalankan 3D simetris ...")
Vs, fxs = solve3d(0.0)
xp = xs[ic:]
mask = (xp >= a) & (xp <= b)
Vnum = Vs[ic:, ic, kc][mask]
Vana = V0 * np.log(b / xp[mask]) / np.log(b / a)
err3d = np.max(np.abs(Vnum - Vana)) / V0 * 100
print("selisih 3D penuh vs analitik koaksial = %.2f %%  (2D aksisimetrik: 2.37 %%)" % err3d)

# (b) Geometri tak simetris: elektroda digeser 2 mm dari sumbu
print("menjalankan 3D elektroda geser 2 mm ...")
Vg, fxg = solve3d(2.0)
Es = emag(Vs)
Eg = emag(Vg)

# Medan diukur di TENGAH CELAH tiap sisi (jauh dari permukaan, bebas artefak tangga grid)
def e_di(E, x_mm):
    i = int(round((x_mm + b) / h))
    return E[i, ic, kc]

# geser 2 mm: celah sempit +x dari 6..10 (tengah 8), celah lebar -x dari -2..-10 (tengah -6)
E_sempit = e_di(Eg, 8.0)
E_lebar = e_di(Eg, -6.0)
# pembanding simetris di titik dengan jarak sama dari permukaan elektroda (1/4 celah 6 mm -> r = 6 dan 7? pakai r = 7, tengah celah simetris)
E_sim_tengah = e_di(Es, 7.0)
E_ana_tengah = V0 / (7.0 * np.log(b / a))
print("E tengah celah simetris (r = 7 mm)  = %.1f V/mm  (analitik %.1f)" % (E_sim_tengah, E_ana_tengah))
print("E tengah celah SEMPIT (x = +8 mm)   = %.1f V/mm" % E_sempit)
print("E tengah celah LEBAR  (x = -6 mm)   = %.1f V/mm" % E_lebar)
print("rasio sempit/lebar = %.2f  -> geser 2 mm mengubah medan lokal ~%.0f %%" %
      (E_sempit / E_lebar, (E_sempit / E_lebar - 1) * 100))

# Gambar: irisan bidang tengah |E| simetris vs geser + profil validasi
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
fig.suptitle("UJI REDUKSI 2D vs 3D PENUH: KAPASITOR SIMETRIS vs ELEKTRODA GESER 2 mm", fontweight="bold")

vmax = np.percentile(Eg[:, :, kc], 99.5)
p1 = a1.imshow(Es[:, :, kc].T, origin="lower", extent=[-b, b, -b, b], cmap="magma", vmin=0, vmax=vmax)
a1.set_title("|E| bidang tengah, simetris")
a1.set_xlabel("x (mm)")
a1.set_ylabel("y (mm)")
fig.colorbar(p1, ax=a1, label="|E| (V/mm)")

p2 = a2.imshow(Eg[:, :, kc].T, origin="lower", extent=[-b, b, -b, b], cmap="magma", vmin=0, vmax=vmax)
a2.set_title("|E| bidang tengah, elektroda geser 2 mm")
a2.set_xlabel("x (mm)")
a2.set_ylabel("y (mm)")
fig.colorbar(p2, ax=a2, label="|E| (V/mm)")

a3.plot(xp[mask], Vnum, "bo", ms=3, label="3D penuh (simetris)")
a3.plot(xp[mask], Vana, "r-", lw=1.5, label="analitik koaksial")
a3.set_title("Validasi V(r) bidang tengah (err %.2f %%)" % err3d)
a3.set_xlabel("r (mm)")
a3.set_ylabel("V (Volt)")
a3.legend()
a3.grid(alpha=0.3)

fig.savefig("uji_reduksi_2D_vs_3D.png", dpi=130)
print("plot tersimpan: uji_reduksi_2D_vs_3D.png")
