# Kasus 5 - Distribusi medan magnet OIS (bola termagnetisasi seragam)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display

# Parameter magnet
mu0 = 4 * np.pi * 1e-7
R = 1.0
M = 1.0e6                            # magnetisasi (A/m), orde magnet NdFeB
m = (4 / 3) * np.pi * R**3 * M       # momen dipol magnetik

# Domain
x = np.linspace(-4, 4, 300)
y = np.linspace(-4, 4, 300)
X, Y = np.meshgrid(x, y)
r = np.sqrt(X**2 + Y**2)
ca = X / (r + 1e-12)
sa = Y / (r + 1e-12)
ins = r < R
mid = len(y) // 2


# Medan untuk arah magnetisasi phi
def field(phi):
    mx = np.cos(phi)
    my = np.sin(phi)
    mdotr = mx * ca + my * sa
    pref = (mu0 / (4 * np.pi)) * m / (r**3 + 1e-12)
    Bx = pref * (3 * mdotr * ca - mx)     # eksterior: dipol
    By = pref * (3 * mdotr * sa - my)
    Bx[ins] = (2 / 3) * mu0 * M * mx      # interior: seragam (2/3) mu0 M
    By[ins] = (2 / 3) * mu0 * M * my
    return Bx, By


phis = np.linspace(0, 2 * np.pi, 30, endpoint=False)
Bx0, By0 = field(0.0)
Bmag0 = np.sqrt(Bx0**2 + By0**2)
vmax = np.percentile(Bmag0, 99)
print("B interior =", (2 / 3) * mu0 * M, "Tesla")

# Dashboard animasi: |B| + garis fluks + profil
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("DISTRIBUSI MEDAN MAGNET OIS (BOLA TERMAGNETISASI)",
             fontsize=12, fontweight="bold", color="purple")

im = ax1.imshow(Bmag0, origin="lower", extent=[-4, 4, -4, 4], cmap="inferno", vmin=0, vmax=vmax)
ax1.set_title("Distribusi Medan Magnet")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
fig.colorbar(im, ax=ax1, label="|B| (Tesla)")

ax2.streamplot(X, Y, Bx0, By0, density=1.4)
ax2.set_title("Garis Fluks Magnetik")
ax2.set_xlabel("x")
ax2.set_ylabel("y")

ln, = ax3.plot(x, Bmag0[mid, :], lw=2)
ax3.set_ylim(0, vmax * 1.1)
ax3.set_title("Profil Medan Magnet")
ax3.set_xlabel("x")
ax3.set_ylabel("|B| (Tesla)")
ax3.grid(True)


def update(f):
    Bx, By = field(phis[f])
    Bm = np.sqrt(Bx**2 + By**2)
    im.set_data(Bm)
    ax2.clear()
    ax2.set_title("Garis Fluks Magnetik")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.streamplot(X, Y, Bx, By, density=1.4)
    ln.set_ydata(Bm[mid, :])
    ax1.set_title("Distribusi Medan Magnet  (arah %d deg)" % int(np.degrees(phis[f])))
    return [im, ln]


ani = FuncAnimation(fig, update, frames=len(phis), interval=120, blit=False)
plt.tight_layout()
plt.close()

# Plot statis (arah magnetisasi sumbu x)
figs, (b1, b2, b3) = plt.subplots(1, 3, figsize=(15, 5))
figs.suptitle("DISTRIBUSI MEDAN MAGNET OIS (arah magnetisasi sumbu x)", fontweight="bold")

s1 = b1.imshow(Bmag0, origin="lower", extent=[-4, 4, -4, 4], cmap="inferno", vmin=0, vmax=vmax)
b1.set_title("Distribusi Medan Magnet")
b1.set_xlabel("x")
b1.set_ylabel("y")
figs.colorbar(s1, ax=b1, label="|B| (Tesla)")

b2.streamplot(X, Y, Bx0, By0, density=1.4)
b2.set_title("Garis Fluks Magnetik")
b2.set_xlabel("x")
b2.set_ylabel("y")

b3.plot(x, Bmag0[mid, :], lw=2)
b3.set_title("Profil Medan Magnet")
b3.set_xlabel("x")
b3.set_ylabel("|B| (Tesla)")
b3.grid(True)

figs.tight_layout()

# Output: animasi video + plot statis
# display(HTML(ani.to_html5_video()))
figs.savefig('../figures/kasus5.png', dpi=200, bbox_inches='tight')
# plt.show()
