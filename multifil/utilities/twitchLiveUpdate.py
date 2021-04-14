import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob as g
import json
import numpy as np

UUID = input("run name? ")

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
filename = "C:/tmp/" + UUID + "/" + UUID + ".data.json"
m_filename = g.glob("C:/tmp/" + UUID + "/" + UUID + ".meta.json")[0]

with open(m_filename, "r") as metafile:
    meta = json.load(metafile)
# for key in meta.keys():
#     print(key)
ap = meta['pCa']
for i in range(len(ap)):
    ap[i] = 10 ** -ap[i]
x_ap_ar = list(np.arange(0, len(ap) / 2, 0.5))


def animate(_):
    try:
        with open(filename, "r") as datafile:
            data = json.load(datafile)
        y_ar = data['axial_force']
        x_ar = list(np.arange(0, len(y_ar) / 2, 0.5))
        ax1.clear()
        max_force = 105
        if max(y_ar) > max_force:
            max_force = max(y_ar)
        ax1.set_ylim(-5, max_force)
        ax2 = ax1.twinx()
        ax2.plot(x_ap_ar, ap, color='blue')
        ax1.plot(x_ar, y_ar, color='black')
    except ValueError:
        print("concurrent access issue, passing this time")
    except FileNotFoundError:
        print("*", end="")


ani = animation.FuncAnimation(fig, animate, interval=750)
plt.show()