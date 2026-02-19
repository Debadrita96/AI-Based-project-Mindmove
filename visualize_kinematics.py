import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# TODO: change to your data
kinematics_chosen = np.load("kinematics_day_one.pkl", allow_pickle=True)[
    "index_slow"
]
kinematics_chosen = kinematics_chosen.reshape(kinematics_chosen.shape[0], 21, 3)
kinematics_chosen = np.transpose(kinematics_chosen, (0, 2, 1))

# create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# set axis limits
ax.set_xlim(kinematics_chosen[:, 0].min(), kinematics_chosen[:, 0].max())
ax.set_ylim(kinematics_chosen[:, 1].min(), kinematics_chosen[:, 1].max())
ax.set_zlim(kinematics_chosen[:, 2].min(), kinematics_chosen[:, 2].max())

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# create joint and finger plots
(joints_plot,) = ax.plot(*kinematics_chosen[0], "o", color="black")

(thumb_plot,) = ax.plot(*kinematics_chosen[0, :, [0, 4, 3, 2, 1]].T, color="blue")
(index_plot,) = ax.plot(*kinematics_chosen[0, :, [0, 8, 7, 6, 5]].T, color="red")
(middle_plot,) = ax.plot(*kinematics_chosen[0, :, [0, 12, 11, 10, 9]].T, color="green")
(ring_plot,) = ax.plot(*kinematics_chosen[0, :, [0, 16, 15, 14, 13]].T, color="yellow")
(pinky_plot,) = ax.plot(*kinematics_chosen[0, :, [0, 20, 19, 18, 17]].T, color="orange")


sample_slider = Slider(
    ax=fig.add_axes([0.25, 0.1, 0.65, 0.03]),
    label="Sample (a. u.)",
    valmin=0,
    valmax=kinematics_chosen.shape[0] - 1,
    valstep=1,
    valinit=0,
)


def update(val):
    kinematics_new_sample = kinematics_chosen[int(val)]

    joints_plot._verts3d = tuple(kinematics_new_sample)

    thumb_plot._verts3d = tuple(kinematics_new_sample[:, [0, 4, 3, 2, 1]])
    index_plot._verts3d = tuple(kinematics_new_sample[:, [0, 8, 7, 6, 5]])
    middle_plot._verts3d = tuple(kinematics_new_sample[:, [0, 12, 11, 10, 9]])
    ring_plot._verts3d = tuple(kinematics_new_sample[:, [0, 16, 15, 14, 13]])
    pinky_plot._verts3d = tuple(kinematics_new_sample[:, [0, 20, 19, 18, 17]])

    fig.canvas.draw_idle()


sample_slider.on_changed(update)

plt.show()
