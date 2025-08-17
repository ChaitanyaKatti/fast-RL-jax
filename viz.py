import Ogre.HighPy as ohi
import numpy as np

# Add the path to your mesh files
ohi.user_resource_locations.add("./")

# Create a window and display the mesh
ohi.window_create("hello_world", (800, 600))
ohi.imshow("hello_world", )
ohi.mesh_show("hello_world", "cf2.glb", position=(0.2, 0.2, -1))
ohi.point_light("hello_world", position=(0, 0, 100))

k = np.array([[800, 0, 400],
              [0, 800, 400],
              [0, 0, 1]])
ohi.camera_intrinsics("hello_world", k, (800, 600))
# Start the rendering loop
while ohi.window_draw("hello_world") != 27:
    pass
