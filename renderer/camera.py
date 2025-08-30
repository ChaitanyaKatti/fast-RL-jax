import math
import numpy as np


class OrbitCamera:
    def __init__(self):
        self.radius = 10.0
        self.theta = math.pi / 4  # Polar angle
        self.phi = -3*math.pi / 4    # Azimuthal angle
        self.position = np.array([-5.0, -5.0, 5.0 * np.sqrt(2)])
        self.target = np.array([0.0, 0.0, 0.0])

        self.last_x = 0.0
        self.last_y = 0.0
        self.first_mouse = True
        self.sensitivity = 0.01

    def get_view_matrix(self):
        up = np.array([0.0, 0.0, 1.0])
        
        # Compute view matrix
        forward = self.target - self.position
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        new_up = np.cross(right, forward)

        view_matrix = np.eye(4, dtype=np.float32)
        view_matrix[0, :3] = right
        view_matrix[1, :3] = new_up
        view_matrix[2, :3] = -forward
        view_matrix[:3, 3] = -np.dot(right, self.position), - \
            np.dot(new_up, self.position), np.dot(forward, self.position)
        return view_matrix

    def process_mouse(self, xpos, ypos):
        if self.first_mouse:
            self.last_x = xpos
            self.last_y = ypos
            self.first_mouse = False
            return

        x_offset = (xpos - self.last_x) * self.sensitivity
        y_offset = (self.last_y - ypos) * self.sensitivity

        self.last_x = xpos
        self.last_y = ypos

        self.phi -= x_offset
        self.theta += y_offset
        # Constrain theta between epsilon and pi-epsilon to avoid gimbal lock
        self.theta = np.clip(self.theta, 0.1, math.pi - 0.1)
        self.update_position()
    
    def update_position(self):    
        # Convert spherical coordinates to Cartesian
        x = self.radius * math.sin(self.theta) * math.cos(self.phi)
        y = self.radius * math.sin(self.theta) * math.sin(self.phi)
        z = self.radius * math.cos(self.theta)
        self.position = np.array([x, y, z]) + self.target
    
    def get_perspective_matrix(self, fov, aspect, near, far):
        f = 1.0 / math.tan(math.radians(fov) / 2.0)
        projection = np.zeros((4, 4), dtype=np.float32)
        projection[0, 0] = f / aspect
        projection[1, 1] = f
        projection[2, 2] = (far + near) / (near - far)
        projection[2, 3] = 2.0 * far * near / (near - far)
        projection[3, 2] = -1.0
        return projection
    