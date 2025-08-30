import math
import time
import pygame
import glfw
import numpy as np
from OpenGL.GL import *
from .shaders import simple_shader, text_shader
from .camera import OrbitCamera
from .model import ColladaModel


class DroneRenderer:
    def __init__(self, width=800, height=600):
        if not glfw.init():
            raise Exception("GLFW initialization failed!")

        # Initialize pygame for text rendering
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont('monospace', 24)

        # FPS calculation variables
        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 144
        self.fps_update_interval = 0.5  # Update FPS every 0.5 seconds

        # Window dimensions
        self.width = width
        self.height = height
        self.camera = OrbitCamera()

        # Create window with OpenGL context
        glfw.window_hint(glfw.DECORATED, glfw.TRUE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
        self.window = glfw.create_window(
            width, height, "Drone", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Window creation failed!")

        glfw.make_context_current(self.window)
        glfw.swap_interval(0)  # Disable V-Sync

        # Set callbacks
        glfw.set_cursor_pos_callback(self.window, self._mouse_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_window_size_callback(
            self.window, self._window_resize_callback)

        # Compile shaders
        self.simple_shader = simple_shader()
        self.text_shader = text_shader()

        # Create FPS texture
        self.fps_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.fps_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        # Create and setup axis VAO/VBO
        self.axis_vertices = np.array([
            # X-axis (Red)
            0, 0, 0,  1, 0, 0,
            1, 0, 0,  1, 0, 0,

            # Y-axis (Green)
            0, 0, 0,  0, 1, 0,
            0, 1, 0,  0, 1, 0,

            # Z-axis (Blue)
            0, 0, 0,  0, 0, 1,
            0, 0, 1,  0, 0, 1
        ], dtype=np.float32)

        self.axis_VAO = glGenVertexArrays(1)
        self.axis_VBO = glGenBuffers(1)

        glBindVertexArray(self.axis_VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.axis_VBO)
        glBufferData(GL_ARRAY_BUFFER, self.axis_vertices.nbytes,
                     self.axis_vertices, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                              6 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
                              6 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)

        # Create and setup grid VAO/VBO
        grid_size = 2
        grid_step = 1.0
        grid_vertices = []
        grid_color = [0.5, 0.5, 0.5]  # Gray color for grid

        for i in np.arange(-grid_size, grid_size + grid_step, grid_step):
            # Lines along X-axis
            grid_vertices.extend([i, -grid_size, 0] + grid_color)
            grid_vertices.extend([i,  grid_size, 0] + grid_color)

            # Lines along Y-axis
            grid_vertices.extend([-grid_size, i, 0] + grid_color)
            grid_vertices.extend([ grid_size, i, 0] + grid_color)

        self.grid_vertices = np.array(grid_vertices, dtype=np.float32)

        self.grid_VAO = glGenVertexArrays(1)
        self.grid_VBO = glGenBuffers(1)

        glBindVertexArray(self.grid_VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.grid_VBO)
        glBufferData(
            GL_ARRAY_BUFFER,
            self.grid_vertices.nbytes,
            self.grid_vertices,
            GL_STATIC_DRAW,
        )
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)

        # Text VAO/VBO setup
        self.text_VAO = glGenVertexArrays(1)
        self.text_VBO = glGenBuffers(1)
        glBindVertexArray(self.text_VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.text_VBO)
        glBufferData(GL_ARRAY_BUFFER, 24 * 4, None, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        # Point VAO/VBO
        self.point_VAO = glGenVertexArrays(1)
        self.point_VBO = glGenBuffers(1)
        self.point_vertices = np.array([0, 0, 0, 1, 0, 0], dtype=np.float32) # Color: Magenta
        # Create a circle
        num_points = 400
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = 2*math.cos(angle)
            y = 2*math.sin(angle)
            self.point_vertices = np.append(self.point_vertices, [x, y, 0, 0, 1, 0])
        self.point_vertices = np.array(self.point_vertices, dtype=np.float32)

        glBindVertexArray(self.point_VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.point_VBO)
        glBufferData(
            GL_ARRAY_BUFFER,
            self.point_vertices.nbytes,
            self.point_vertices,
            GL_STATIC_DRAW,
        )
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

        # Collada model for the drone
        self.drone_model = ColladaModel("renderer/models/cf2_assembly.dae")
        self.prop_1 = ColladaModel("renderer/models/ccw_prop.dae")
        self.prop_2 = ColladaModel("renderer/models/cw_prop.dae")
        self.prop_3 = ColladaModel("renderer/models/ccw_prop.dae")
        self.prop_4 = ColladaModel("renderer/models/cw_prop.dae")
        
        # Set OpenGL state
        glEnable(GL_DEPTH_TEST)
        glLineWidth(2.0)
        glClearColor(0.2, 0.2, 0.4, 1.0)  # Dark gray background
        glEnable(GL_MULTISAMPLE)
        glEnable (GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glPointSize(5.0)

    def update_fps(self):
        current_time = time.time()
        self.frame_count += 1

        if current_time - self.last_time >= self.fps_update_interval:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time

    def _mouse_callback(self, window, xpos, ypos):
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            self.camera.process_mouse(xpos, ypos)
        else:
            self.camera.first_mouse = True

    def _scroll_callback(self, window, xoffset, yoffset):
        self.camera.radius -= yoffset * 0.5
        self.camera.radius = np.clip(self.camera.radius, 1.0, 200.0)
        self.camera.update_position()

    def _window_resize_callback(self, window, width, height):
        self.width = width
        self.height = height
        glViewport(0, 0, width, height)

    def render(self, position, rotation_matrix, time, debug_info=None):
        self.update_fps()
        self.camera.target = position
        self.camera.update_position()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.simple_shader)

        view_matrix = self.camera.get_view_matrix()
        projection_matrix = self.camera.get_perspective_matrix(
            45.0, self.width / self.height, 0.1, 10000.0)

        # Upload view and projection matrices (common for both grid and axis)
        glUniformMatrix4fv(glGetUniformLocation(
            self.simple_shader, "view"), 1, GL_TRUE, view_matrix)
        glUniformMatrix4fv(glGetUniformLocation(
            self.simple_shader, "projection"), 1, GL_TRUE, projection_matrix)

        # Draw grid with identity model matrix (fixed in place)
        grid_model_matrix = np.eye(4, dtype=np.float32)
        glUniformMatrix4fv(glGetUniformLocation(
            self.simple_shader, "model"), 1, GL_TRUE, grid_model_matrix)
        glBindVertexArray(self.grid_VAO)
        glDrawArrays(GL_LINES, 0, len(self.grid_vertices) // 6)

        # Draw point
        # glBindVertexArray(self.point_VAO)
        # glDrawArrays(GL_POINTS, 0, len(self.point_vertices) // 6)

        # Draw axis with transformed model matrix
        axis_model_matrix = np.eye(4, dtype=np.float32)
        axis_model_matrix[:3, :3] = rotation_matrix
        axis_model_matrix[:3, 3] = position
        glUniformMatrix4fv(glGetUniformLocation(
            self.simple_shader, "model"), 1, GL_TRUE, axis_model_matrix)
        glBindVertexArray(self.axis_VAO)
        glDrawArrays(GL_LINES, 0, len(self.axis_vertices) // 6)

        # Draw drone model
        self.drone_model.render(axis_model_matrix, view_matrix, projection_matrix, self.camera.position)
        # Draw propellers
        propeller_positions = np.array([
            [ 0.31, -0.31, 0.06], # Motor 1 
            [-0.31, -0.31, 0.06], # Motor 2
            [-0.31,  0.31, 0.06], # Motor 3
            [ 0.31,  0.31, 0.06], # Motor 4
        ]) @ rotation_matrix.T  # Rotate positions
        angle = time * 100  # Rotate speed
        propeller_rotations = np.array([
            [[np.cos(angle), np.sin(angle), 0], [-np.sin(angle), np.cos(angle), 0], [0, 0, 1]],  # CCW
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]],  # CW
            [[np.cos(angle), np.sin(angle), 0], [-np.sin(angle), np.cos(angle), 0], [0, 0, 1]],  # CCW
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]],  # CW
        ]) @ rotation_matrix.T  # Rotate orientations
        propeller_models = [self.prop_1, self.prop_2, self.prop_3, self.prop_4]
        for i, (pos, rot, model) in enumerate(zip(propeller_positions, propeller_rotations, propeller_models)):
            prop_model_matrix = np.eye(4, dtype=np.float32)
            prop_model_matrix[:3, :3] = rot.T
            prop_model_matrix[:3, 3] = position + np.array(pos)
            model.render(prop_model_matrix, view_matrix, projection_matrix, self.camera.position)
            
        # Draw FPS text
        fps_text = f"FPS: {self.fps:.1f}"
        self.render_text(fps_text, -self.width / 2 + 10, self.height / 2 - 30)

        # Draw time text
        time_text = f"Time: {time:.2f}s"
        self.render_text(time_text, -self.width / 2 + 10, self.height / 2 - 60)
        
        # Print variable dict debug info on right side
        np.printoptions(precision=2, suppress=True)
        if debug_info:
            y_offset = 30
            for key, value in debug_info.items():
                info_text = f"{key}: {value}"
                self.render_text(info_text, -self.width / 2 + 10, -self.height / 2 + y_offset)
                y_offset += 30
        
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def render_text(self, text, x, y, color=(1, 1, 1)):
        # Render text to surface
        text_surface = self.font.render(text, True, (255, 255, 255))
        text_data = pygame.image.tostring(text_surface, 'RGBA', True)

        # Update texture
        glBindTexture(GL_TEXTURE_2D, self.fps_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_surface.get_width(), text_surface.get_height(),
                     0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        # Prepare matrices
        ortho = np.array([
            [2/self.width, 0, 0, 0],
            [0, -2/self.height, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Prepare rendering
        glUseProgram(self.text_shader)
        glUniformMatrix4fv(glGetUniformLocation(
            self.text_shader, "projection"), 1, GL_FALSE, ortho)
        glUniform3f(glGetUniformLocation(
            self.text_shader, "textColor"), *color)

        glActiveTexture(GL_TEXTURE0)
        glBindVertexArray(self.text_VAO)

        # Calculate vertices
        w, h = text_surface.get_width(), text_surface.get_height()
        vertices = np.array([
            x, y + h,    0, 0,
            x, y,        0, 1,
            x + w, y,    1, 1,
            x, y + h,    0, 0,
            x + w, y,    1, 1,
            x + w, y + h, 1, 0,
        ], dtype=np.float32)

        glBindBuffer(GL_ARRAY_BUFFER, self.text_VBO)
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glDrawArrays(GL_TRIANGLES, 0, 6)

        glDisable(GL_BLEND)
        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)

    def cleanup(self):
        glDeleteVertexArrays(1, [self.axis_VAO])
        glDeleteVertexArrays(1, [self.grid_VAO])
        glDeleteBuffers(1, [self.axis_VBO])
        glDeleteBuffers(1, [self.grid_VBO])
        glDeleteProgram(self.simple_shader)
        glDeleteTextures([self.fps_texture])
        glfw.terminate()
        pygame.quit()

    def should_close(self):
        return glfw.window_should_close(self.window)


if __name__ == "__main__":
    renderer = DroneRenderer()

    while not renderer.should_close():
        # Example: rotate the axis around Z
        angle = glfw.get_time()
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])
        position = np.array([0, 0, 0])
        renderer.render(position, rotation, glfw.get_time())

    renderer.cleanup()
