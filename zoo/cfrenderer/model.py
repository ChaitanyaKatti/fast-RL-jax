import numpy as np
import collada
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

def load_dae_model(file_path):
    # Load the DAE file
    mesh = collada.Collada(file_path)
    
    vertex_data = []
    
    for scene in mesh.scenes:
        for node in scene.nodes:
            transform_matrix = np.eye(4)
            if isinstance(node, collada.scene.Node):
                for transform in node.transforms:
                    if isinstance(transform, collada.scene.MatrixTransform):
                        transform_matrix = transform.matrix  # Store the transformation matrix
                        break
                # Check if the node has instance_geometry
                for instance in node.children:
                    if isinstance(instance, collada.scene.GeometryNode):
                        for primitive in instance.geometry.primitives:
                            positions = primitive.vertex[primitive.vertex_index].reshape(-1, 3)  # Flatten to (N, 3)
                            transformed_positions = np.ones((positions.shape[0], 4))  # Homogeneous coordinates
                            transformed_positions[:, :3] = positions  # Copy x, y, z
                            transformed_positions = (transform_matrix @ transformed_positions.T).T
                            positions = transformed_positions[:, :3].flatten()  # Copy back x, y, z
                            if primitive.normal is not None:
                                normals = primitive.normal[primitive.normal_index].flatten()
                            else:
                                normals = np.zeros_like(positions)
                                for i in range(0, len(positions), 9):
                                    v1 = positions[i:i+3]
                                    v2 = positions[i+3:i+6]
                                    v3 = positions[i+6:i+9]
                                    normal = np.cross(v2 - v1, v3 - v1)
                                    normal = normal / np.linalg.norm(normal)
                                    normals[i:i+3] = normal
                                    normals[i+3:i+6] = normal
                                    normals[i+6:i+9] = normal
                            if primitive.material and hasattr(mesh.materials.get(primitive.material), 'effect'):
                                diffuse = mesh.materials.get(primitive.material).effect.diffuse
                                if isinstance(diffuse, tuple):
                                    colors = np.tile(diffuse[:3], len(positions) // 3)
                                else:
                                    colors = np.ones(len(positions)) * 0.8
                            else:
                                colors = np.ones(len(positions)) * 0.8
                            for i in range(0, len(positions), 3):
                                vertex_data.extend([
                                    positions[i], positions[i+1], positions[i+2],
                                    normals[i], normals[i+1], normals[i+2],
                                    colors[i], colors[i+1], colors[i+2]
                                ])
    print(f"Loaded model with {len(vertex_data) // 9} vertices")

    return np.array(vertex_data, dtype=np.float32)

def create_model_vao(vertex_data):
    # Create and bind VAO
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    
    # Create and bind VBO
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    
    # Upload data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
    
    stride = 9 * 4  # 9 floats per vertex (3 position + 3 normal + 3 color) * 4 bytes per float
    
    # Position attribute
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_TRUE, stride, ctypes.c_void_p(0))
    
    # Normal attribute
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_TRUE, stride, ctypes.c_void_p(12))  # 3 * 4 bytes offset
    
    # Color attribute
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_TRUE, stride, ctypes.c_void_p(24))  # 6 * 4 bytes offset
    
    # Unbind VAO and VBO
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    
    return vao, vbo, len(vertex_data) // 9  # Return vertex count

# Usage example:
class ColladaModel:
    def __init__(self, file_path):
        # Load model data
        self.vertex_data = load_dae_model(file_path)
        
        # Create VAO and VBO
        self.vao, self.vbo, self.vertex_count = create_model_vao(self.vertex_data)
        
        # Shader for rendering the model
        self.shader = compileProgram(
            compileShader("""
            #version 330 core
            layout (location = 0) in vec3 position;
            layout (location = 1) in vec3 normal;
            layout (location = 2) in vec3 color;
            
            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;
            
            out vec3 FragPos;
            out vec3 Normal;
            out vec3 Color;
            
            void main() {
                FragPos = vec3(model * vec4(10.0*position, 1.0));
                Normal = mat3(transpose(inverse(model))) * normal;
                Color = color;
                gl_Position = projection * view * model * vec4(10.0*position, 1.0);
            }
        """, GL_VERTEX_SHADER),
        compileShader("""
            #version 330 core
            in vec3 FragPos;
            in vec3 Normal;
            in vec3 Color;
            
            out vec4 FragColor;
            
            uniform vec3 viewPos;
            uniform vec3 lightDir;
            
            void main() {
                
                // Ambient
                float ambientStrength = 0.2;
                vec3 ambient = ambientStrength * Color;
                
                // Diffuse
                vec3 norm = normalize(Normal);
                float diff = max(dot(norm, lightDir), 0.0);
                vec3 diffuse = diff * Color;
                
                // Specular
                float specularStrength = 0.5;
                vec3 viewDir = normalize(viewPos - FragPos);
                vec3 reflectDir = reflect(-lightDir, norm);
                float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
                vec3 specular = specularStrength * spec * vec3(1.0);
                
                vec3 result = ambient + diffuse + specular;
                FragColor = vec4(result, 1.0);
            }
        """, GL_FRAGMENT_SHADER)
        )
    
    def render(self, model_matrix, view_matrix, projection_matrix, view_pos, light_dir=np.array([1.0, 1.0, 1.0])):
        glUseProgram(self.shader)
        
        # Set uniforms
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "model"), 1, GL_TRUE, model_matrix)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_TRUE, view_matrix)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "projection"), 1, GL_TRUE, projection_matrix)
        glUniform3fv(glGetUniformLocation(self.shader, "viewPos"), 1, view_pos)
        glUniform3fv(glGetUniformLocation(self.shader, "lightDir"), 1, light_dir/np.linalg.norm(light_dir))

        # Draw model
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
        glBindVertexArray(0)