from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader


def simple_shader():
    return compileProgram(
        compileShader("""
            #version 330 core
            layout (location = 0) in vec3 position;
            layout (location = 1) in vec3 color;
            out vec3 fragColor;
            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;
            void main()
            {
                fragColor = color;
                gl_Position = projection * view * model * vec4(position, 1.0);
            }
            """, GL_VERTEX_SHADER),
        compileShader("""
            #version 330 core
            in vec3 fragColor;
            out vec4 color;
            void main()
            {
                color = vec4(fragColor, 1.0);
            }
            """, GL_FRAGMENT_SHADER)
    )


def text_shader():
    return compileProgram(
        compileShader("""
            #version 330 core
            layout (location = 0) in vec4 vertex;
            out vec2 TexCoords;
            uniform mat4 projection;

            void main()
            {
                gl_Position = projection * vec4(vertex.xy, 0.0, 1.0);
                TexCoords = vertex.zw;
            }
            """, GL_VERTEX_SHADER),
        compileShader("""
            #version 330 core
            in vec2 TexCoords;
            out vec4 color;
            uniform sampler2D text;
            uniform vec3 textColor;

            void main()
            {
                vec4 sampled = texture(text, TexCoords);
                //color = vec4(TexCoords, 1.0, 1.0);
                color = sampled * vec4(textColor, 1.0);
            }
            """, GL_FRAGMENT_SHADER)
    )
