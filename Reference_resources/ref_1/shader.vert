#version 450

layout(location = 0) in vec3 inPosition;

layout (binding = 0) uniform RGB {
    vec3 rgb;
} rgbs;

layout(location = 0) out vec3 frag_color;

void main() {
    gl_Position = vec4(inPosition, 1.0);
    frag_color = rgbs.rgb;
}