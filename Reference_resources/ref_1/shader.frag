#version 450

layout(location = 0) in vec3 frag_color;
layout(location = 0) out vec3 outColor;

void main() {
    outColor = vec3(0.5, 0.8, 0.5);
    outColor = frag_color;
}