#version 150
in vec3 position;
in vec3 normal;
out vec3 eyeNormal;
out vec3 eyePosition;
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
void main() {
	eyeNormal = mat3( view * model ) * normal;
	eyePosition = vec3( view * model * vec4( position, 1.0f ) );
	gl_Position = proj * view * model * vec4( position, 1.0f );
}