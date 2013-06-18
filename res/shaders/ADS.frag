#version 150
in vec3 eyePosition;
in vec3 eyeNormal;
out vec4 outColor;
uniform mat4 view;
uniform vec4 lightPosition;
uniform vec3 diffuseColor;
void main() {
	vec4 eyeLightPosition = view * lightPosition;
	vec3 normal = normalize( eyeNormal );
	vec3 toLightDir = normalize( eyeLightPosition.xyz - eyeLightPosition.w * eyePosition );
	vec3 lightIntensity = vec3( 1.0f, 1.0f, 1.0f );
	vec3 ambientColor = 0.2f * diffuseColor;
	vec3 ambient = ambientColor;
	vec3 diffuse = diffuseColor * max ( dot( normal, toLightDir ), 0.0f );
	outColor = vec4( lightIntensity * ( ambient + diffuse ), 1.0f );
}