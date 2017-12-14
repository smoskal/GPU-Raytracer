#include "Light.h"

namespace MyRayTracer {

	// Get a pointLight with the specified information
	pointLight getPointLight(point position, color diffuseColor, 
		color specularColor) {
			pointLight light;
			light.position = position;
			light.diffuseColor = diffuseColor;
			light.specularColor = specularColor;
			return light;
	}
}