#include "Sphere.h"
#include <cmath>

namespace MyRayTracer {

	// Get a sphere with the specified center, radius, and color
	sphere getSphere(point center, float radius, color diffuseColor,
		color specularColor, float shininess, float kr, float kt, float n) {
		sphere s;
		s.center = center;
		s.radius = radius;
		s.diffuseColor = diffuseColor;
		s.specularColor = specularColor;
		s.shininess = shininess;
		s.kr = kr;
		s.kt = kt;
		s.n = n;
		return s;
	}
}