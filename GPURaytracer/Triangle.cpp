#include "Triangle.h"

namespace MyRayTracer {

	// Get a triangle with the specified points and color
	triangle getTriangle(point a, point b, point c,
		color diffuseColor, color specularColor, float shininess,
		float kr) {
		triangle t;
		t.a = a;
		t.b = b;
		t.c = c;
		t.diffuseColor = diffuseColor;
		t.specularColor = specularColor;
		t.shininess = shininess;
		t.normal = normalize(getCrossProduct(getVector(a, b), 
			getVector(a, c)));
		t.kr = kr;
		return t;
	}
}