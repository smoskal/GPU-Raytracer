#include "Plane.h"

namespace MyRayTracer {

	// Get a plane with the specified information
	plane getPlane(point position, vector normal, color diffuseColor1, 
		color diffuseColor2, float checkerSize, color specularColor, float shininess, float kr) {
		plane p;
		p.position = position;
		p.normal = normal;
		p.diffuseColor1 = diffuseColor1;
		p.diffuseColor2 = diffuseColor2;
		p.checkerSize = checkerSize;
		p.specularColor = specularColor;
		p.shininess	= shininess;
		p.kr = kr;
		return p;
	}
}