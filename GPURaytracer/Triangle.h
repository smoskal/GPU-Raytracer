#ifndef __TRIANGLE_H__
#define __TRIANGLE_H__

#include "Space3D.h"
#include "RGBColor.h"
#include "ShadeRecord.h"

namespace MyRayTracer {

	// Struct used to represent a triangle
	struct triangle_t {
		
		// The vertices of the triangle
		point a, b, c;
		
		// The normal to the surface of the triangle
		vector normal;
		
		// The diffuse color of the triangle
		color diffuseColor;
		
		// The specular color of the triangle
		color specularColor;
		
		// The shininess exponent of the triangle
		float shininess;
		
		// The reflection coefficient of the triangle
		float kr;
	};

	typedef triangle_t triangle;

	// Get a triangle with the specified points and color
	triangle getTriangle(point a, point b, point c,
		color diffuseColor, color specularColor, float shininess, float kr);

	// Determine whether this triangle was hit by the specified ray
	__host__ __device__ bool hit(const ray *ray, float *t, record *sr, const triangle *tri);
}

#endif