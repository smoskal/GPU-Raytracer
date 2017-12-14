#ifndef __SPHERE_H__
#define __SPHERE_H__

#include "Space3D.h"
#include "RGBColor.h"
#include "ShadeRecord.h"

namespace MyRayTracer {

	// Struct used to represent a sphere
	struct sphere_t {
		
		// The center of the sphere
		point center;
		
		// The radius of the sphere
		float radius;
		
		// The diffuse color of the sphere
		color diffuseColor;
		
		// The specular color of the sphere
		color specularColor;
		
		// The shininess exponent of the sphere
		float shininess;
		
		// The reflection coefficient of the sphere
		float kr;
		
		// The refraction coefficient of the sphere
		float kt;
		
		// The index of refraction of the sphere
		float n;
	};
	
	typedef sphere_t sphere;
	
	// Get a sphere with the specified center, radius, and color
	sphere getSphere(point center, float radius, color diffuseColor,
		color specularColor, float shininess, float kr, float kt, float n);
	
	// Determine whether this sphere was hit by the specified ray
	__host__ __device__ bool hit(const ray *ray, float *t, record *sr, const sphere *s);
}

#endif