#ifndef __PLANE_H__
#define __PLANE_H__

#include "Space3D.h"
#include "RGBColor.h"
#include "ShadeRecord.h"

namespace MyRayTracer {

	// Struct used to represent a 3D plane
	struct plane_t {
		
		// A point on the plane
		point position;
		
		// The normal at the above point
		vector normal;
		
		// The first diffuse color of the plane (make both the same for non-checkered plane)
		color diffuseColor1;
		
		// The second diffuse color of the plane (make both the same for non-checkered plane)
		color diffuseColor2;
		
		// The size of the squares in a checkered plane
		float checkerSize;
		
		// The specular color of the plane
		color specularColor;
		
		// The shininess exponent of the plane
		float shininess;
		
		// The reflection coefficient of the plane
		float kr;
	};

	typedef plane_t plane;

	// Get a plane with the specified information
	plane getPlane(point position, vector normal, color diffuseColor1, 
		color diffuseColor2, float checkerSize, color specularColor, float shininess, float kr);
	
	// Determine whether this plane was hit by the specified ray
	__host__ __device__ bool hit(const ray *ray, float *t, record *sr, const plane *p);

}

#endif