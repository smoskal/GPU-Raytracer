#ifndef __LIGHT_H__
#define __LIGHT_H__

#include "Space3D.h"
#include "RGBColor.h"
#include "ShadeRecord.h"
#include <cmath>

namespace MyRayTracer {

	// Struct used to represent a point light
	struct point_light_t {
		point position;
		color diffuseColor;
		color specularColor;
	};

	typedef point_light_t pointLight;

	// Get a pointLight with the specified information
	pointLight getPointLight(point position, color diffuseColor, 
		color specularColor);

	// Determine the color of a pixel as a result of illumination
	// Uses Blinn-Phong Illumination Model (without ambient component)
	__host__ __device__ color illuminate(pointLight light, record sr);
}

#endif