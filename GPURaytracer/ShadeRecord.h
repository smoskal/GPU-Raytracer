#ifndef __SHADERECORD_H__
#define __SHADERECORD_H__

#include "Space3D.h"
#include "RGBColor.h"

namespace MyRayTracer {

	// Record of information about an intersection of a ray and an object
	struct shade_rec_t {
	
		// Whether there was an intersection (if false, ignore other values)
		bool intersection;
		
		// The point at which the ray intersected with an object
		point intersection_point;
		
		// The normal at the intersection point
		vector normal;
		
		// The intersecting ray
		ray ray;
		
		// The overall RGB color at the intersection point as seen by the camera
		color rgb;
		
		// The diffuse color of the intersected object
		color diffuseColor;
		
		// The specular color of the intersected object
		color specularColor;
		
		// The shininess exponent of the intersected object
		float shininess;
		
		// The reflection coefficient of the intersected object
		float kr;
		
		// The refraction coefficient of the intersected object
		float kt;
		
		// The index of refraction of the intersected object
		float n;
	};

	typedef shade_rec_t record;
}

#endif