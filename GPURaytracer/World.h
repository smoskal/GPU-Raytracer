#ifndef __WORLD_H__
#define __WORLD_H__

#include "RGBColor.h"
#include "Space3D.h"
#include "ShadeRecord.h"
#include "Plane.h"
#include "Sphere.h"
#include "Triangle.h"
#include "Light.h"
#include <string>
#include <iostream>
#include <fstream>
#include <cuda_runtime_api.h>

namespace MyRayTracer {
	
	// Index of refraction of air
	const float N_AIR = 1.0f;

	// The world in which the scene exists
	struct world_t {
		
		// The default color to use if there are no intersections
		color backgroundColor;
		
		// The number of planes in the world
		int numPlanes;
		
		// The number of spheres in the world
		int numSpheres;
		
		// The number of triangles in the world
		int numTriangles;
		
		// The array of planes
		plane *planes;
		
		// The size of the planes array
		int planesSize;
		
		// The array of spheres
		sphere *spheres;
		
		// The size of the spheres array
		int spheresSize;
		
		// The array of triangles
		triangle *triangles;
		
		// The size of the triangles array
		int trianglesSize;
		
		// The light in the world
		pointLight light;
	};
	
	typedef world_t world;

	// Constructor
	world getWorld(color backgroundColor);
	
	// Destructor
	void deleteWorld(world *w);
	
	// Add a plane
	void addPlane(world *w, plane p);
	
	// Add a sphere
	void addSphere(world *w, sphere s);

	// Add a triangle
	void addTriangle(world *w, triangle t);

	// Add a cube defined by two opposite corners
	void addCube(world *w, point a, point h, color diffuseColor,
		color specularColor, float shininess, float kr);

	// Check for hits
	__host__ __device__ record checkForHits(int numPlanes, plane *planes, 
		int numSpheres, sphere *spheres, int numTriangles, triangle *triangles, 
		ray r, color backgroundColor, pointLight light);
	
	// Build the world
	world build(std::string filepath);
}

#endif