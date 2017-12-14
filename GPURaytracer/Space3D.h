#ifndef __SPACE3D_H__
#define __SPACE3D_H__

#include <cuda_runtime_api.h>

namespace MyRayTracer {

	// struct used to define a point in 3D space
	struct point_3d_t {
		float x, y, z;
	};

	// Point in 3D space
	typedef point_3d_t point;

	// Vector in 3D space
	typedef point_3d_t vector;
	
	// struct used to define a ray in 3D space
	struct ray_t {
		point origin;
		vector direction;
	};
	
	// Ray in 3D space
	typedef ray_t ray;

	// Get point (0,0,0) (default constructor)
	point getPoint();

	// Get vector (0,0,0)^T (default constructor)
	vector getVector();
	
	// Get ray originating at (0,0,0) with direction (0,0,0)^T (default constructor)
	ray getRay();

	// Get point (a,a,a)
	point getPoint(const float a);

	// Get vector (a,a,a)^T
	vector getVector(const float a);

	// Get point (x,y,z)
	__host__ __device__ point getPoint(const float x, const float y, const float z);

	// Get vector (x,y,z)^T
	__host__ __device__ vector getVector(const float x, const float y, const float z);

	// Get the vector joining points a and b
	__host__ __device__ vector getVector(const point a, const point b);
	
	// Get a ray with the specified origin and direction
	__host__ __device__ ray getRay(point origin, vector direction);

	// Get the resulting point from adding vector v to point a
	point addVector(const point a, const vector v);

	// Get the resulting point from subtracting vector v from point a
	point subtractVector(const point a, const vector v);

	// Get the distance between two points
	float getDistance(const point a, const point b);
	
	// Get the dot product of two vectors
	__host__ __device__ float getDotProduct(const vector v1, const vector v2);

	// Get the cross product of two vectors
	__host__ __device__ vector getCrossProduct(const vector v1, const vector v2);

	// Get the length of a vector
	__host__ __device__ float getLength(vector v);

	// Convert a vector to a unit vector
	__host__ __device__ vector normalize(vector v);
}

#endif