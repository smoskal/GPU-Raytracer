#include "Space3D.h"
#include <cmath>
#include <cuda_runtime_api.h>

namespace MyRayTracer {

	// Get point (0,0,0) (default constructor)
	point getPoint() {
		point p;
		p.x = 0;
		p.y = 0;
		p.z = 0;
		return p;
	}

	// Get vector (0,0,0)^T (default constructor)
	vector getVector() {
		vector v;
		v.x = 0;
		v.y = 0;
		v.z = 0;
		return v;
	}

	// Get ray originating at (0,0,0) with direction (0,0,0)^T (default constructor)
	ray getRay() {
		point p;
		p.x = 0;
		p.y = 0;
		p.z = 0;
		vector v;
		v.x = 0;
		v.y = 0;
		v.z = 0;
		ray r;
		r.origin = p;
		r.direction = v;
		return r;
	}

	// Get point (a,a,a)
	point getPoint(const float a) {
		point p;
		p.x = a;
		p.y = a;
		p.z = a;
		return p;
	}

	// Get vector (a,a,a)^T
	vector getVector(const float a) {
		vector v;
		v.x = a;
		v.y = a;
		v.z = a;
		return v;
	}

	// Get the resulting point from adding vector v to point a
	point addVector(const point a, const vector v) {
		point p;
		p.x = a.x + v.x;
		p.y = a.y + v.y;
		p.z = a.z + v.z;
		return p;
	}

	// Get the resulting point from subtracting vector v from point a
	point subtractVector(const point a, const vector v) {
		point p;
		p.x = a.x - v.x;
		p.y = a.y - v.y;
		p.z = a.z - v.z;
		return p;
	}

	// Get the distance between two points
	float getDistance(const point a, const point b) {
		return (float)(sqrt(pow((float)(b.x - a.x), 2.0f) + 
			pow((float)(b.y - a.y), 2.0f) + pow(((float)b.z - a.z), 2.0f)));
	}
}