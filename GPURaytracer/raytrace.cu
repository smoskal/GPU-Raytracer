#include "Camera.h"
#include "World.h"
#include "Space3D.h"
#include "Plane.h"
#include "Sphere.h"
#include "Triangle.h"
#include "Light.h"
#include <cfloat>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

namespace MyRayTracer {

	// Check for a CUDA error
	bool checkForError(cudaError_t error) {

		// If there was no error, return true
		if (error == cudaSuccess) {
			return true;
		} 

		// If there was an error, print the description and return false
		else {
			std::cerr << cudaGetErrorString(error) << std::endl;
			return false;
		}
	}

	// Determine whether this plane was hit by the specified ray
	__host__ __device__ bool hit(const ray *ray, float *t, record *sr, const plane *p) {
		sr->intersection = false;
		float w_denom = getDotProduct(p->normal, ray->direction);
		if (w_denom == 0) {
		
			// If the ray and normal are orthogonal, there is no intersection
			return false;
		}
		float w_num = -1 * (getDotProduct(p->normal, getVector(ray->origin.x, 
			ray->origin.y, ray->origin.z)) - getDotProduct(p->normal,
			getVector(p->position.x, p->position.y, p->position.z)));
		float w = w_num / w_denom;
		if (w < 0) {
		
			// If the intersection occurs behind the camera, ignore it
			return false;
		} else {
			*t = w;
			point intersection = getPoint(ray->origin.x + 
				(ray->direction.x * *t), ray->origin.y + 
				(ray->direction.y * *t), ray->origin.z + 
				(ray->direction.z * *t));
			sr->intersection = true;
			sr->intersection_point = intersection;
			sr->normal = normalize(getVector(intersection, ray->origin));
			sr->ray = *ray;
			sr->kr = p->kr;

			// Calculate the correct color
			float x = intersection.x;
			float z = intersection.z;
			int ix = (int)floor(x / p->checkerSize);
			int iz = (int)floor(z / p->checkerSize);
			sr->diffuseColor = (((ix + iz) % 2) == 0) ? 
				p->diffuseColor1 : p->diffuseColor2;
			sr->specularColor = p->specularColor;
			sr->shininess = p->shininess;
			return true;
		}
	}

	// Determine whether this sphere was hit by the specified ray
	__host__ __device__ bool hit(const ray *ray, float *t, record *sr, const sphere *s) {

		const float EPSILON = 0.01f;

		sr->intersection = false;
		float a = pow(ray->direction.x, 2) + pow(ray->direction.y, 2)
			+ pow(ray->direction.z, 2);
		float b = 2 * ((ray->direction.x * (ray->origin.x - s->center.x))
			+ (ray->direction.y * (ray->origin.y - s->center.y))
			+ (ray->direction.z * (ray->origin.z - s->center.z)));
		float c = pow(ray->origin.x - s->center.x, 2)
			+ pow(ray->origin.y - s->center.y, 2)
			+ pow(ray->origin.z - s->center.z, 2) - pow(s->radius, 2);
		float discriminant = pow(b, 2) - (4 * a * c);

		if (discriminant < 0) {
		
			// There is no intersection
			return false;
		} else if ((discriminant == 0) && ((-1 * b / (2 * a)) >= EPSILON)) {
		
			// The ray is tangent to the sphere, intersecting at a single point
			*t = -1 * b / (2 * a);
			point intersection = getPoint(ray->origin.x + 
				(ray->direction.x * *t), ray->origin.y + 
				(ray->direction.y * *t), ray->origin.z + 
				(ray->direction.z * *t));
			sr->intersection = true;
			sr->normal = normalize(getVector(s->center, intersection));
			sr->intersection_point = intersection;
			sr->diffuseColor = s->diffuseColor;
			sr->specularColor = s->specularColor;
			sr->shininess = s->shininess;
			sr->ray = *ray;
			sr->kr = s->kr;
			sr->kt = s->kt;
			sr->n = s->n;
			return true;
		} else {
		
			// The ray passes through the sphere, intersecting in two places
			float root1 = ((-1 * b) + sqrt(discriminant)) / (2 * a);
			float root2 = ((-1 * b) - sqrt(discriminant)) / (2 * a);

			if ((root1 < 0) && (root2 < 0)) {
			
				// Both intersections occur behind the camera
				return false;
			} else if (root2 >= EPSILON) {
			
				// Take the closer intersection if possible
				*t = root2;
				point intersection = getPoint(ray->origin.x + 
					(ray->direction.x * *t), ray->origin.y + 
					(ray->direction.y * *t), ray->origin.z + 
					(ray->direction.z * *t));
				sr->intersection = true;
				sr->normal = normalize(getVector(s->center, intersection));
				sr->intersection_point = intersection;
				sr->diffuseColor = s->diffuseColor;
				sr->specularColor = s->specularColor;
				sr->shininess = s->shininess;
				sr->ray = *ray;
				sr->kr = s->kr;
				sr->kt = s->kt;
				sr->n = s->n;
				return true;
			} else if (root1 >= EPSILON) {
			
				// Take the farther intersection when the closer one is behind the camera
				*t = root1;
				point intersection = getPoint(ray->origin.x + 
					(ray->direction.x * *t), ray->origin.y + 
					(ray->direction.y * *t), ray->origin.z + 
					(ray->direction.z * *t));
				sr->intersection = true;
				sr->normal = normalize(getVector(s->center, intersection));
				sr->intersection_point = intersection;
				sr->diffuseColor = s->diffuseColor;
				sr->specularColor = s->specularColor;
				sr->shininess = s->shininess;
				sr->ray = *ray;
				sr->kr = s->kr;
				sr->kt = s->kt;
				sr->n = s->n;
				return true;
			} else {
				return false;
			}
		}
	}

	// Determine whether this triangle was hit by the specified ray using barycentric coordinates
	__host__ __device__ bool hit(const ray *ray, float *t, record *sr, const triangle *tri) {
		vector e1 = getVector(tri->a, tri->b);
		vector e2 = getVector(tri->a, tri->c);
		vector vT = getVector(tri->a, ray->origin);
		vector vP = getCrossProduct(ray->direction, e2);
		vector vQ = getCrossProduct(vT, e1);

		if (getDotProduct(vP, e1) == 0.0f) {
			return false;
		}

		float factor = 1.0f / getDotProduct(vP, e1);

		float _t = getDotProduct(vQ, e2) * factor;
		float u = getDotProduct(vP, vT) * factor;
		float v = getDotProduct(vQ, ray->direction) * factor;

		if ((_t < 0.0f) || (u < 0.0f) || (v < 0.0f) || ((u + v) > 1.0f)) {
			return false;
		}

		*t = _t;

		sr->diffuseColor = tri->diffuseColor;
		sr->specularColor = tri->specularColor;
		sr->shininess = tri->shininess;
		sr->ray = *ray;
		sr->intersection = true;

		float hitX = ray->origin.x + (*t * ray->direction.x);
		float hitY = ray->origin.y + (*t * ray->direction.y);
		float hitZ = ray->origin.z + (*t * ray->direction.z);
		sr->intersection_point = getPoint(hitX, hitY, hitZ);
		sr->normal = getCrossProduct(e1, e2);
		sr->kr = tri->kr;

		return true;
	}

	// Check for hits
	__host__ __device__ record checkForHits(int numPlanes, plane *planes, 
		int numSpheres, sphere *spheres, int numTriangles, triangle *triangles, 
		ray r, color backgroundColor, pointLight light) {

			// Max reflection depth
			const int MAX_DEPTH = 20;

			// Min intersection distance
			const float EPSILON = 0.01f;

			const float N_AIR = 1.0f;

			// The record of the closest hit
			record hitRecord;

			// The current ray
			ray currRay = r;

			// Reflection depth
			int depth = 1;

			// Whether to process reflection
			bool reflect = true;

			// Whether to process refraction
			bool refract = true;

			// the cumulative reflection/refraction coefficient
			float kr = 1.0f;

			// Loop for reflection
			while ((depth <= MAX_DEPTH) && (reflect || refract)) {

				// The distance of the closest hit
				float minDistance = FLT_MAX;

				// Temp record for this depth
				record rec;
				rec.intersection = false;
				rec.rgb = backgroundColor;

				// Check for hits on planes
				for (int i = 0; i < numPlanes; i++) {
					record sr;
					float distance = FLT_MAX;
					if (hit(&currRay, &distance, &sr, &(planes[i]))) {
						if ((distance < minDistance) && (distance >= EPSILON)) {
							minDistance = distance;
							rec = sr;
							rec.rgb = illuminate(light, rec);
						}
					}
				}

				// Check for hits on spheres
				for (int i = 0; i < numSpheres; i++) {
					record sr;
					float distance = FLT_MAX;
					if (hit(&currRay, &distance, &sr, &(spheres[i]))) {
						if ((distance < minDistance)) {
							minDistance = distance;
							rec = sr;
							rec.rgb = illuminate(light, rec);
						}
					}
				}

				// Check for hits on triangles
				for (int i = 0; i < numTriangles; i++) {
					record sr;
					float distance = FLT_MAX;
					if (hit(&currRay, &distance, &sr, &(triangles[i]))) {
						if ((distance < minDistance) && (distance >= EPSILON)) {
							minDistance = distance;
							rec = sr;
							rec.rgb = illuminate(light, rec);
						}
					}
				}

				if (depth == 1) {

					// Set hitRecord on the first hit
					hitRecord = rec;
				} else {

					// Accumulate color
					hitRecord.rgb.r += ((int)floor(kr * rec.rgb.r + 0.5f));
					hitRecord.rgb.g += ((int)floor(kr * rec.rgb.g + 0.5f));
					hitRecord.rgb.b += ((int)floor(kr * rec.rgb.b + 0.5f));
				}

				// Default assumption is no reflection or refraction
				reflect = false;
				refract = false;

				// Handle reflection or refraction
				if (rec.intersection && (rec.kr > 0.0f) && (depth < MAX_DEPTH)) {
					reflect = true;
					depth++;

					// Adjust reflection coefficient
					kr *= rec.kr;

					// Get vector to source
					vector toSource = normalize(getVector(rec.intersection_point, 
						rec.ray.origin));

					// Get reflection vector
					float rX = toSource.x + 2 * ((getDotProduct(toSource, rec.normal))
						/ pow(getDotProduct(rec.normal, rec.normal), 2.0f)) * rec.normal.x;
					float rY = toSource.y + 2 * ((getDotProduct(toSource, rec.normal))
						/ pow(getDotProduct(rec.normal, rec.normal), 2.0f)) * rec.normal.y;
					float rZ = toSource.z + 2 * ((getDotProduct(toSource, rec.normal))
						/ pow(getDotProduct(rec.normal, rec.normal), 2.0f)) * rec.normal.z;
					vector reflect = normalize(getVector(rX, rY, rZ));

					// Spawn reflection ray
					currRay = getRay(rec.intersection_point, reflect);
				} else if (rec.intersection && (rec.kt > 0.0f) && (depth < MAX_DEPTH)) {
					refract = true;
					depth++;

					// Adjust reflection/refraction coefficient
					kr *= rec.kt;

					// Get normalized incoming direction and normal vectors
					vector d = normalize(rec.ray.direction);
					//d.x *= -1; d.y *= -1; d.z *= -1;
					vector n = normalize(rec.normal);

					// Get indices of refraction
					float n_dot_d = getDotProduct(n, d);
					float ni = (n_dot_d < 0) ? rec.n : N_AIR;
					float nt = (n_dot_d < 0) ? N_AIR : rec.n;

					// Handle being inside an object
					if (n_dot_d < 0) {
						n.x = -n.x;
						n.y = -n.y;
						n.z = -n.z;
						n_dot_d = getDotProduct(n, d);
					}

					// Determine whether there is total internal reflection
					float root = 1 - (ni * ni * (1 - (n_dot_d * n_dot_d)) / (nt * nt));

					// Total internal reflection
					if (root <= 0) {

						// Get vector to source
						vector toSource = normalize(getVector(hitRecord.intersection_point, 
							hitRecord.ray.origin));

						// Get reflection vector
						float rX = toSource.x + 2 * ((getDotProduct(toSource, rec.normal))
							/ pow(getDotProduct(rec.normal, rec.normal), 2.0f)) * rec.normal.x;
						float rY = toSource.y + 2 * ((getDotProduct(toSource, rec.normal))
							/ pow(getDotProduct(rec.normal, rec.normal), 2.0f)) * rec.normal.y;
						float rZ = toSource.z + 2 * ((getDotProduct(toSource, rec.normal))
							/ pow(getDotProduct(rec.normal, rec.normal), 2.0f)) * rec.normal.z;
						vector reflect = normalize(getVector(rX, rY, rZ));

						// Spawn refraction ray
						currRay = getRay(rec.intersection_point, reflect);

					} else if (ni == nt) {

						// No bending
						currRay = getRay(rec.intersection_point, rec.ray.direction);

					} else {

						// Calculate refraction vector
						vector n_dot = getVector(n.x * n_dot_d, n.y * n_dot_d, n.z * n_dot_d);
						vector d_minus = getVector(d.x - n_dot.x, d.y - n_dot.y, d.z - n_dot.z);
						float n_ratio = ni / nt;
						vector t1 = getVector(n_ratio * d_minus.x, n_ratio * d_minus.y, n_ratio * d_minus.z);
						vector t2 = getVector(n.x * sqrt(root), n.y * sqrt(root), n.z * sqrt(root));
						vector refract = getVector(t1.x - t2.x, t1.y - t2.y, t1.z - t2.z);

						// Spawn refraction ray
						currRay = getRay(rec.intersection_point, refract);
					}
				} 
			}

			// Clamp to 255
			if (hitRecord.rgb.r > 255) {
				hitRecord.rgb.r = 255;
			}
			if (hitRecord.rgb.g > 255) {
				hitRecord.rgb.g = 255;
			}
			if (hitRecord.rgb.b > 255) {
				hitRecord.rgb.b = 255;
			}
			return hitRecord;
	}

	// Get a ray with the specified origin and direction
	__host__ __device__ ray getRay(point origin, vector direction) {
		ray r;
		vector dhat = normalize(direction);
		r.origin = origin;
		r.direction = dhat;
		return r;
	}

	// Get point (x,y,z)
	__host__ __device__ point getPoint(const float x, const float y, const float z) {
		point p;
		p.x = x;
		p.y = y;
		p.z = z;
		return p;
	}

	// Get vector (x,y,z)^T
	__host__ __device__ vector getVector(const float x, const float y, const float z) {
		vector v;
		v.x = x;
		v.y = y;
		v.z = z;
		return v;
	}

	// Get the vector joining points a and b
	__host__ __device__ vector getVector(const point a, const point b) {
		vector v;
		v.x = b.x - a.x;
		v.y = b.y - a.y;
		v.z = b.z - a.z;
		return v;
	}

	// Get the length of a vector
	__host__ __device__ float getLength(vector v) {
		return (float)(sqrt(pow((float)v.x, 2.0f) + pow((float)v.y, 2.0f)
			+ pow((float)v.z, 2.0f)));
	}

	// Get the dot product of two vectors
	__host__ __device__ float getDotProduct(const vector v1, const vector v2) {
		return (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
	}

	// Get the cross product of two vectors
	__host__ __device__ vector getCrossProduct(const vector v1, const vector v2) {
		vector result;
		result.x = (v1.y * v2.z) - (v1.z * v2.y);
		result.y = (v1.z * v2.x) - (v1.x * v2.z);
		result.z = (v1.x * v2.y) - (v1.y * v2.x);
		return result;
	}

	// Convert a vector to a unit vector
	__host__ __device__ vector normalize(vector v) {
		vector vhat;
		float length = getLength(v);
		vhat.x = v.x / length;
		vhat.y = v.y / length;
		vhat.z = v.z / length;
		return vhat;
	}

	// Convert an int color to a float color
	__host__ __device__ float floatColor(int color) {
		return color / 255.0f;
	}

	// Determine the color of a pixel as a result of illumination
	// Uses Blinn-Phong Illumination Model (without ambient component)
	__host__ __device__ color illuminate(pointLight light, record sr) {

		// Lighting coefficients
		const float K_DIFFUSE = 0.6f;
		const float K_SPECULAR = 0.4f;

		// Get vector from intersection to light source
		vector source = normalize(getVector(sr.intersection_point, light.position));

		// Get normalized normal vector
		vector normal = normalize(sr.normal);

		float sourceDotNormal = getDotProduct(source, normal);

		// Get diffuse RGB components
		float diffuse_r_f = (sourceDotNormal > 0) ? K_DIFFUSE * floatColor(light.diffuseColor.r) 
			* floatColor(sr.diffuseColor.r) * sourceDotNormal : 0.0f;
		float diffuse_g_f = (sourceDotNormal > 0) ? K_DIFFUSE * floatColor(light.diffuseColor.g)
			* floatColor(sr.diffuseColor.g) * sourceDotNormal: 0.0f;
		float diffuse_b_f = (sourceDotNormal > 0) ? K_DIFFUSE * floatColor(light.diffuseColor.b)
			* floatColor(sr.diffuseColor.b) * sourceDotNormal : 0.0f;

		// Get halfway vector between light source and camera
		vector direction = normalize(sr.ray.direction);
		vector halfway = normalize(getVector(source.x - direction.x, 
			source.y - direction.y, source.z - direction.z));

		float halfDotNormal = getDotProduct(halfway, normal);

		// Get specular RGB components
		float specular_r_f = (sourceDotNormal > 0) && (halfDotNormal > 0) ? K_SPECULAR * floatColor(light.specularColor.r)
			* floatColor(sr.specularColor.r) * powf(halfDotNormal, sr.shininess) : 0.0f;
		float specular_g_f = (sourceDotNormal > 0) && (halfDotNormal > 0) ? K_SPECULAR * floatColor(light.specularColor.g)
			* floatColor(sr.specularColor.g) * powf(halfDotNormal, sr.shininess) : 0.0f;
		float specular_b_f = (sourceDotNormal > 0) && (halfDotNormal > 0) ? K_SPECULAR * floatColor(light.specularColor.b)
			* floatColor(sr.specularColor.b) * powf(halfDotNormal, sr.shininess) : 0.0f;

		// Clamp color values
		int r = ((diffuse_r_f + specular_r_f) < 1.0f) ? 
			(int)floor(255 * (diffuse_r_f + specular_r_f) + 0.5f) : 255;
		int g = ((diffuse_g_f + specular_g_f) < 1.0f) ? 
			(int)floor(255 * (diffuse_g_f + specular_g_f) + 0.5f) : 255;
		int b = ((diffuse_b_f + specular_b_f) < 1.0f) ? 
			(int)floor(255 * (diffuse_b_f + specular_b_f) + 0.5f) : 255;

		// Get resulting color
		color c;
		c.r = r;
		c.g = g;
		c.b = b;

		return c;
	}

	// Shoot a ray to color a single pixel
	__global__ void ColorPixelKernel(int pixelWidth, int pixelHeight,
		float frameWidth, float frameHeight, float focalDistance,
		point *cameraPosition, int numPlanes, plane *planes, int numSpheres,
		sphere *spheres, int numTriangles, triangle *triangles, 
		color *backgroundColor, color *pixelColors, pointLight *light) {

			// Get the relevant pixel
			int pixelX = (blockIdx.x * blockDim.x) + threadIdx.x;
			int pixelY = (blockIdx.y * blockDim.y) + threadIdx.y;

			// If this is a valid pixel
			if ((pixelX < pixelWidth) && (pixelY < pixelHeight)) {

				float wInc = frameWidth / pixelWidth;
				float hInc = frameHeight / pixelHeight;

				// Get the coordinates
				float x = (wInc - frameWidth) / 2.0f + (pixelX * wInc);
				float y = ((frameHeight - hInc) / 2.0f) - (pixelY * hInc);

				// Shoot a ray
				ray r = getRay(getPoint(0, 0, 0), 
					normalize(getVector(getPoint(0, 0, 0), 
					getPoint(x, y, focalDistance))));
				record sr = checkForHits(numPlanes, planes, numSpheres, spheres, 
					numTriangles, triangles, r, *backgroundColor, *light);

				// Set the pixel color
				pixelColors[(pixelY * pixelWidth) + pixelX] = sr.intersection ?
					sr.rgb : *backgroundColor;
			}
	}

	// Render the scene in the specified world
	void renderScene(camera *c, world *w) {
		plane *dPlanes;
		sphere *dSpheres;
		triangle *dTriangles;
		point *dCameraPosition;
		color *dBackgroundColor;
		color *dPixelColors;
		pointLight *dLight;

		// Allocate device memory
		cudaMalloc((void**)&dPlanes, w->numPlanes * sizeof(plane));
		if (!checkForError(cudaGetLastError())) {
			exit(1);
		}
		cudaMalloc((void**)&dSpheres, w->numSpheres * sizeof(sphere));
		if (!checkForError(cudaGetLastError())) {
			exit(1);
		}
		cudaMalloc((void**)&dTriangles, w->numTriangles * sizeof(triangle));
		if (!checkForError(cudaGetLastError())) {
			exit(1);
		}
		cudaMalloc((void**)&dCameraPosition, sizeof(point));
		if (!checkForError(cudaGetLastError())) {
			exit(1);
		}
		cudaMalloc((void**)&dBackgroundColor, sizeof(color));
		if (!checkForError(cudaGetLastError())) {
			exit(1);
		}
		cudaMalloc((void**)&dPixelColors, c->vp.pixelWidth * c->vp.pixelHeight
			* sizeof(color));
		if (!checkForError(cudaGetLastError())) {
			exit(1);
		}
		cudaMalloc((void**)&dLight, sizeof(pointLight));
		if (!checkForError(cudaGetLastError())) {
			exit(1);
		}

		// Copy data to device
		cudaMemcpy(dPlanes, w->planes, w->numPlanes * sizeof(plane), 
			cudaMemcpyHostToDevice);
		if (!checkForError(cudaGetLastError())) {
			exit(1);
		}
		cudaMemcpy(dSpheres, w->spheres, w->numSpheres * sizeof(sphere),
			cudaMemcpyHostToDevice);
		if (!checkForError(cudaGetLastError())) {
			exit(1);
		}
		cudaMemcpy(dTriangles, w->triangles, w->numTriangles * sizeof(triangle),
			cudaMemcpyHostToDevice);
		if (!checkForError(cudaGetLastError())) {
			exit(1);
		}
		cudaMemcpy(dCameraPosition, &(c->position), sizeof(point),
			cudaMemcpyHostToDevice);
		if (!checkForError(cudaGetLastError())) {
			exit(1);
		}
		cudaMemcpy(dBackgroundColor, &(w->backgroundColor), sizeof(color),
			cudaMemcpyHostToDevice);
		if (!checkForError(cudaGetLastError())) {
			exit(1);
		}
		cudaMemcpy(dLight, &(w->light), sizeof(pointLight),
			cudaMemcpyHostToDevice);
		if (!checkForError(cudaGetLastError())) {
			exit(1);
		}

		// Set dimensions
		int TILE_SIZE = 16;
		dim3 dimGrid((int)ceil((float)c->vp.pixelWidth / TILE_SIZE), 
			(int)ceil((float)c->vp.pixelHeight / TILE_SIZE));
		dim3 dimBlock(TILE_SIZE, TILE_SIZE);

		// Call kernel
		ColorPixelKernel<<<dimGrid, dimBlock>>>(c->vp.pixelWidth, 
			c->vp.pixelHeight, c->vp.frameWidth, c->vp.frameHeight, 
			c->focalDistance, dCameraPosition, w->numPlanes, 
			dPlanes, w->numSpheres, dSpheres, w->numTriangles,
			dTriangles, dBackgroundColor, dPixelColors, dLight);
		if (!checkForError(cudaGetLastError())) {
			exit(1);
		}

		// Synchronize
		cudaThreadSynchronize();
		if (!checkForError(cudaGetLastError())) {
			exit(1);
		}

		// Get results
		cudaMemcpy(c->vp.pixelColors, dPixelColors, c->vp.pixelWidth * 
			c->vp.pixelHeight * sizeof(color), cudaMemcpyDeviceToHost);
		if (!checkForError(cudaGetLastError())) {
			exit(1);
		}

		// Free device memory
		cudaFree(dPlanes);
		if (!checkForError(cudaGetLastError())) {
			exit(1);
		}
		cudaFree(dSpheres);
		if (!checkForError(cudaGetLastError())) {
			exit(1);
		}
		cudaFree(dTriangles);
		if (!checkForError(cudaGetLastError())) {
			exit(1);
		}
		cudaFree(dCameraPosition);
		if (!checkForError(cudaGetLastError())) {
			exit(1);
		}
		cudaFree(dBackgroundColor);
		if (!checkForError(cudaGetLastError())) {
			exit(1);
		}
		cudaFree(dPixelColors);
		if (!checkForError(cudaGetLastError())) {
			exit(1);
		}
		cudaFree(dLight);
		if (!checkForError(cudaGetLastError())) {
			exit(1);
		}
	}
}