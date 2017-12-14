#include "Camera.h"

namespace MyRayTracer {

	// Get a camera with the specified information
	camera getCamera(vector upVector, point position,
		point pov, viewPlane vp, float focalDistance) {
			camera c;
			c.upVector = upVector;
			c.position = position;
			c.pov = pov;
			c.vp = vp;
			c.focalDistance = focalDistance;
			return c;
	}

	// Convert world coordinates to camera coordinates
	void convertCoordinates(camera *c, world *w)  {

		// Get camera space axes
		vector eye = getVector(c->position.x, c->position.y, c->position.z);
		vector n = getVector(c->position, c->pov);
		vector u = getCrossProduct(c->upVector, n);
		vector v = getCrossProduct(n, u);

		// Normalize the vectors
		u = normalize(u);
		v = normalize(v);
		n = normalize(n);

		// Set up conversion matrix
		matrix m = getMatrix(4, 4);
		setIndex(&m, 0, 0, u.x);
		setIndex(&m, 0, 1, u.y);
		setIndex(&m, 0, 2, u.z);
		setIndex(&m, 0, 3, -1 * getDotProduct(eye, u));
		setIndex(&m, 1, 0, v.x);
		setIndex(&m, 1, 1, v.y);
		setIndex(&m, 1, 2, v.z);
		setIndex(&m, 1, 3, -1 * getDotProduct(eye, v));
		setIndex(&m, 2, 0, n.x);
		setIndex(&m, 2, 1, n.y);
		setIndex(&m, 2, 2, n.z);
		setIndex(&m, 2, 3, -1 * getDotProduct(eye, n));
		setIndex(&m, 3, 0, 0.0f);
		setIndex(&m, 3, 1, 0.0f);
		setIndex(&m, 3, 2, 0.0f);
		setIndex(&m, 3, 3, 1.0f);

		// Convert position and normal of planes
		for (int i = 0; i < w->numPlanes; i++) {
			matrix posMat = getMatrix(w->planes[i].position);
			matrix newPos = multiply(&m, &posMat);
			w->planes[i].position.x = getIndex(&newPos, 0, 0);
			w->planes[i].position.y = getIndex(&newPos, 1, 0);
			w->planes[i].position.z = getIndex(&newPos, 2, 0);
			deleteMatrix(&posMat);
			deleteMatrix(&newPos);
		}

		// Convert centers of spheres
		for (int i = 0; i < w->numSpheres; i++) {
			matrix cMat = getMatrix(w->spheres[i].center);
			matrix newCenter = multiply(&m, &cMat);
			w->spheres[i].center.x = getIndex(&newCenter, 0, 0);
			w->spheres[i].center.y = getIndex(&newCenter, 1, 0);
			w->spheres[i].center.z = getIndex(&newCenter, 2, 0);
			deleteMatrix(&cMat);
			deleteMatrix(&newCenter);
		}

		// Convert vertices of triangles
		for (int i = 0; i < w->numTriangles; i++) {
			matrix aMat = getMatrix(w->triangles[i].a);
			matrix newA = multiply(&m, &aMat);
			w->triangles[i].a.x = getIndex(&newA, 0, 0);
			w->triangles[i].a.y = getIndex(&newA, 1, 0);
			w->triangles[i].a.z = getIndex(&newA, 2, 0);
			deleteMatrix(&aMat);
			deleteMatrix(&newA);
			matrix bMat = getMatrix(w->triangles[i].b);
			matrix newB = multiply(&m, &bMat);
			w->triangles[i].b.x = getIndex(&newB, 0, 0);
			w->triangles[i].b.y = getIndex(&newB, 1, 0);
			w->triangles[i].b.z = getIndex(&newB, 2, 0);
			deleteMatrix(&bMat);
			deleteMatrix(&newB);
			matrix cMat = getMatrix(w->triangles[i].c);
			matrix newC = multiply(&m, &cMat);
			w->triangles[i].c.x = getIndex(&newC, 0, 0);
			w->triangles[i].c.y = getIndex(&newC, 1, 0);
			w->triangles[i].c.z = getIndex(&newC, 2, 0);
			deleteMatrix(&cMat);
			deleteMatrix(&newC);
		}

		// Convert the light position
		matrix lMat = getMatrix(w->light.position);
		matrix newCenter = multiply(&m, &lMat);
		w->light.position.x = getIndex(&newCenter, 0, 0);
		w->light.position.y = getIndex(&newCenter, 1, 0);
		w->light.position.z = getIndex(&newCenter, 2, 0);
		deleteMatrix(&lMat);
		deleteMatrix(&newCenter);

		deleteMatrix(&m);
	}
}