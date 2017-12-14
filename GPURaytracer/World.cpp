#include "World.h"
#include <cfloat>
#include <cuda_runtime_api.h>

namespace MyRayTracer {

	// Constructor
	world getWorld(color backgroundColor) {
		world w;
		w.backgroundColor = backgroundColor;
		w.numPlanes = 0;
		w.numSpheres = 0;
		w.numTriangles = 0;
		w.planesSize = 10;
		w.planes = new plane[w.planesSize];
		w.spheresSize = 10;
		w.spheres = new sphere[w.spheresSize];
		w.trianglesSize = 10;
		w.triangles = new triangle[w.trianglesSize];
		return w;
	}

	// Destructor
	void deleteWorld(world *w) {
		delete w->planes;
		w->planes = 0;
		delete w->spheres;
		w->spheres = 0;
		delete w->triangles;
		w->triangles = 0;
		w->numPlanes = 0;
		w->numSpheres = 0;
		w->numTriangles = 0;
	}

	// Add a plane
	void addPlane(world *w, plane p) {
		if (w->numPlanes == w->planesSize) {
			plane *temp = w->planes;
			w->planesSize *= 2;
			w->planes = new plane[w->planesSize];
			for (int i = 0; i < w->numPlanes; i++) {
				w->planes[i] = temp[i];
			}
			delete temp;
			temp = 0;
		}
		w->planes[w->numPlanes] = p;
		w->numPlanes++;
	}

	// Add a sphere
	void addSphere(world *w, sphere s) {
		if (w->numSpheres == w->spheresSize) {
			sphere *temp = w->spheres;
			w->spheresSize *= 2;
			w->spheres = new sphere[w->spheresSize];
			for (int i = 0; i < w->numSpheres; i++) {
				w->spheres[i] = temp[i];
			}
			delete temp;
			temp = 0;
		}
		w->spheres[w->numSpheres] = s;
		w->numSpheres++;
	}

	// Add a triangle
	void addTriangle(world *w, triangle t) {
		if (w->numTriangles == w->trianglesSize) {
			triangle *temp = w->triangles;
			w->trianglesSize *= 2;
			w->triangles = new triangle[w->trianglesSize];
			for (int i = 0; i < w->numTriangles; i++) {
				w->triangles[i] = temp[i];
			}
			delete temp;
			temp = 0;
		}
		w->triangles[w->numTriangles] = t;
		w->numTriangles++;
	}

	// Add a cube defined by two opposite corners
	void addCube(world *w, point a, point h, color diffuseColor,
		color specularColor, float shininess, float kr) {

			// Get the remaining corners
			point b = getPoint(h.x, a.y, a.z);
			point c = getPoint(a.x, h.y, a.z);
			point d = getPoint(h.x, h.y, a.z);
			point e = getPoint(a.x, a.y, h.z);
			point f = getPoint(h.x, a.y, h.z);
			point g = getPoint(a.x, h.y, h.z);

			// Add the triangles
			addTriangle(w, getTriangle(a, b, c, diffuseColor, specularColor, shininess, kr));
			addTriangle(w, getTriangle(b, d, c, diffuseColor, specularColor, shininess, kr));
			addTriangle(w, getTriangle(a, b, e, diffuseColor, specularColor, shininess, kr));
			addTriangle(w, getTriangle(b, f, e, diffuseColor, specularColor, shininess, kr));
			addTriangle(w, getTriangle(a, c, e, diffuseColor, specularColor, shininess, kr));
			addTriangle(w, getTriangle(c, g, e, diffuseColor, specularColor, shininess, kr));
			addTriangle(w, getTriangle(c, d, g, diffuseColor, specularColor, shininess, kr));
			addTriangle(w, getTriangle(d, h, g, diffuseColor, specularColor, shininess, kr));
			addTriangle(w, getTriangle(b, d, f, diffuseColor, specularColor, shininess, kr));
			addTriangle(w, getTriangle(d, h, f, diffuseColor, specularColor, shininess, kr));
			addTriangle(w, getTriangle(e, f, g, diffuseColor, specularColor, shininess, kr));
			addTriangle(w, getTriangle(f, h, g, diffuseColor, specularColor, shininess, kr));
	}

	// Build the world
	world build(std::string filepath) {
		std::ifstream file;
		file.open(filepath);
		std::string line;

		getline(file, line);
		while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
			(line[0] == '\n')) {
				getline(file, line);
		}
		int background_r = atoi(line.c_str());
		getline(file, line);
		while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
			(line[0] == '\n')) {
				getline(file, line);
		}
		int background_g = atoi(line.c_str());
		getline(file, line);
		while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
			(line[0] == '\n')) {
				getline(file, line);
		}
		int background_b = atoi(line.c_str());
		world w = getWorld(getColor(background_r, background_g, background_b));

		while (getline(file, line)) {
			while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || (line[0] == '\n')) {
				getline(file, line);
			}
			std::cout << line << std::endl;
			if (line == "SPHERE") {
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float center_x = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float center_y = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float center_z = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float radius = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int diffuse_r = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int diffuse_g = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int diffuse_b = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int specular_r = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int specular_g = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int specular_b = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float shininess = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float kr = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float kt = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float n = (float)atof(line.c_str());
				color diffuse = getColor(diffuse_r, diffuse_g, diffuse_b);
				color specular = getColor(specular_r, specular_g, specular_b);
				point center = getPoint(center_x, center_y, center_z);
				addSphere(&w, getSphere(center, radius, diffuse, specular, shininess, kr, kt, n));
			} else if (line == "PLANE") {
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float pos_x = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float pos_y = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float pos_z = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float norm_x = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float norm_y = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float norm_z = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int color1_r = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int color1_g = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int color1_b = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int color2_r = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int color2_g = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int color2_b = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float checkerSize = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int specular_r = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int specular_g = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int specular_b = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float shininess = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float kr = (float)atof(line.c_str());
				color specular = getColor(specular_r, specular_g, specular_b);
				point position = getPoint(pos_x, pos_y, pos_z);
				vector normal = getVector(norm_x, norm_y, norm_z);
				color c1 = getColor(color1_r, color1_g, color1_b);
				color c2 = getColor(color2_r, color2_g, color2_b);
				addPlane(&w, getPlane(position, normal, c1, c2, checkerSize, specular, shininess, kr));
			} else if (line == "TRIANGLE") {
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float ax = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float ay = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float az = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float bx = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float by = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float bz = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float cx = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float cy = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float cz = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int diffuse_r = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int diffuse_g = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int diffuse_b = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int specular_r = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int specular_g = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int specular_b = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float shininess = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float kr = (float)atof(line.c_str());
				color diffuse = getColor(diffuse_r, diffuse_g, diffuse_b);
				color specular = getColor(specular_r, specular_g, specular_b);
				point a = getPoint(ax, ay, az);
				point b = getPoint(bx, by, bz);
				point c = getPoint(cx, cy, cz);
				addTriangle(&w, getTriangle(a, b, c, diffuse, specular, shininess, kr));
			} else if (line == "CUBE") {
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float ax = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float ay = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float az = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float bx = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float by = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float bz = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int diffuse_r = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int diffuse_g = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int diffuse_b = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int specular_r = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int specular_g = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int specular_b = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float shininess = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float kr = (float)atof(line.c_str());
				color diffuse = getColor(diffuse_r, diffuse_g, diffuse_b);
				color specular = getColor(specular_r, specular_g, specular_b);
				point a = getPoint(ax, ay, az);
				point b = getPoint(bx, by, bz);
				addCube(&w, a, b, diffuse, specular, shininess, kr);
			} if (line == "LIGHT") {
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float pos_x = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float pos_y = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				float pos_z = (float)atof(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int diffuse_r = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int diffuse_g = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int diffuse_b = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int specular_r = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int specular_g = atoi(line.c_str());
				getline(file, line);
				while ((line.empty()) || (line[0] == '#') || (line[0] == '\r') || 
					(line[0] == '\n')) {
						getline(file, line);
				}
				int specular_b = atoi(line.c_str());

				point pos = getPoint(pos_x, pos_y, pos_z);
				color diffuse = getColor(diffuse_r, diffuse_g, diffuse_b);
				color specular = getColor(specular_r, specular_g, specular_b);
				w.light = getPointLight(pos, diffuse, specular);
			}
		}
		return w;
	}
}