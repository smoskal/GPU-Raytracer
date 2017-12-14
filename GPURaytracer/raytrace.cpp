#include "World.h"
#include "Camera.h"
#include "Space3D.h"
#include <sstream>

#ifdef TIMING

#include <iostream>
#include <fstream>
#include <ctime>

#endif

using namespace MyRayTracer;

int main() {

	// Build the world
	world w = build("./world.cfg");

	// Define a camera
	int j = 1;
	camera c = getCamera(getVector(0, 1, 0), getPoint(-10, 0, -150),
		getPoint(0, 0, 0), getViewPlane(400 * j, 300 * j, 800.0, 600.0), 
		800.0);

	// Convert from world space to camera space
	convertCoordinates(&c, &w);

#ifdef TIMING

	// Set up timing output file
	std::ofstream file;
	file.open("timing.txt", std::ofstream::out | std::ofstream::app);
		
	// Number of iterations to perform
	int NUM_ITERATIONS = 50;

	// Start time
	clock_t time = clock();

	for (int i = 0; i < NUM_ITERATIONS; i++) {

#endif

	// Render the scene
	renderScene(&c, &w);

#ifdef TIMING
	}

	// Stop the clock
	time = clock() - time;

	// Print the results
	file << "Execution time for " << c.vp.pixelWidth << "x" << c.vp.pixelHeight
		<< " image = " << ((float)time / CLOCKS_PER_SEC / NUM_ITERATIONS) 
		<< " seconds" << std::endl;

	file.close();

#endif

	// Save the image
	std::ostringstream ss;
	ss << "GPURefraction" << "_" << c.vp.pixelWidth << "x" << c.vp.pixelHeight;
	std::string filename = ss.str();
	saveImage(&(c.vp), filename);

	// Deallocate heap space
	deleteWorld(&w);
	deleteViewPlane(&(c.vp));
}