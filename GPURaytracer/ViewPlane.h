#ifndef __VIEWPLANE_H__
#define __VIEWPLANE_H__

#include "RGBColor.h"
#include <string>
#include <iostream>
#include <fstream>

namespace MyRayTracer {

	// The field of view of the camera
	struct view_plane_t {
		
		// The width of the output image in pixels
		int pixelWidth;
		
		// The height of the output image in pixels
		int pixelHeight;
		
		// The width of the view plane in terms of 3D space
		float frameWidth;
		
		// The height of the view plane in terms of 3D space
		float frameHeight;
		
		// The array of pixel colors
		color *pixelColors;
	};
	
	typedef view_plane_t viewPlane;
	
	// Get a viewPlane of the specified size
	viewPlane getViewPlane(int pixelWidth, int pixelHeight, float frameWidth, 
		float frameHeight);
	
	// Set a pixel's color
	void setPixelColor(int pixelX, int pixelY, color *pixelColor, viewPlane *vp);

	// Get a pixel's color
	color getPixelColor(int pixelX, int pixelY, viewPlane *vp);
	
	// Delete a viewPlane to prevent memory leaks
	void deleteViewPlane(viewPlane *vp);

	// Save an image to file
	void saveImage(viewPlane *vp, std::string filename);
}

#endif