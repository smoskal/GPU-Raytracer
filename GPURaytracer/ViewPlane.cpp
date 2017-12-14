#include "ViewPlane.h"

namespace MyRayTracer {

	// Get a viewPlane of the specified size
	viewPlane getViewPlane(int pixelWidth, int pixelHeight, float frameWidth, 
		float frameHeight) {
		viewPlane vp;
		vp.pixelWidth = pixelWidth;
		vp.pixelHeight = pixelHeight;
		vp.frameWidth = frameWidth;
		vp.frameHeight = frameHeight;
		vp.pixelColors = new color[pixelWidth*pixelHeight];
		return vp;
	}

	// Set a pixel's color
	void setPixelColor(int pixelX, int pixelY, color *pixelColor, viewPlane *vp) {
		if ((vp->pixelWidth > pixelX) && (vp->pixelHeight > pixelY)) {
			vp->pixelColors[(pixelX * vp->pixelWidth) + pixelY] = *pixelColor;
		}
	}

	// Get a pixel's color
	color getPixelColor(int pixelX, int pixelY, viewPlane *vp) {
		if ((vp->pixelWidth > pixelX) && (vp->pixelHeight > pixelY)) {
			return vp->pixelColors[(pixelX * vp->pixelWidth) + pixelY];
		} else {
			return ERROR;
		}
	}

	// Delete a viewPlane to prevent memory leaks
	void deleteViewPlane(viewPlane *vp) {
		delete vp->pixelColors;
		vp->pixelColors = 0;
	}

	// Save an image to file
	void saveImage(viewPlane *vp, std::string filename) {
		std::ofstream file;
		file.open("./images/" + filename + ".ppm");
		file << "P3" << std::endl << vp->pixelWidth << " " << vp->pixelHeight 
			<< std::endl << "255" << std::endl;
		for (int j = 0; j < vp->pixelHeight; j++) {
			for (int i = 0; i < vp->pixelWidth; i++) {
				color current = vp->pixelColors[(j * vp->pixelWidth) + i];
				file << current.r << " " << current.g << " " << current.b << " ";
			}
			file << std::endl;
		}
		file.close();
	}
}