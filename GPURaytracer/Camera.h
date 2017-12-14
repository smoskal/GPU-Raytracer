#ifndef __CAMERA_H__
#define __CAMERA_H__

#include "Space3D.h"
#include "ViewPlane.h"
#include "World.h"
#include "Matrix.h"
#include "Light.h"

namespace MyRayTracer {

	// Struct representing a camera looking at a scene to be rendered
	struct camera_t {
	
		// Orientation vector specifying which direction is "up" for the camera
		vector upVector;
		
		// The position of the camera in world coordinates
		point position;
		
		// The point at which the camera is looking in world coordinates
		point pov;
		
		// The view plane of the camera
		viewPlane vp;
		
		// The focal distance of the camera
		float focalDistance;
	};

	typedef camera_t camera;

	// Get a camera with the specified information
	camera getCamera(vector upVector, point position,
		point pov, viewPlane vp, float focalDistance);

	// Convert world coordinates to camera coordinates
	void convertCoordinates(camera *c, world *w) ;

	// Render the scene in the specified world
	void renderScene(camera *c, world *w);

}

#endif