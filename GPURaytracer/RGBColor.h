#ifndef __RGBCOLOR_H__
#define __RGBCOLOR_H__

namespace MyRayTracer {

	// Struct used to define a color using RGB values
	struct rgb_color_t {
		int r, g, b;
	};

	typedef rgb_color_t color;

	// Get a color with the specified RGB values (pseudo-constructor)
	color getColor(const int r, const int g, const int b);
	
	// Contants
	const color RED = getColor(255, 0, 0);
	const color GREEN = getColor(0, 255, 0);
	const color BLUE = getColor(0, 0, 255);
	const color BLACK = getColor(0, 0, 0);
	const color WHITE = getColor(255, 255, 255);
	const color MAGENTA = getColor(255, 0, 255);
	const color YELLOW = getColor(255, 255, 0);
	const color CYAN = getColor(0, 255, 255);
	const color ERROR = getColor(-1, -1, -1);

	// addend1 + addend2
	color sum(const color *addend1, const color *addend2);

	// factor1 * factor2
	color product(const color *factor1, const color *factor2);

	// Scale result's RGB values
	color scale(const color *source, const float scale);
	
	// Whether two colors are equal
	bool equals(const color *c1, const color *c2);

	// Whether a color instance is valid
	bool isValidColor(const color *c);
}

#endif