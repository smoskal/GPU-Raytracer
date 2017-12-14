#include "RGBColor.h"

namespace MyRayTracer {

	// Get a color with the specified RGB values (pseudo-constructor)
	color getColor(const int r, const int g, const int b) {
		color c;
		c.r = r;
		c.g = g;
		c.b = b;
		return c;
	}

	// addend1 + addend2
	color sum(const color *addend1, const color *addend2) {
		color c;
		c.r = addend1->r + addend2->r;
		c.g = addend1->g + addend2->g;
		c.b = addend1->b + addend2->b;
		return c;
	}

	// factor1 * factor2
	color product(const color *factor1, const color *factor2) {
		color c;
		c.r = factor1->r * factor2->r;
		c.g = factor1->g * factor2->g;
		c.b = factor1->b * factor2->b;
		return c;
	}

	// Scale result's RGB values
	color scale(const color *source, const float scale) {
		color c;
		c.r = (int)(source->r * scale);
		c.g = (int)(source->g * scale);
		c.b = (int)(source->b * scale);
		return c;
	}

	// Whether two colors are equal
	bool equals(const color *c1, const color *c2) {
		return (c1->r == c2->r) && (c1->g == c2->g) && (c1->b == c2->b);
	}

	// Whether a color instance is valid
	bool isValidColor(const color *c) {
		return (c->r >= 0) && (c->r <= 255) && (c->g >= 0) && (c->g <= 255) 
			&& (c->b >= 0) && (c->b <= 255);
	}
}