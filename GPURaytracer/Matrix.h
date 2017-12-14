#ifndef __MATRIX_H__
#define __MATRIX_H__

#include "Space3D.h"

namespace MyRayTracer {

	// A struct used to represent a matrix
	struct matrix_t {
		int rows;
		int cols;
		float *data;
	};

	typedef matrix_t matrix;

	// Get a matrix of the specified size
	matrix getMatrix(int rows, int cols);

	// Convert a vector into a matrix
	matrix getMatrix(vector v);

	// Set an index in a matrix
	void setIndex(matrix *m, int row, int col, float entry);

	// Get an index in a matrix
	float getIndex(matrix *m, int row, int col);

	// Multiply two matrices and return the product
	matrix multiply(matrix *a, matrix *b);

	// Delete a matrix
	void deleteMatrix(matrix *m);
}

#endif