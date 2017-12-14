#include "Matrix.h"

namespace MyRayTracer {

	// Get a matrix of the specified size
	matrix getMatrix(int rows, int cols) {
		matrix m;
		m.rows = rows;
		m.cols = cols;
		m.data = new float[rows * cols];
		return m;
	}

	// Convert a vector into a matrix
	matrix getMatrix(vector v) {
		matrix m = getMatrix(4, 1);
		setIndex(&m, 0, 0, v.x);
		setIndex(&m, 1, 0, v.y);
		setIndex(&m, 2, 0, v.z);
		setIndex(&m, 3, 0, 1.0f);
		return m;
	}

	// Set an index in a matrix
	void setIndex(matrix *m, int row, int col, float entry) {
		if ((m->rows > row) && (m->cols > col)) {
			if (m->data == 0) {
				m->data = new float[m->rows * m->cols];
			}
			m->data[(row * m->cols) + col] = entry;
		}
	}

	// Get an index in a matrix
	float getIndex(matrix *m, int row, int col) {
		if ((m->rows > row) && (m->cols > col)) {
			if (m->data == 0) {
				m->data = new float[m->rows * m->cols];
			}
			return m->data[(row * m->cols) + col];
		}
		return 0.0f;
	}

	// Multiply two matrices and return the product
	matrix multiply(matrix *a, matrix *b) {
		matrix result = getMatrix(a->rows, b->cols);
		for (int i = 0; i < a->rows; i++) {
			for (int j = 0; j < b->cols; j++) {
				float entry = 0.0f;
				for (int k = 0; k < a->cols; k++) {
					entry += getIndex(a, i, k) * getIndex(b, k, j);
				}
				setIndex(&result, i, j, entry);
			}
		}
		return result;
	}

	// Delete a matrix
	void deleteMatrix(matrix *m) {
		delete m->data;
		m->data = 0;
	}
}