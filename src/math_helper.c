#include "math_helper.h"

void divV3(double* X, double Y, double* Z) {
    Z[0] = X[0] / Y;
    Z[1] = X[1] / Y;
    Z[2] = X[2] / Y;
}

void subV3(double* X, double* Y, double* Z) {
    Z[0] = X[0] - Y[0];
    Z[1] = X[1] - Y[1];
    Z[2] = X[2] - Y[2];
}

double dotV3(double* X, double* Y) {
    return X[0] * Y[0] +
    X[1] * Y[1] +
    X[2] * Y[2];
}

void crossV3(double* X, double* Y, double* Z) {
    Z[0] = X[1]*Y[2] - X[2]*Y[1];
    Z[1] = X[2]*Y[0] - X[0]*Y[2];
    Z[2] = X[0]*Y[1] - X[1]*Y[0];
}