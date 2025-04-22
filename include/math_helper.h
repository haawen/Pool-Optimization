#ifndef MATH_H
#define MATH_H

// Z[i] = X[i] - Y[i]
void divV3(double* X, double Y, double* Z);

// Z[i] = X[i] - Y[i]
void subV3(double* X, double* Y, double* Z);

double dotV3(double* X, double* Y);

void crossV3(double* X, double* Y, double* Z);

#endif