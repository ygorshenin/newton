#ifndef MATH_NEWTON_H_
#define MATH_NEWTON_H_
#pragma once

#include <valarray>

namespace math {

class Newton {
public:
  typedef std::valarray<double> Vector;
  typedef std::valarray<Vector> Matrix;

  // Solves forward linear programming problem tr(c) * x -> min, where x
  // is an n-dimensional vector, such that A * x = b, and all components
  // of x is non-negative numbers.
  void SolveLinearSystem(
      const Matrix& A,
      const Vector& b,
      const Vector& c,
      const Vector& x0,
      const Vector& p0,
      double beta,
      double external_tolerance,
      double internal_tolerance,
      Vector* result);

private:
  void CheckSystem(const Matrix& A,
		   const Vector& b,
		   const Vector& c,
		   const Vector& x0,
		   const Vector& p0,
		   Vector* result);

  double GetVectorNorm(const Vector& v) const;

  // Solves A * x = b via preconditioned conjugate gradient method.
  void GetMaximizationDirection(const Matrix& A,
				const Vector& b,
				double epsilon,
				Vector* x);
  void ComputeProjection(const Matrix& tr_A,
			 const Vector& c,
			 const Vector& x,
			 const Vector& p,
			 double beta,
			 Vector* projection,
			 Vector* projection_plus);
};

}  // namespace math

#endif  // MATH_NEWTON_H_
