#ifndef MATH_NEWTON_H_
#define MATH_NEWTON_H_
#pragma once

#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/vector.hpp"
#include "boost/scoped_ptr.hpp"

namespace math {

class Newton {
 public:
  typedef ::boost::numeric::ublas::zero_matrix<double> ZeroMatrix;
  typedef ::boost::numeric::ublas::identity_matrix<double> IdentityMatrix;
  typedef ::boost::numeric::ublas::matrix<double> Matrix;
  typedef ::boost::numeric::ublas::vector<double> Vector;
  typedef Matrix::size_type size_type;

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
  void Initialize(const Matrix& A,
		  const Vector& b,
		  const Vector& c,
		  const Vector& x0,
		  const Vector& p0,
		  Vector* result);

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

  size_type m_, n_;
  ::boost::scoped_ptr<Vector> M_;
};

}  // namespace math

#endif  // MATH_NEWTON_H_
