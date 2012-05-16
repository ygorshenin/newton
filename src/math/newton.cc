#include "math/newton.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>

#include "boost/numeric/ublas/io.hpp"

namespace math {

using boost::numeric::ublas::inner_prod;
using boost::numeric::ublas::norm_2;
using boost::numeric::ublas::norm_inf;
using boost::numeric::ublas::prod;
using boost::numeric::ublas::trans;

void Newton::SolveLinearSystem(const Matrix& A,
			       const Vector& b,
			       const Vector& c,
			       const Vector& x0,
			       const Vector& p0,
			       double beta,
			       double external_tolerance,
			       double internal_tolerance,
			       Vector* result) {
  Initialize(A, b, c, x0, p0, result);

  const IdentityMatrix I(m_, m_);

  Matrix tr_A = trans(A);
  Matrix D(n_, n_);
  Vector projection(n_), projection_plus(n_);
  Matrix H(m_, m_);
  Vector g(m_);
  Vector x(x0), p(p0);
  Vector delta_x(n_), delta_p(m_);

  for (size_t i = 0; i < n_; ++i) {
    for (size_t j = 0; j < n_; ++j)
      D(i, j) = 0.0;
  }

  do {
    do {
      ComputeProjection(tr_A, c, x, p, beta,
			&projection, &projection_plus);
      for (size_type i = 0; i < n_; ++i)
	D(i, i) = projection(i) > 0.0 ? 1.0 : 0.0;
      g = prod(A, projection_plus) - b;

      H = I + prod(A, Matrix(prod(D, tr_A)));

      GetMaximizationDirection(H, g, internal_tolerance, &delta_p);
      p -= delta_p;
//       std::clog << "p=" << p << std::endl;
    } while (norm_2(delta_p) > internal_tolerance);
    ComputeProjection(tr_A, c, x, p, beta, &projection, &projection_plus);
    delta_x = x - projection_plus;
    x = projection_plus;

    g = prod(A, x) - b;
//     std::clog << "g=" << g << std::endl;
//     std::clog << "x=" << x << std::endl;
//     std::clog << "value=" << inner_prod(x, c) << std::endl;
  } while (norm_2(delta_x) > external_tolerance);

  *result = x;
}

void Newton::Initialize(const Matrix& A,
			const Vector& b,
			const Vector& c,
			const Vector& x0,
			const Vector& p0,
			Vector* result) {
  m_ = A.size1();
  n_ = A.size2();

  assert(b.size() == m_);
  assert(c.size() == n_);
  assert(x0.size() == n_);
  assert(p0.size() == m_);
  assert(result->size() == n_);

  M_.reset(new Vector(m_, m_));
}

void Newton::GetMaximizationDirection(const Matrix& A,
				      const Vector& b,
				      double epsilon,
				      Vector* x) {
  const size_type n = A.size1();

  for (size_type i = 0; i < n; ++i)
    (*M_)(i) = A(i, i) * A(i, i);

  std::fill(x->begin(), x->end(), 0.0);

  Vector r0(b), r(n);
  Vector r0_tilded(n), r_tilded(n);
  for (size_type i = 0; i < n; ++i)
    r0_tilded(i) = r0(i) * (*M_)(i);

  Vector p(r0_tilded);
  while (true) {
    double alpha = -inner_prod(r0_tilded, r0) / inner_prod(p, prod(A, p));
    *x -= alpha * p;
    r = prod(A, p);
    r *= alpha;
    r += r0;

//     std::clog << "r=" << r << std::endl;
    if (norm_inf(r) < epsilon)
      break;

    for (size_type i = 0; i < n; ++i)
      r_tilded(i) = r(i) * (*M_)(i);
    double beta = inner_prod(r_tilded, r) / inner_prod(r0_tilded, r0);
    
    p *= beta;
    p += r_tilded;

    r0 = r;
    r0_tilded = r_tilded;
  }
}

void Newton::ComputeProjection(const Matrix& tr_A,
			       const Vector& c,
			       const Vector& x,
			       const Vector& p,
			       double beta,
			       Vector* projection,
			       Vector* projection_plus) {
  *projection = x + prod(tr_A, p) - beta * c;
  for (size_type i = 0; i < projection->size(); ++i)
    (*projection_plus)(i) = std::max(0.0, (*projection)(i));
}

}  // namespace math
