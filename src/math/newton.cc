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

  Vector d(n_);
  Matrix tr_A = trans(A);
  Vector projection(n_), projection_plus(n_);
  Matrix H(m_, m_);
  Vector g(m_);
  Vector x(x0), p(p0);
  Vector delta_x(n_), delta_p(m_);

  Matrix T0(n_, m_), T1(m_, m_);
  
  do {
    do {
      ComputeProjection(tr_A, c, x, p, beta,
			&projection, &projection_plus);
      for (size_type i = 0; i < n_; ++i)
	d(i) = projection(i) > 0.0 ? 1.0 : 0.0;
      g.assign(prod(A, projection_plus));
      g.minus_assign(b);

      // Computing H = I + A * D * tr_A
      for (size_type i = 0; i < n_; ++i)
	for (size_type j = 0; j < m_; ++j)
	  T0(i, j) = d(i) * tr_A(i, j);
      T1 = prod(A, T0);
      H = I + T1;

      GetMaximizationDirection(H, g, internal_tolerance, &delta_p);
      p.minus_assign(delta_p);
    } while (norm_inf(delta_p) > internal_tolerance);
    ComputeProjection(tr_A, c, x, p, beta, &projection, &projection_plus);
    delta_x.assign(x);
    delta_x.minus_assign(projection_plus);
    x.assign(projection_plus);

    g.assign(prod(A, x));
    g.minus_assign(b);
  } while (norm_inf(delta_x) > external_tolerance);

  result->assign(x);
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
    r.assign(prod(A, p));
    double alpha = -inner_prod(r0_tilded, r0) / inner_prod(p, r);
    x->minus_assign(alpha * p);
    r *= alpha;
    r.plus_assign(r0);

    if (norm_inf(r) < epsilon)
      break;

    for (size_type i = 0; i < n; ++i)
      r_tilded(i) = r(i) * (*M_)(i);
    double beta = inner_prod(r_tilded, r) / inner_prod(r0_tilded, r0);
    
    p *= beta;
    p.plus_assign(r_tilded);

    r0.assign(r);
    r0_tilded.assign(r_tilded);
  }
}

void Newton::ComputeProjection(const Matrix& tr_A,
			       const Vector& c,
			       const Vector& x,
			       const Vector& p,
			       double beta,
			       Vector* projection,
			       Vector* projection_plus) {
  projection->assign(x);
  projection->plus_assign(prod(tr_A, p));
  projection->minus_assign(beta * c);
  for (size_type i = 0; i < projection->size(); ++i)
    (*projection_plus)(i) = std::max(0.0, (*projection)(i));
}

}  // namespace math
