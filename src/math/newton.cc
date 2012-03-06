#include "math/newton.h"

#include <algorithm>
#include <cassert>
#include <cstddef>

namespace math {

using boost::numeric::ublas::inner_prod;
using boost::numeric::ublas::norm_2;
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
			       Vector* result) const {
  const size_type m = A.size1(), n = A.size2();
  assert(b.size() == m);
  assert(c.size() == n);
  assert(x0.size() == n);
  assert(p0.size() == m);
  assert(result != NULL);
  assert(result->size() == n);

  const IdentityMatrix I(m, m);

  Matrix tr_A = trans(A);
  Matrix D(n, n);
  Vector projection(n), projection_plus(n);
  Matrix H(m, m);
  Vector g(m);
  Vector x(x0), p(p0);
  Vector delta_x(n), delta_p(m);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j)
      D(i, i) = 0.0;
  }

  do {
    do {
      ComputeProjection(tr_A, c, x, p, beta,
			&projection, &projection_plus);
      for (size_type i = 0; i < n; ++i)
	D(i, i) = projection(i) > 0.0 ? 1.0 : 0.0;
      g = prod(A, projection_plus) - b;

      H = I + prod(A, Matrix(prod(D, tr_A)));

      GetMaximizationDirection(H, g, internal_tolerance, &delta_p);
      p -= delta_p;
    } while (norm_2(delta_p) < internal_tolerance);
    ComputeProjection(tr_A, c, x, p, beta, &projection, &projection_plus);
    delta_x = x - projection_plus;
    x = projection_plus;
  } while (norm_2(delta_x) < external_tolerance);

  *result = x;
}

void Newton::GetMaximizationDirection(const Matrix& A,
				      const Vector& b,
				      double epsilon,
				      Vector* x) const {
  const size_type n = A.size1();

  Matrix M(n, n);
  for (size_type i = 0; i < n; ++i) {
    for (size_type j = 0; j < n; ++j)
      M(i, i) = 0.0;
    M(i, i) = A(i, i) * A(i, i);
  }

  std::fill(x->begin(), x->end(), 0.0);

  Vector r0(b), r(n);
  Vector r0_tilded(n), r_tilded(n);
  for (size_type i = 0; i < n; ++i)
    r0_tilded(i) = r0(i) * M(i, i);

  Vector p(r0_tilded);
  while (true) {
    double alpha = -inner_prod(r0_tilded, r0) / inner_prod(p, prod(A, p));
    *x -= alpha * p;
    r = r0 + alpha * prod(A, p);

    if (abs(*std::max_element(r0.begin(), r0.end())) < epsilon)
      break;

    for (size_type i = 0; i < n; ++i)
      r_tilded(i) = r(i) * M(i, i);
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
			       Vector* projection_plus) const {
  *projection = x + prod(tr_A, p) - beta * c;
  for (size_type i = 0; i < projection->size(); ++i)
    (*projection_plus)(i) = std::max(0.0, (*projection)(i));
}

}  // namespace math
