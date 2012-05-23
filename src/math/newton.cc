#include "math/newton.h"

#include <omp.h>

#include <algorithm>
#include <cassert>

namespace math {

void Newton::SolveLinearSystem(const Matrix& A,
			       const Vector& b,
			       const Vector& c,
			       const Vector& x0,
			       const Vector& p0,
			       double beta,
			       double external_tolerance,
			       double internal_tolerance,
			       Vector* result) {
  int m = A.size(), n = c.size();

  Vector d(n);
  Matrix tr_A(Vector(m), n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j)
      tr_A[i][j] = A[j][i];
  }
  Vector projection(n), projection_plus(n);
  Matrix H(Vector(m), m);
  Vector g(m);
  Vector x(x0), p(p0);
  Vector delta_x(n), delta_p(m);

  do {
    do {
      ComputeProjection(tr_A, c, x, p, beta, &projection, &projection_plus);

#pragma omp parallel for default(none) shared(d, n, projection)
      for (int i = 0; i < n; ++i)
	d[i] = projection[i] > 0.0 ? 1.0 : 0.0;

#pragma omp parallel for default(none) shared(A, b, g, m, n, projection_plus)
      for (int i = 0; i < m; ++i) {
	g[i] = 0.0;
	for (int j = 0; j < n; ++j)
	  g[i] += A[i][j] * projection_plus[j];
	g[i] -= b[i];
      }

#pragma omp parallel for default(none) shared(A, d, H, m, n)
      for (int i = 0; i < m; ++i) {
	for (int j = 0; j < m; ++j) {
	  H[i][j] = 0.0;
	  for (int k = 0; k < n; ++k)
	    H[i][j] += A[i][k] * A[j][k] * d[k];
	}
	H[i][i] += 1.0;
      }

      GetMaximizationDirection(H, g, internal_tolerance, &delta_p);

#pragma omp parallel for default(none) shared(m, p, delta_p)
      for (int i = 0; i < m; ++i)
	p[i] -= delta_p[i];
    } while (GetVectorNorm(delta_p) > internal_tolerance);
    ComputeProjection(tr_A, c, x, p, beta, &projection, &projection_plus);

#pragma omp parallel for default(none) shared(delta_x, n, projection_plus, x)
    for (int i = 0; i < n; ++i) {
      delta_x[i] = x[i] - projection_plus[i];
      x[i] = projection_plus[i];
    }

#pragma omp parallel for default(none) shared(A, b, g, m, n, x)
    for (int i = 0; i < m; ++i) {
      g[i] = 0.0;
      for (int j = 0; j < n; ++j)
	g[i] += A[i][j] * x[j];
      g[i] -= b[i];
    }
  } while (GetVectorNorm(delta_x) > external_tolerance);

  for (int i = 0; i < n; ++i)
    (*result)[i] = x[i];
}

void Newton::CheckSystem(const Matrix& A,
			 const Vector& b,
			 const Vector& c,
			 const Vector& x0,
			 const Vector& p0,
			 Vector* result) {
  int m = A.size(), n = c.size();

  assert(static_cast<int>(b.size()) == m);
  assert(static_cast<int>(x0.size()) == n);
  assert(static_cast<int>(p0.size()) == m);
  assert(static_cast<int>(result->size()) == n);
}

double Newton::GetVectorNorm(const Vector& v) const {
  int n = v.size();
  double global_norm = 0.0;
#pragma omp parallel default(none) shared(global_norm, n, v)
  {
    double t;
    double local_norm = 0.0;
#pragma omp for nowait
    for (int i = 0; i < n; ++i) {
      t = fabs(v[i]);
      if (t > local_norm)
	local_norm = t;
    }
#pragma omp critical
    {
      if (local_norm > global_norm)
	global_norm = local_norm;
    }
  }
  return global_norm;
}

void Newton::GetMaximizationDirection(const Matrix& H,
				      const Vector& g,
				      double eps,
				      Vector* x) {
  int n = H.size();
  assert(static_cast<int>(g.size()) == n);
  assert(static_cast<int>(x->size()) == n);

  Vector m(n);
  Vector r0(n), r(n);
  Vector r0_tilded(n), r_tilded(n);
  Vector p(n);
#pragma omp parallel for default(none) shared(g, H, m, n, p, r0, r0_tilded, x)
  for (int i = 0; i < n; ++i) {
    m[i] = H[i][i] * H[i][i];
    r0[i] = g[i];
    p[i] = r0_tilded[i] = g[i] * m[i];
    (*x)[i] = 0.0;
  }

  double u, v;
  while (true) {
    u = 0.0;
    v = 0.0;
#pragma omp parallel for default(none) reduction(+:u, v)       \
  shared(H, n, p, r, r0, r0_tilded)
    for (int i = 0; i < n; ++i) {
      r[i] = 0.0;
      for (int j = 0; j < n; ++j)
	r[i] += H[i][j] * p[j];
      u += r0_tilded[i] * r0[i];
      v += p[i] * r[i];
    }

    double alpha = -u / v;
#pragma omp parallel for default(none) shared(alpha, n, p, r, r0, x)
    for (int i = 0; i < n; ++i) {
      (*x)[i] -= alpha * p[i];
      r[i] = r[i] * alpha + r0[i];
    }

    if (GetVectorNorm(r) < eps)
      break;

    u = 0.0;
    v = 0.0;
#pragma omp parallel for default(none) reduction(+:u, v) \
  shared(m, n, r, r_tilded, r0, r0_tilded)
    for (int i = 0; i < n; ++i) {
      r_tilded[i] = r[i] * m[i];
      u += r_tilded[i] * r[i];
      v += r0_tilded[i] * r0[i];
    }

    double beta = u / v;
#pragma omp parallel for default(none) shared(beta, n, p, r, r_tilded, r0, r0_tilded)
    for (int i = 0; i < n; ++i) {
      p[i] = p[i] * beta + r_tilded[i];
      r0[i] = r[i];
      r0_tilded[i] = r_tilded[i];
    }
  }
}

void Newton::ComputeProjection(const Matrix& tr_A,
			       const Vector& c,
			       const Vector& x,
			       const Vector& p,
			       double beta,
			       Vector* projection,
			       Vector* projection_plus) {
  int n = tr_A.size(), m = p.size();
  assert(static_cast<int>(c.size()) == n);
  assert(static_cast<int>(x.size()) == n);
  assert(static_cast<int>(projection->size()) == n);
  assert(static_cast<int>(projection_plus->size()) == n);

#pragma omp parallel for default(none) \
  shared(beta, c, m, n, p, projection, projection_plus, tr_A, x)
  for (int i = 0; i < n; ++i) {
    (*projection)[i] = 0.0;
    for (int j = 0; j < m; ++j)
      (*projection)[i] += tr_A[i][j] * p[j];
    (*projection)[i] += x[i] - beta * c[i];
    (*projection_plus)[i] = std::max(0.0, (*projection)[i]);
  }
}

}  // namespace math
