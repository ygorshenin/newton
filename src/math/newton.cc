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
  Initialize(A, b, c, x0, p0, result);

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

#pragma omp parallel default(shared)
      {
#pragma omp for
        for (int i = 0; i < n; ++i)
          d[i] = projection[i] > 0.0 ? 1.0 : 0.0;

#pragma omp for
        for (int i = 0; i < m; ++i) {
          g[i] = 0.0;
          for (int j = 0; j < n; ++j)
            g[i] += A[i][j] * projection_plus[j];
          g[i] -= b[i];
        }

#pragma omp for
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < m; ++j) {
            H[i][j] = 0.0;
            for (int k = 0; k < n; ++k)
              H[i][j] += A[i][k] * A[j][k] * d[k];
          }
          H[i][i] += 1.0;
        }

#pragma omp single
        GetMaximizationDirection(H, g, internal_tolerance, &delta_p);

#pragma omp for
        for (int i = 0; i < m; ++i)
          p[i] -= delta_p[i];
      }
    } while (GetVectorNorm(delta_p) > internal_tolerance);
    ComputeProjection(tr_A, c, x, p, beta, &projection, &projection_plus);

#pragma omp parallel default(shared)
    {
#pragma omp for
      for (int i = 0; i < n; ++i) {
        delta_x[i] = x[i] - projection_plus[i];
        x[i] = projection_plus[i];
      }

#pragma omp for
      for (int i = 0; i < m; ++i) {
        g[i] = 0.0;
        for (int j = 0; j < n; ++j)
          g[i] += A[i][j] * x[j];
        g[i] -= b[i];
      }
    }
  } while (GetVectorNorm(delta_x) > external_tolerance);

  for (int i = 0; i < n; ++i)
    (*result)[i] = x[i];
}

void Newton::Initialize(const Matrix& A,
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

  m_.resize(m);
  r0_.resize(m);
  r_.resize(m);
  r0_tilded_.resize(m);
  r_tilded_.resize(m);
  p_.resize(m);
}

double Newton::GetVectorNorm(const Vector& v) const {
  int n = v.size();
  double global_norm = 0.0;
#pragma omp parallel default(shared)
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

#pragma omp parallel for default(shared)
  for (int i = 0; i < n; ++i) {
    m_[i] = H[i][i] * H[i][i];
    r0_[i] = g[i];
    p_[i] = r0_tilded_[i] = g[i] * m_[i];
    (*x)[i] = 0.0;
  }

  double u, v;
  while (true) {
#pragma omp parallel default(shared)
    {

#pragma omp single
      {
        u = 0.0;
        v = 0.0;
      }

#pragma omp for reduction(+:u, v)
      for (int i = 0; i < n; ++i) {
        r_[i] = 0.0;
        for (int j = 0; j < n; ++j)
          r_[i] += H[i][j] * p_[j];
        u += r0_tilded_[i] * r0_[i];
        v += p_[i] * r_[i];
      }

      double alpha = -u / v;

#pragma omp for
      for (int i = 0; i < n; ++i) {
        (*x)[i] -= alpha * p_[i];
        r_[i] = r_[i] * alpha + r0_[i];
      }
    }

    if (GetVectorNorm(r_) < eps)
      break;

#pragma omp parallel default(shared)
    {

#pragma omp single
      {
        u = 0.0;
        v = 0.0;
      }

#pragma omp for reduction(+:u, v)
      for (int i = 0; i < n; ++i) {
        r_tilded_[i] = r_[i] * m_[i];
        u += r_tilded_[i] * r_[i];
        v += r0_tilded_[i] * r0_[i];
      }

      double beta = u / v;

#pragma omp for
      for (int i = 0; i < n; ++i) {
        p_[i] = p_[i] * beta + r_tilded_[i];
        r0_[i] = r_[i];
        r0_tilded_[i] = r_tilded_[i];
      }
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

#pragma omp parallel for default(shared)
  for (int i = 0; i < n; ++i) {
    (*projection)[i] = 0.0;
    for (int j = 0; j < m; ++j)
      (*projection)[i] += tr_A[i][j] * p[j];
    (*projection)[i] += x[i] - beta * c[i];
    (*projection_plus)[i] = std::max(0.0, (*projection)[i]);
  }
}

}  // namespace math
