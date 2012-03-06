#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

namespace ublas = boost::numeric::ublas;
using std::cout;
using std::cerr;
using std::endl;

// Solves A * x = b via preconditioned conjugate gradient method.
template<typename T>
inline void GetMaximizationDirection(const ublas::matrix<T>& A,
				     const ublas::vector<T>& b,
				     T epsilon,
				     ublas::vector<T>* x) {
  typedef typename ublas::matrix<T>::size_type size_type;

  const T kZero = static_cast<T>(0);
  const T kOne = static_cast<T>(1);
  const size_type n = A.size1();

  ublas::zero_matrix<T> M(n, n);
  for (size_type i = 0; i < n; ++i)
    M(i, i) = A(i, i) * A(i, i);

  std::fill(x->begin(), x->end(), kZero);

  ublas::vector<T> r0(b), r0_tilded(n);
  ublas::vector<T> r(n), r_tilded(n);
  for (size_type i = 0; i < n; ++i)
    r0_tilded(i) = r0(i) * M(i, i);

  ublas::vector<T> p(r0_tilded);

  while (true) {
    double alpha = -ublas::prod(r0_tilded, r0) / ublas::prod(p, prod(A, p));
    *x -= alpha * p;
    r = r0 + alpha * ublas::prod(A, p);

    if (abs(*std::max_element(r0.begin(), r0.end())) < epsilon)
      break;

    for (size_type i = 0; i < n; ++i)
      r_tilded(i) = r(i) * M(i, i);
    double beta = ublas::prod(r_tilded, r) / ublas::prod(r0_tilded, r0);
    
    p *= beta;
    p += r_tilded;

    r0 = r;
    r0_tilded = r_tilded;
  }
}

template<typename T>
inline void ComputeProjection(const ublas::matrix<T>& tr_A,
			      const ublas::vector<T>& c,
			      const ublas::vector<T>& x,
			      const ublas::vector<T>& p,
			      T beta,
			      const ublas::vector<T>* projection,
			      const ublas::vector<T>* projection_plus) {
  *projection = x + ublas::prod(tr_A, p) - beta * c;
  for (typename ublas::vector<T>::size_type i = 0; i < projection->size(); ++i)
    (*projection_plus)(i) = std::max(static_cast<T>(0), (*projection)(i));
}

// Solves forward linear programming problem tr(c) * x -> min, where x
// is an n-dimensional vector, such that A * x = b, and all components
// of x is non-negative numbers.
template<typename T>
void NewtonMethod(const ublas::matrix<T>& A,
		  const ublas::vector<T>& b,
		  const ublas::vector<T>& c,
		  const ublas::vector<T>& x,
		  const ublas::vector<T>& p,
		  T beta,
		  T external_tolerance,
		  T internal_tolerance) {
  typedef typename ublas::matrix<T>::size_type size_type;

  const T kZero = static_cast<T>(0);
  const T kOne = static_cast<T>(1);

  const size_type m = A.size1(), n = A.size2();
  assert(b.size() == m);
  assert(c.size() == n);
  assert(x.size() == n);
  assert(p.size() == m);

  const ublas::identity_matrix<T> I(m, m);

  ublas::matrix<T> tr_A = ublas::trans(A);
  ublas::zero_matrix<T> D(n, n);
  ublas::vector<T> projection(n), projection_plus(n);
  ublas::matrix<T> H(m, m);
  ublas::vector<T> G(m);
  ublas::vector<T> delta_x(n);
  ublas::vector<T> delta_p(m);

  do {
    do {
      ComputeProjection(tr_A, c, x, p, beta, &projection, &projection_plus);
      for (size_type i = 0; i < n; ++i)
	D(i, i) = projection(i) > kZero ? kOne : kZero;
      G = A * projection_plus - b;
      H = I + A * D * tr_A;
      GetMaximizationDirection(H, G, internal_tolerance, &delta_p);
      p -= delta_p;
    } while (ublas::norm_2(delta_p) < internal_tolerance);
    ComputeProjection(tr_A, c, x, p, beta, &projection, &projection_plus);
    delta_x = x - projection_plus;
    x = projection_plus;
  } while (ublas::norm_2(delta_x) < external_tolerance);
}

int main(int argc, char** argv) {
  std::ios_base::sync_with_stdio(0);
  return 0;
}
