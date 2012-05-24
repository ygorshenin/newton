#include <cmath>
#include <iostream>
#include <tr1/memory>
#include <valarray>

#include "math/newton.h"

namespace tr1 = std::tr1;

using tr1::shared_ptr;

typedef std::valarray<double> Vector;
typedef std::valarray<std::valarray<double> > Matrix;

namespace {

void ReadProblemDescription(std::istream& is,
			    size_t *m, size_t *n,
			    shared_ptr<Matrix>* A,
			    shared_ptr<Vector>* b,
			    shared_ptr<Vector>* c,
			    double* beta,
			    double* external_tolerance,
			    double* internal_tolerance) {
  is >> *m >> *n;
  
  A->reset(new Matrix(Vector(*n), *m));
  for (size_t i = 0; i < *m; ++i)
    for (size_t j = 0; j < *n; ++j)
      is >> (*A->get())[i][j];

  b->reset(new Vector(*m));
  for (size_t i = 0; i < *m; ++i)
    is >> (*b->get())[i];

  c->reset(new Vector(*n));
  for (size_t i = 0; i < *n; ++i)
    is >> (*c->get())[i];

  is >> *beta >> *external_tolerance >> *internal_tolerance;
}

}  // namespace

int main() {
  std::ios_base::sync_with_stdio(0);

  size_t m, n;
  shared_ptr<Matrix> A;
  shared_ptr<Vector> b, c;
  double beta, external_tolerance, internal_tolerance;

  std::clog << "Reading problem description..." << std::endl;
  ReadProblemDescription(std::cin,
			 &m, &n,
			 &A, &b, &c,
			 &beta,
			 &external_tolerance,
			 &internal_tolerance);

  math::Newton newton;

  Vector x0(0.0, n), p0(1.0, m);

  Vector result(n);
  std::clog << "Solving linear system..." << std::endl;
  newton.SolveLinearSystem(*A, *b, *c,
			   x0, p0,
			   beta,
			   external_tolerance,
			   internal_tolerance,
			   &result);
  for (size_t i = 0; i < result.size(); ++i)
    std::cout << (int) round(result[i]) << ' ';
  std::cout << std::endl;

  int f = 0;
  for (size_t i = 0; i < result.size(); ++i)
    f += (int) round((*c)[i]) * (int) round(result[i]);
  std::cout << f << std::endl;
  return 0;
}
