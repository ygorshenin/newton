#include <algorithm>
#include <cstddef>
#include <iostream>
#include <iterator>

#include "boost/numeric/ublas/io.hpp"
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/vector.hpp"
#include "boost/scoped_ptr.hpp"
#include "math/newton.h"

namespace ublas = boost::numeric::ublas;
using boost::scoped_ptr;
using std::cin;
using std::cout;
using std::endl;

typedef boost::numeric::ublas::matrix<double> Matrix;
typedef boost::numeric::ublas::vector<double> Vector;
typedef Matrix::size_type size_type;

void ReadProblemDescription(std::istream& is,
			    size_type *m, size_type *n,
			    scoped_ptr<Matrix>* A,
			    scoped_ptr<Vector>* b,
			    scoped_ptr<Vector>* c,
			    double* beta,
			    double* external_tolerance,
			    double* internal_tolerance) {
  is >> *m >> *n;
  
  A->reset(new Matrix(*m, *n));
  for (size_type i = 0; i < *m; ++i)
    for (size_type j = 0; j < *n; ++j)
      cin >> (*A->get())(i, j);

  b->reset(new Vector(*m));
  for (size_type i = 0; i < *m; ++i)
    cin >> (*b->get())(i);

  c->reset(new Vector(*n));
  for (size_type i = 0; i < *n; ++i)
    cin >> (*c->get())(i);

  is >> *beta >> *external_tolerance >> *internal_tolerance;
}

int main(int argc, char** argv) {
  std::ios_base::sync_with_stdio(0);

  size_type m, n;
  scoped_ptr<Matrix> A;
  scoped_ptr<Vector> b, c;
  double beta, external_tolerance, internal_tolerance;

  ReadProblemDescription(cin,
			 &m, &n,
			 &A, &b, &c,
			 &beta,
			 &external_tolerance,
			 &internal_tolerance);

  math::Newton newton;

  Vector x0(n), p0(m);
  std::fill(x0.begin(), x0.end(), 0.0);
  std::fill(p0.begin(), p0.end(), 1.0);

  Vector result(n);
  newton.SolveLinearSystem(*A, *b, *c,
			   x0, p0,
			   beta,
			   external_tolerance,
			   internal_tolerance,
			   &result);
  // Display optiomal values.
  std::copy(result.begin(),
	    result.end(),
	    std::ostream_iterator<double>(cout, " "));
  cout << endl;
  // Display optiomal function value.
  cout << ublas::inner_prod(*c, result) << endl;
  return 0;
}
