#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

namespace po = boost::program_options;
namespace ublas = boost::numeric::ublas;
using std::cout;
using std::cerr;
using std::endl;

// Tolerance for Newton's external iterations.
const double kDefaultExternalTolerance = 1e-4;
// Tolerance for Newton's internal iterations;
const double kDefaultInternalTolerance = 1e-4;

double FLAGS_external_tolerance;
double FLAGS_internal_tolerance;

bool ParseCommandLine(int argc, char** argv) {
  po::options_description description("Allowed options");
  description.add_options()
    ("help", "produce help message")
    ("external_tolerance",
     po::value<double>(&FLAGS_external_tolerance)->default_value(
         kDefaultExternalTolerance),
     "tolerance for Netwton's external iterations")
    ("internal_tolerance",
     po::value<double>(&FLAGS_internal_tolerance)->default_value(
         kDefaultInternalTolerance),
     "tolerance for Netton's internal iterations");
  po::variables_map variables_map;
  po::store(po::parse_command_line(argc, argv, description), variables_map);
  po::notify(variables_map);

  if (variables_map.count("help")) {
    cerr << description << endl;
    return false;
  }
  return true;
}

template<typename T>
inline void ComputeProjection(const ublas::matrix<T>& tr_A,
			      const ublas::vector<T>& c,
			      const ublas::vector<T>& x,
			      const ublas::vector<T>& p,
			      double beta,
			      const ublas::vector<T>* projection,
			      const ublas::vector<T>* projection_plus) {
  *projection = x + ublas::prod(tr_A, p) - beta * c;
  for (ublas::vector<T>::size_type i = 0; i < projection->size(); ++i)
    (*projection_plus)(i) = std::max(kZero, (*projection)(i));
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
		  double beta,
		  double external_tolerance,
		  double internal_tolerance) {
  const kZero = static_cast<T>(0);
  const kOne = static_cast<T>(1);

  const ublas::identity_matrix<T> I(m, m);

  const ublas::matrix<T>::size_type m = A.size1(), n = A.size2();
  assert(b.size() == m);
  assert(c.size() == n);
  assert(x.size() == n);
  assert(p.size() == m);

  ublas::matrix<T> tr_A = ublas::trans(A);
  ublas::zero_matrix<T> D(n, n);
  ublas::vector<T> projection(n), projection_plus(n);
  ublas::vector<T> G(m);
  ublas::matrix<T> H(m, m);
  ublas::vector<T> delta_x(n);
  ublas::vector<T> delta_p(m);

  do {
    do {
      ComputeProjection(tr_A, c, x, p, beta, &projection, &projection_plus);
      for (ublas::matrix<T>::size_type i = 0; i < n; ++i)
	D(i, i) = projection(i) > kZero ? kOne : kZero;
      G = b - A * projection_plus;
      H = I + A * D * tr_A;
      // Compute delta_p as solution of SoLAE H * delta_p = -G.
      p -= delta_p;
    } while (ublas::norm_2(delta_p) < internal_tolerance);
    ComputeProjection(tr_A, c, x, p, beta, &projection, &projection_plus);
    delta_x = x - projection_plus;
    x = projection_plus;
  } while (ublas::norm_2(delta_x) < external_tolerance);
}

int main(int argc, char** argv) {
  if (!ParseCommandLine(argc, argv))
    return 1;
  return 0;
}
