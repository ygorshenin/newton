#include <iostream>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

#include "math/newton.h"

namespace ublas = boost::numeric::ublas;
using std::cout;
using std::cerr;
using std::endl;

int main(int argc, char** argv) {
  std::ios_base::sync_with_stdio(0);

  math::Newton newton;
  return 0;
}
