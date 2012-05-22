#ifndef MATH_MPI_UTILS_H_
#define MATH_MPI_UTILS_H_
#pragma once

#include <cstddef>

#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/vector.hpp"
#include "boost/scoped_array.hpp"

namespace math {

struct Range {
  template<class Archive>
  void serialize(Archive& ar, unsigned long version) {
    ar & from;
    ar & to;
  }

  size_t from, to;
};

class State {
public:
  typedef boost::numeric::ublas::identity_matrix<double> IdentityMatrix;
  typedef boost::numeric::ublas::matrix<double> Matrix;
  typedef boost::numeric::ublas::vector<double> Vector;

  State();
  virtual ~State();

  void MasterInitialize(const Matrix& A);
  void MasterMultiply(const Vector& d, Matrix& H);
  void MasterShutdown();
  void StartSlaveServer();

private:
  enum Commands {
    kCommandMultiply = 0,
    kCommandStopSlaveServer,
  };

  enum Messages {
    kMessageRange = 0,
    kMessageResult,
  };

  void MasterDivideTask();
  void SlaveInitialize();
  void SlaveMultiply();
  void Multiply();

  size_t num_rows_, num_cols_;
  size_t from_, to_;
  boost::scoped_array<double> A_;
  boost::scoped_array<double> d_;
  boost::scoped_array<double> tr_T0_;
  boost::scoped_array<double> P_;
  boost::scoped_array<double> tr_P_;
  boost::scoped_array<double> tr_H_;

  std::vector<Range> ranges_;
};

}  // namespace math

#endif  // MATH_MPI_UTILS_H_
