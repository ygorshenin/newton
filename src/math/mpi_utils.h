#ifndef MATH_MPI_UTILS_H_
#define MATH_MPI_UTILS_H_
#pragma once

#include <cstddef>

#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/vector.hpp"
#include "boost/scoped_array.hpp"

namespace math {

struct Range {
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
  void MasterComputeH(const Vector& d, Matrix& H);
  void MasterMultiplyMatrixByVector(const Vector& p, Vector& r);
  void MasterShutdown();
  void StartSlaveServer();

private:
  enum Commands {
    kCommandComputeH = 0,
    kCommandMultiplyMatrixByVector,
    kCommandStopSlaveServer,
  };

  enum Messages {
    kMessageRange = 0,
    kMessageResult,
  };

  void MasterDivideTask();
  void SlaveInitialize();
  void SlaveComputeH();
  void SlaveMultiplyMatrixByVector();

  void ComputeH();
  void MultiplyMatrixByVector();

  size_t num_rows_, num_cols_;
  boost::scoped_array<double> A_;
  boost::scoped_array<double> d_;
  boost::scoped_array<double> tr_T0_;
  boost::scoped_array<double> P_;
  boost::scoped_array<double> tr_P_;
  boost::scoped_array<double> H_;
  boost::scoped_array<double> tr_H_;

  boost::scoped_array<double> p_;
  boost::scoped_array<double> r_;

  std::vector<Range> row_ranges_;

  size_t row_from_, row_to_;
};

}  // namespace math

#endif  // MATH_MPI_UTILS_H_
