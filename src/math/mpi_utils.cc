#include "math/mpi_utils.h"

#include <algorithm>

#include "boost/mpi.hpp"
#include "boost/mpi/collectives.hpp"
#include "boost/mpi/communicator.hpp"
#include "boost/mpi/environment.hpp"

namespace mpi = boost::mpi;

namespace math {

namespace {

const int kMPIRootRank = 0;

struct MatrixDimensions {
  template<class Archive>
  void serialize(Archive& ar, unsigned long version) {
    ar & num_rows;
    ar & num_cols;
  }

  size_t num_rows, num_cols;
};

void DivideTask(size_t n, size_t m, std::vector<Range>* ranges) {
  ranges->resize(m);
  if (m == 0)
    return;
  size_t large_size = (n + m - 1) / m, large_count = n % m;
  size_t small_size = n / m, small_count = m - (n % m);

  (void) small_count;

  assert(large_count + small_count == m);
  assert(large_size * large_count + small_size * small_count == n);

  (*ranges)[0].from = 0;
  (*ranges)[0].to = large_size;
  for (size_t i = 1; i < large_count; ++i) {
    (*ranges)[i].from = (*ranges)[i - 1].to;
    (*ranges)[i].to = (*ranges)[i].from + large_size;
  }
  for (size_t i = std::max(static_cast<size_t>(1), large_count);
       i < m;
       ++i) {
    (*ranges)[i].from = (*ranges)[i - 1].to;
    (*ranges)[i].to = (*ranges)[i].from + small_size;
  }
}

}  // namespace

State::State(): num_rows_(0), num_cols_(0) {
}

State::~State() {
}

void State::MasterInitialize(const Matrix& A) {
  mpi::communicator world;
  assert(world.rank() == kMPIRootRank);

  num_rows_ = A.size1();
  num_cols_ = A.size2();

  MatrixDimensions dimensions;
  dimensions.num_rows = num_rows_;
  dimensions.num_cols = num_cols_;
  mpi::broadcast(world, dimensions, kMPIRootRank);

  const size_t num_elems = num_rows_ * num_cols_;
  A_.reset(new double[num_elems]);
  d_.reset(new double[num_cols_]);

  size_t index = 0;
  for (size_t i = 0; i < num_rows_; ++i) {
    for (size_t j = 0; j < num_cols_; ++j)
      A_[index++] = A(i, j);
  }
  mpi::broadcast(world, A_.get(), num_elems, kMPIRootRank);

  DivideTask(num_rows_, world.size(), &ranges_);
  from_ = ranges_[0].from;
  to_ = ranges_[0].to;

  std::vector<mpi::request> requests;
  for (int i = 1; i < world.size(); ++i)
    requests.push_back(world.isend(i, kMessageRange, ranges_[i]));
  mpi::wait_all(requests.begin(), requests.end());

  tr_T0_.reset(new double[(to_ - from_) * num_cols_]);

  P_.reset(new double[num_rows_ * (to_ - from_)]);
  tr_P_.reset(new double[(to_ - from_) * num_rows_]);
  tr_H_.reset(new double[num_rows_ * num_rows_]);
}

void State::MasterMultiply(const Vector& d, Matrix& H) {
  mpi::communicator world;
  assert(H.size1() == num_rows_);
  assert(H.size2() == num_rows_);
  assert(world.rank() == kMPIRootRank);
  assert(d.size() == num_cols_);

  Commands command = kCommandMultiply;
  mpi::broadcast(world, command, kMPIRootRank);

  std::copy(d.begin(), d.end(), d_.get());
  mpi::broadcast(world, d_.get(), num_cols_, kMPIRootRank);
  Multiply();

  std::copy(tr_P_.get(), tr_P_.get() + (to_ - from_) * num_rows_, tr_H_.get());
  std::vector<mpi::request> requests;
  size_t offset = (to_ - from_) * num_rows_;
  for (int i = 1; i < world.size(); ++i) {
    size_t block_size = (ranges_[i].to - ranges_[i].from) * num_rows_;
    requests.push_back(world.irecv(i, kMessageResult, tr_H_.get() + offset, block_size));
    offset += block_size;
  }
  mpi::wait_all(requests.begin(), requests.end());
  for (size_t i = 0; i < num_rows_; ++i) {
    for (size_t j = 0; j < num_rows_; ++j) {
      H(i, j) = tr_H_[j * num_rows_ + i];
    }
    H(i, i) += 1.0;
  }
}

void State::MasterShutdown() {
  mpi::communicator world;
  assert(world.rank() == kMPIRootRank);

  Commands command = kCommandStopSlaveServer;
  mpi::broadcast(world, command, kMPIRootRank);
}

void State::StartSlaveServer() {
  mpi::communicator world;
  assert(world.rank() != kMPIRootRank);

  SlaveInitialize();

  Commands command;
  bool is_run = true;
  while (is_run) {
    mpi::broadcast(world, command, kMPIRootRank);
    switch (command) {
    case kCommandMultiply:
      SlaveMultiply();
      break;
    case kCommandStopSlaveServer:
      is_run = false;
      break;
    }
  }
}

void State::SlaveInitialize() {
  mpi::communicator world;
  assert(world.rank() != kMPIRootRank);

  MatrixDimensions dimensions;
  mpi::broadcast(world, dimensions, kMPIRootRank);
  num_rows_ = dimensions.num_rows;
  num_cols_ = dimensions.num_cols;

  const size_t num_elems = num_rows_ * num_cols_;

  A_.reset(new double[num_elems]);
  d_.reset(new double[num_cols_]);

  mpi::broadcast(world, A_.get(), num_elems, kMPIRootRank);

  Range range;
  world.recv(kMPIRootRank, kMessageRange, range);
  from_ = range.from;
  to_ = range.to;

  tr_T0_.reset(new double[(to_ - from_) * num_cols_]);

  P_.reset(new double[num_rows_ * (to_ - from_)]);
  tr_P_.reset(new double[(to_ - from_) * num_rows_]);
}

void State::SlaveMultiply() {
  mpi::communicator world;
  assert(world.rank() != kMPIRootRank);

  mpi::broadcast(world, d_.get(), num_cols_, kMPIRootRank);
  Multiply();
  world.send(kMPIRootRank, kMessageResult, tr_P_.get(), (to_ - from_) * num_rows_);
}

void State::Multiply() {
  size_t tr_T0_index = 0;
  for (size_t i = 0; i < (to_ - from_); ++i) {
    size_t A_base_index = (from_ + i) * num_cols_;
    for (size_t j = 0; j < num_cols_; ++j)
      tr_T0_[tr_T0_index++] = A_[A_base_index + j] * d_[j];
  }

  size_t P_index = 0;
  for (size_t i = 0; i < num_rows_; ++i) {
    for (size_t j = 0; j < (to_ - from_); ++j) {
      P_[P_index] = 0.0;
      for (size_t k = 0; k < num_cols_; ++k)
	P_[P_index] += A_[i * num_cols_ + k] * tr_T0_[j * num_cols_ + k];
      ++P_index;
    }
  }

  size_t tr_P_index = 0;
  for (size_t j = 0; j < (to_ - from_); ++j) {
    for (size_t i = 0; i < num_rows_; ++i)
      tr_P_[tr_P_index++] = P_[i * (to_ - from_) + j];
  }
}

}  // namespace math
