/*!
 * \file tl/sunway/op/gemm.cc
 * \brief Sunway implementation selection for tl.gemm.
 */

#include "op/gemm.h"

#include "backend/common/target_utils.h"

namespace tvm {
namespace tl {

using namespace ffi;

namespace sunway {

namespace {

constexpr const char *kSunwayScalar = "sunway.scalar";

} // namespace

struct Gemm {
  static String SelectInst(const GemmNode &op, int block_size, Target target) {
    (void)op;
    (void)block_size;
    (void)target;
    return kSunwayScalar;
  }

  static std::pair<int, int>
  ComputeWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                       int block_size, Target target, String gemm_inst) {
    (void)M;
    (void)N;
    (void)block_size;
    (void)target;
    (void)gemm_inst;
    policy.m_warp = 1;
    policy.n_warp = 1;
    return {1, 1};
  }

  static bool ReuseExistingSharedLayout(String gemm_inst) {
    (void)gemm_inst;
    return false;
  }
};

} // namespace sunway

namespace {

bool MatchSunwayGemmTarget(Target target) { return TargetIsSunway(target); }

bool RegisterSunwayGemm() {
  RegisterGemmImpl(GemmImpl{
      "sunway.Gemm",
      MatchSunwayGemmTarget,
      sunway::Gemm::SelectInst,
      sunway::Gemm::ComputeWarpPartition,
      sunway::Gemm::ReuseExistingSharedLayout,
  });
  return true;
}

const bool sunway_gemm_registered = RegisterSunwayGemm();

} // namespace

} // namespace tl
} // namespace tvm
