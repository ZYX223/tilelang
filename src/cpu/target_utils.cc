/*!
 * \file tl/cpu/target_utils.cc
 * \brief CPU target attribute helpers.
 */

#include "cpu/target_utils.h"

#include <tvm/ffi/reflection/registry.h>

#include "dlpack/dlpack.h"

namespace tvm {
namespace tl {

bool TargetIsCPU(Target target) {
  if (target->GetTargetDeviceType() != kDLCPU) {
    return false;
  }
  for (const auto &key : target->keys) {
    if (key == "sunway") {
      return false;
    }
  }
  return true;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.TargetIsCPU",
                        [](Target target) { return TargetIsCPU(target); });
}

} // namespace tl
} // namespace tvm
