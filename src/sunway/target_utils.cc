/*!
 * \file tl/sunway/target_utils.cc
 * \brief Sunway target attribute helpers.
 */

#include "sunway/target_utils.h"

#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace tl {

bool TargetIsSunway(Target target) {
  for (const auto &key : target->keys) {
    if (key == "sunway") {
      return true;
    }
  }
  return false;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.TargetIsSunway",
                        [](Target target) { return TargetIsSunway(target); });
}

} // namespace tl
} // namespace tvm
