/*!
 * \file tl/sunway/op/fill.cc
 * \brief Scalar Sunway lowering for tl.fill.
 */

#include "op/fill.h"

#include "backend/common/target_utils.h"
#include "op/utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace sunway {

struct Fill {
  static Stmt Lower(const FillNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    (void)analyzer;
    ICHECK(IsFragmentBuffer(op.dst) || IsLocalBuffer(op.dst, true) ||
           IsSharedBuffer(op.dst))
        << "Sunway G0 fill supports only LDM-backed buffers, got scope `"
        << op.dst.scope() << "`";
    ICHECK(lower_args.thread_index.defined() &&
           lower_args.thread_bounds.defined())
        << "Sunway TileOps require a logical worker domain";

    Array<Var> loop_vars;
    Array<PrimExpr> indices;
    for (size_t axis = 0; axis < op.region.size(); ++axis) {
      Var loop_var("sunway_fill_i" + std::to_string(axis),
                   op.region[axis]->extent.dtype());
      loop_vars.push_back(loop_var);
      indices.push_back(op.region[axis]->min + loop_var);
    }

    Stmt body = BufferStore(op.dst, op.value, indices);
    for (int axis = static_cast<int>(op.region.size()) - 1; axis >= 0;
         --axis) {
      body = For(loop_vars[axis], 0, op.region[axis]->extent,
                 ForKind::kSerial, body);
    }
    return IfThenElse(
        EQ(lower_args.thread_index, lower_args.thread_bounds->min), body);
  }
};

} // namespace sunway

namespace {

bool MatchSunwayFillTarget(Target target) { return TargetIsSunway(target); }

bool RegisterSunwayFill() {
  RegisterFillImpl(FillImpl{
      "sunway.Fill",
      MatchSunwayFillTarget,
      sunway::Fill::Lower,
  });
  return true;
}

const bool sunway_fill_registered = RegisterSunwayFill();

} // namespace

} // namespace tl
} // namespace tvm
