/*!
 * \file tl/sunway/op/copy.cc
 * \brief Sunway two-dimensional DMA descriptors for tl.copy.
 */

#include "op/copy.h"

#include "backend/common/target_utils.h"
#include "op/utils.h"

#include <tvm/tirx/builtin.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace sunway {

namespace {

bool IsLDMBuffer(const Buffer &buffer) {
  return IsSharedBuffer(buffer) || IsFragmentBuffer(buffer) ||
         IsLocalBuffer(buffer, true);
}

PrimExpr RowStride(const Buffer &buffer) {
  if (!buffer->strides.empty()) {
    return buffer->strides[0];
  }
  return buffer->shape[1];
}

Stmt GuardWorkerZero(Stmt body, const LowerArgs &lower_args) {
  ICHECK(lower_args.thread_index.defined() &&
         lower_args.thread_bounds.defined())
      << "Sunway TileOps require a logical worker domain";
  return IfThenElse(
      EQ(lower_args.thread_index, lower_args.thread_bounds->min), body);
}

} // namespace

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op,
                               const LayoutInferArgs &layout_args,
                               InferLevel level) {
    (void)op;
    (void)layout_args;
    (void)level;
    return {};
  }

  static Stmt Lower(const CopyNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    ICHECK_EQ(op.src_range.size(), 2)
        << "Sunway G0 copy requires a rank-2 source, got " << op.src->name;
    ICHECK_EQ(op.dst_range.size(), 2)
        << "Sunway G0 copy requires a rank-2 destination, got "
        << op.dst->name;
    ICHECK(op.src->dtype == op.dst->dtype)
        << "Sunway G0 copy requires identical source and destination dtypes";
    ICHECK(analyzer->CanProveEqual(op.src_range[0]->extent,
                                  op.dst_range[0]->extent) &&
           analyzer->CanProveEqual(op.src_range[1]->extent,
                                   op.dst_range[1]->extent))
        << "Sunway G0 copy requires identical two-dimensional regions";

    if (!op.src->strides.empty()) {
      ICHECK(analyzer->CanProveEqual(op.src->strides[1], 1))
          << "Sunway G0 copy requires a compact source innermost dimension";
    }
    if (!op.dst->strides.empty()) {
      ICHECK(analyzer->CanProveEqual(op.dst->strides[1], 1))
          << "Sunway G0 copy requires a compact destination innermost dimension";
    }

    const bool is_get = IsGlobalBuffer(op.src) && IsLDMBuffer(op.dst);
    const bool is_put = IsLDMBuffer(op.src) && IsGlobalBuffer(op.dst);
    ICHECK(is_get || is_put)
        << "Sunway G0 copy supports only global-to-LDM or LDM-to-global "
           "transfers, got source scope `"
        << op.src.scope() << "` and destination scope `" << op.dst.scope()
        << "`";

    const int element_bytes = op.src->dtype.bytes() * op.src->dtype.lanes();
    PrimExpr source = MakeAccessPtrFromRegion(
        BufferRegion(op.src, op.src_range), kAccessRead, true);
    PrimExpr destination = MakeAccessPtrFromRegion(
        BufferRegion(op.dst, op.dst_range), kAccessWrite, true);
    PrimExpr rows = op.src_range[0]->extent;
    PrimExpr row_bytes = op.src_range[1]->extent * element_bytes;
    PrimExpr source_stride_bytes = RowStride(op.src) * element_bytes;
    PrimExpr destination_stride_bytes = RowStride(op.dst) * element_bytes;
    const char *name = is_get ? "tilelang_sunway_dma_get_2d"
                              : "tilelang_sunway_dma_put_2d";

    Array<PrimExpr> arguments{
        StringImm(name),          source,      destination, rows,
        row_bytes,                source_stride_bytes,
        destination_stride_bytes,
    };
    Stmt descriptor = Evaluate(
        Call(DataType::Int(32), builtin::call_extern(), arguments));
    return GuardWorkerZero(descriptor, lower_args);
  }
};

} // namespace sunway

namespace {

bool MatchSunwayCopyTarget(Target target) { return TargetIsSunway(target); }

bool RegisterSunwayCopy() {
  RegisterCopyImpl(CopyImpl{
      "sunway.Copy",
      MatchSunwayCopyTarget,
      100,
      sunway::Copy::InferLayout,
      sunway::Copy::Lower,
  });
  return true;
}

const bool sunway_copy_registered = RegisterSunwayCopy();

} // namespace

} // namespace tl
} // namespace tvm
