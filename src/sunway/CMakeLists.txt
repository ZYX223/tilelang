# Sunway target matching and TileOp lowering have no native SDK dependency.
# Generated MPE/CPE sources are compiled separately by the AOT toolchain.

file(GLOB TILE_LANG_SUNWAY_SRCS
  src/sunway/op/*.cc
  src/sunway/target_utils.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_SUNWAY_SRCS})
