/* Shared by the metal4/ kernels, which are concatenated into one source after
 * this file (Makefile, METAL4_SRCS) and compiled as their own library at
 * Metal 4 (qwasar_metal.m, qw_tmm_library). */

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;
