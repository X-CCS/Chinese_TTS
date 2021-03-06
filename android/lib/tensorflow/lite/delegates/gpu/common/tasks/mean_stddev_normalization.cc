/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/delegates/gpu/common/tasks/mean_stddev_normalization.h"

#include <string>

#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

namespace {

std::string GetVectorReduceCode() {
  return R"(float reduce_vector(float4 v) {
  return dot(v, (float4)(1.0f));
})";
}

std::string GetReduceCode() {
  // If it is supported, use the built-in work_group_reduce_add function.
  // Otherwise, implement a reduction using __local memory.

  // In the reduction step add upper half of the still-to-be-summed vector to
  // the lower half, while taking care of odd sizes and rounding. E.g.:
  // Number of items still to be summed before: 5
  // Local memory before: [a, b, c, d, e];
  // Local memory after: [a+d, b+e, c, d, e];
  // Threads doing work: id < 2 = floor(5/2)
  // Offset to the added items: 3 = ceil(5/2)
  // Number of items still to be summed after: 3 = ceil(5/2)
  return R"(
#if (__OPENCL_C_VERSION__ >= 200) && (__OPENCL_C_VERSION__ < 300) && \
  !defined(__opencl_c_work_group_collective_functions)
  #define __opencl_c_work_group_collective_functions 1
#endif

#ifdef __opencl_c_work_group_collective_functions
#define local_reduce(item, tmp) work_group_reduce_add(item)
#else  // !defined(__opencl_c_work_group_collective_functions)
float local_reduce(float item, __local float* tmp) {
  const int local_id = get_local_id(0);
  tmp[local_id] = item;
  barrier(CLK_LOCAL_MEM_FENCE);
  // The number of items still need to be summed
  int reduction_size = get_local_size(0);
  while (reduction_size > 1) {
    const int active_thread_limit = reduction_size / 2;
    const int offset = (reduction_size + 1) / 2;
    if (local_id < active_thread_limit) {
      item += tmp[local_id + offset];
      tmp[local_id] = item;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    reduction_size = offset;
  }
  return tmp[0];
}
#endif  // defined(__opencl_c_work_group_collective_functions)
)";
}

std::string GetFilterCode() {
  return R"(
float4 filter_outside_tensor(float4 x, int num_channels, int slice) {
  return select(x, (float4)(0.0f), slice * 4 + (int4)(0, 1, 2, 3) >= num_channels);
}
)";
}
}  // namespace

MeanStdDevNormalization::MeanStdDevNormalization(const OperationDef& definition,
                                                 const GpuInfo& gpu_info,
                                                 const int tensor_slices)
    : GPUOperation(definition) {
  // The kernel code does not inherently need a fixed size, but in order to not
  // hardcode the __local array's size for the reductions, we would need to pass
  // that size to the kernel at runtime, and that is currently not supported.
  // For now, fix workgroup size to the biggest supported by the device, but not
  // larger than the number of tensor slices.
  int desired_work_group_size =
      std::min(tensor_slices, gpu_info.GetMaxWorkGroupSizeForX());
  if (gpu_info.IsMali()) {
    // Don't use more than 64 work items per work group on ARM Mali. They
    // implement local memory using the global memory, larger workgroups have
    // severe performance penalty.
    desired_work_group_size = 64;
  }
  if (gpu_info.IsAdreno()) {
    AdrenoInfo info = gpu_info.adreno_info;
    if (info.IsAdreno3xx()) {
      if (info.adreno_gpu == AdrenoGpu::kAdreno320 ||
          info.adreno_gpu == AdrenoGpu::kAdreno330) {
        desired_work_group_size = 128;
      } else {
        desired_work_group_size = 64;
      }
    } else if (info.IsAdreno4xx()) {
      if (info.adreno_gpu == AdrenoGpu::kAdreno430) {
        desired_work_group_size = 256;
      } else {
        desired_work_group_size = 128;
      }
    } else if (info.IsAdreno5xx()) {
      if (info.adreno_gpu == AdrenoGpu::kAdreno530 ||
          info.adreno_gpu == AdrenoGpu::kAdreno540) {
        desired_work_group_size = 256;
      } else {
        desired_work_group_size = 128;
      }
    }
  }
  if (gpu_info.IsPowerVR()) {
    desired_work_group_size = 64;
  }
  while (desired_work_group_size >= tensor_slices * 2) {
    desired_work_group_size /= 2;
  }
  work_group_size_.x = desired_work_group_size;
  work_group_size_.y = 1;  // Required
  work_group_size_.z = 1;  // Required
  code_ = GetNormalizationCode();
  if (gpu_info.IsCL30OrHigher()) {
    compiler_options_.push_back(CompilerOptions::kCl30);
  } else if (gpu_info.IsCL20OrHigher()) {
    compiler_options_.push_back(CompilerOptions::kCl20);
  }
}

std::string MeanStdDevNormalization::GetNormalizationCode() {
  AddSrcTensor("src_tensor", definition_.src_tensors[0]);
  AddDstTensor("dst_tensor", definition_.dst_tensors[0]);

  std::string c;
  c += GetVectorReduceCode();
  c += GetReduceCode();
  c += GetFilterCode();
  c += "__attribute__((reqd_work_group_size(" +
       std::to_string(work_group_size_.x) + ", 1, 1)))\n";
  c += R"(__kernel void main_function($0) {
#ifndef __opencl_c_work_group_collective_functions
  __local float tmp[)" +
       std::to_string(work_group_size_.x) + R"(];
#endif
  const int B = get_global_id(1);
  // Calculate the total sum of the input tensor.
  // First, get a local sum of input[local_id_x + N*local_size_x] for all N.
  float4 private_sum4 = (float4)(0.0f);
  for (int S = get_local_id(0); S < args.src_tensor.Slices(); S += get_local_size(0)) {
    const float4 t = args.src_tensor.Read<float>(0, 0, S, B);
    private_sum4 += filter_outside_tensor(t, args.src_tensor.Channels(), S);
  }
  // Reduce the vector to a single float and do a workgroup reduce.
  const float private_sum = reduce_vector(private_sum4);
  const float sum = local_reduce(private_sum, tmp);
  // Calculate the mean
  const float mean = sum / args.src_tensor.Channels();
  // Calculate the squared sum of the difference from the mean.
  float4 private_sum_diff_sq4 = (float4)(0.0f);
  for (int S = get_local_id(0); S < args.src_tensor.Slices(); S += get_local_size(0)) {
    const float4 t = args.src_tensor.Read<float>(0, 0, S, B);
    const float4 diff = filter_outside_tensor(t - mean, args.src_tensor.Channels(), S);
    // sum_diff_sq += diff??
    private_sum_diff_sq4 = mad(diff, diff, private_sum_diff_sq4);
  }
  // Reduce
  const float private_sum_diff_sq = reduce_vector(private_sum_diff_sq4);
  const float sum_diff_sq = local_reduce(private_sum_diff_sq, tmp);
  // Calculate 1/stddev (with the 'regulazing constant' as in tensor_utils.cc)
  const float variance = sum_diff_sq / args.src_tensor.Channels();
  const float stddev_inv = native_rsqrt(variance + 1.0e-8f);
  // Calculate (t-mean)/stddev for each element
  for (int S = get_local_id(0); S < args.src_tensor.Slices(); S += get_local_size(0)) {
    const float4 t = args.src_tensor.Read<float>(0, 0, S, B);
    FLT4 result = TO_FLT4((t - mean) * stddev_inv);
    args.dst_tensor.Write(result, 0, 0, S, B);
  }
})";
  return c;
}

int3 MeanStdDevNormalization::GetGridSize() const {
  // To avoid dealing with global reductions, we restrict the grid size to the
  // work group size in the first dimension.
  const int grid_x = work_group_size_.x;
  const int grid_y = src_[0]->Batch();
  const int grid_z = 1;
  return int3(grid_x, grid_y, grid_z);
}

MeanStdDevNormalization CreateMeanStdDevNormalization(
    const OperationDef& definition, const GpuInfo& gpu_info,
    const int tensor_slices) {
  return MeanStdDevNormalization(definition, gpu_info, tensor_slices);
}

}  // namespace gpu
}  // namespace tflite
