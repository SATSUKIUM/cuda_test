#pragma once

#include <cuda_runtime.h>

#if CUDA_VERSION >= 12080

#include <cuda/cmath>

namespace cuda_compat {

using cuda::ceil_div;

}

#else

namespace cuda_compat {

template <typename T, typename U>
__host__ __device__
constexpr auto ceil_div(T a, U b)
{
    return (a + b - 1) / b;
}

}

#endif