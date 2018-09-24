#pragma once

#include "cuda_runtime_api.h"

namespace kinfu
{
    namespace gpu
    {
        void error(const char *error_string, const char *file, const int line, const char *func);
    }
}

#if defined(__GNUC__)
    #define cudaSafeCall(expr)  kinfu::gpu::___cudaSafeCall(expr, __FILE__, __LINE__, __func__)
#else /* defined(__CUDACC__) || defined(__MSVC__) */
    #define cudaSafeCall(expr)  kinfu::gpu::___cudaSafeCall(expr, __FILE__, __LINE__)
#endif

namespace kinfu
{
    namespace gpu
    {
        static inline void ___cudaSafeCall(cudaError_t err, const char *file, const int line, const char *func = "")
        {
            if (cudaSuccess != err)
                error(cudaGetErrorString(err), file, line, func);
        }        

        static inline int divUp(int total, int grain) { return (total + grain - 1) / grain; }
    }

    namespace device
    {
        using kinfu::gpu::divUp;
    }
}
