#ifndef CUDA_ASSERT_OP_CUH
#define CUDA_ASSERT_OP_CUH

/* CUDA-specific includes */
#ifdef __CUDA_ARCH__
#include "cuda_runtime.h"
#endif /* __CUDA_ARCH__ */

/* macOS-specific defines */
#ifdef __APPLE__
#ifndef CUDA_ASSERT_USE_TRAP
#define CUDA_ASSERT_USE_TRAP
#endif /* !CUDA_ASSERT_USE_TRAP */
#endif /* __APPLE__ */

/* standard includes */
#ifdef __cplusplus
#include <cassert>
#include <cstdio>
#else /* !__cplusplus */
#include <assert.h>
#include <stdio.h>
#endif /* ?__cplusplus */

#ifdef CUDA_ASSERT
#error CUDA_ASSERT already defined
#else /* !CUDA_ASSERT */
#ifdef CUDA_ASSERT_USE_TRAP
#define CUDA_ASSERT(x)      \
  if (!(x)) {               \
    (void)printf("%s:%d:%s:\nblock: [%u,%u,%u],\n thread: [%u,%u,%u]\nAssertion `" #x "` failed.\n", __FILE__,__LINE__,__func__, blockId.x,blockId.x,blockIdx.z, threadIdx.x,threadIdx.y,threadIdx.z); \
    asm volatile ("trap;"); \
  }
#else /* !CUDA_ASSERT_USE_TRAP */
#define CUDA_ASSERT assert
#endif /* ?CUDA_ASSERT_USE_TRAP */
#endif /* ?CUDA_ASSERT */

#ifdef CUDA_ASSERT_STATUS
#error CUDA_ASSERT_STATUS already defined
#else /* !CUDA_ASSERT_STATUS */
#ifdef CUDA_ASSERT_USE_TRAP
#define CUDA_ASSERT_STATUS cudaErrorLaunchFailure
#else /* !CUDA_ASSERT_USE_TRAP */
#define CUDA_ASSERT_STATUS cudaErrorAssert
#endif /* ?CUDA_ASSERT_USE_TRAP */
#endif /* ?CUDA_ASSERT_STATUS */

#ifdef CUDA_ASSERT_EQ
#error CUDA_ASSERT_EQ already defined
#else /* !CUDA_ASSERT_EQ */
#define CUDA_ASSERT_EQ(a,b) CUDA_ASSERT((a)==(b))
#endif /* ?CUDA_ASSERT_EQ */

#ifdef CUDA_ASSERT_NE
#error CUDA_ASSERT_NE already defined
#else /* !CUDA_ASSERT_NE */
#define CUDA_ASSERT_NE(a,b) CUDA_ASSERT((a)!=(b))
#endif /* ?CUDA_ASSERT_NE */

#ifdef CUDA_ASSERT_LT
#error CUDA_ASSERT_LT already defined
#else /* !CUDA_ASSERT_LT */
#define CUDA_ASSERT_LT(a,b) CUDA_ASSERT((a)<(b))
#endif /* ?CUDA_ASSERT_LT */

#ifdef CUDA_ASSERT_LE
#error CUDA_ASSERT_LE already defined
#else /* !CUDA_ASSERT_LE */
#define CUDA_ASSERT_LE(a,b) CUDA_ASSERT((a)<=(b))
#endif /* ?CUDA_ASSERT_LE */

#ifdef CUDA_ASSERT_GT
#error CUDA_ASSERT_GT already defined
#else /* !CUDA_ASSERT_GT */
#define CUDA_ASSERT_GT(a,b) CUDA_ASSERT((a)>(b))
#endif /* ?CUDA_ASSERT_GT */

#ifdef CUDA_ASSERT_GE
#error CUDA_ASSERT_GE already defined
#else /* !CUDA_ASSERT_GE */
#define CUDA_ASSERT_GE(a,b) CUDA_ASSERT((a)>=(b))
#endif /* ?CUDA_ASSERT_GE */

#endif /* !CUDA_ASSERT_OP_CUH */
