#ifndef CUDA_TEST_CUH
#define CUDA_TEST_CUH

/* TODO: With CUDA 8.0 on Linux x86_64:
   After a failed GPU assertion, ALL subsequent cudaDeviceSynchronize calls fail,
   even after cudaDeviceReset, with the error code cudaErrorDevicesUnavailable (46).
   The subsequent tests also fail, regardless of whether their assertions were false
   or true!  It seems that the recovery is to terminate the process, but after the
   first failure all others may be unreliable!  E.g., in the following run the test
   ADD_1_3_simple should have passed, but it failed:
$ ./run_000.sh 
Running main() from gtest_main.cc
[==========] Running 3 tests from 1 test case.
[----------] Global test environment set-up.
[----------] 3 tests from FloatxOperationsTest
[ RUN      ] FloatxOperationsTest.ADD_1_1_simple
[       OK ] FloatxOperationsTest.ADD_1_1_simple (1364 ms)
[ RUN      ] FloatxOperationsTest.ADD_1_2_simple
mix_000.cu:32: void <unnamed>::_cutest_FloatxOperationsTest_ADD_1_2_simple(): block: [0,0,0], thread: [0,0,0] Assertion `(double(a))!=(0.50000000000000000000)` failed.
mix_000.cu:25: Failure
Failed
error in teardown cudaDeviceSynchronize: 46
[  FAILED  ] FloatxOperationsTest.ADD_1_2_simple (394 ms)
[ RUN      ] FloatxOperationsTest.ADD_1_3_simple
mix_000.cu:37: Failure
Failed
error in teardown cudaDeviceSynchronize: 46
[  FAILED  ] FloatxOperationsTest.ADD_1_3_simple (0 ms)
[----------] 3 tests from FloatxOperationsTest (1758 ms total)

[----------] Global test environment tear-down
[==========] 3 tests from 1 test case ran. (1758 ms total)
[  PASSED  ] 1 test.
[  FAILED  ] 2 tests, listed below:
[  FAILED  ] FloatxOperationsTest.ADD_1_2_simple
[  FAILED  ] FloatxOperationsTest.ADD_1_3_simple

 2 FAILED TESTS
*/

#ifdef CUDA_TEST
#error CUDA_TEST already defined
#else /* !CUDA_TEST */
#define CUDA_TEST(test, test_case, dev, grid_dim, block_dim, shared_mem, stream)  \
  __device__ void _cutest_##test##_##test_case(void);                             \
  __global__ void _cukern_##test##_##test_case(void)                              \
  {                                                                               \
     _cutest_##test##_##test_case();                                              \
  }                                                                               \
  TEST(test, test_case)                                                           \
  {                                                                               \
     cudaError_t st = cudaSetDevice(dev);                                         \
     if (st != cudaSuccess)                                                       \
       FAIL() << "error in cudaSetDevice(" << (dev) << "): " << st;               \
     if ((st = cudaDeviceReset()) != cudaSuccess)                                 \
       FAIL() << "error in setup cudaDeviceReset: " << st;                        \
     _cukern_##test##_##test_case<<<grid_dim, block_dim, shared_mem, stream>>>(); \
     if ((st = cudaDeviceSynchronize()) == CUDA_ASSERT_STATUS)                    \
       st = (cudaError_t)~cudaSuccess;                                            \
     const cudaError_t _st = cudaDeviceReset();                                   \
     if (_st != cudaSuccess)                                                      \
       FAIL() << "error in teardown cudaDeviceReset: " << _st;                    \
     if (st == (cudaError_t)~cudaSuccess)                                         \
       FAIL() << "ASSERTION FAILED";                                              \
     else if (st != cudaSuccess)                                                  \
       FAIL() << "error in cudaDeviceSynchronize: " << st;                        \
     else                                                                         \
       SUCCEED();                                                                 \
  }                                                                               \
  __device__ void _cutest_##test##_##test_case(void)
#endif /* ?CUDA_TEST */

#endif /* !CUDA_TEST_CUH */
