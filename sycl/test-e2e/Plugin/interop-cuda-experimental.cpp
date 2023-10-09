// REQUIRES: cuda && cuda_dev_kit

// RUN: %{build} %cuda_options -o %t.out
// RUN: %{run} %t.out

#define SYCL_EXT_ONEAPI_BACKEND_CUDA_EXPERIMENTAL 1
#include <sycl/ext/oneapi/experimental/backend/cuda.hpp>
#include <sycl/sycl.hpp>

#include <cuda.h>

#include <assert.h>

void cuda_check(CUresult error) { assert(error == CUDA_SUCCESS); }

template <typename refT, typename T> void check_type(T var) {
  static_assert(std::is_same_v<decltype(var), refT>);
}

#define CUDA_CHECK(error) cuda_check(error)

bool check_queue(sycl::queue &Q) {
  constexpr size_t vec_size = 5;
  double A_Data[vec_size] = {4.0};
  double B_Data[vec_size] = {-3.0};
  double C_Data[vec_size] = {0.0};

  sycl::buffer<double, 1> A_buff(A_Data, sycl::range<1>(vec_size));
  sycl::buffer<double, 1> B_buff(B_Data, sycl::range<1>(vec_size));
  sycl::buffer<double, 1> C_buff(C_Data, sycl::range<1>(vec_size));

  Q.submit([&](sycl::handler &cgh) {
     auto A_acc = A_buff.get_access<sycl::access::mode::read>(cgh);
     auto B_acc = B_buff.get_access<sycl::access::mode::read>(cgh);
     auto C_acc = C_buff.get_access<sycl::access::mode::write>(cgh);
     cgh.parallel_for(sycl::range<1>{vec_size}, [=](sycl::id<1> idx) {
       C_acc[idx] = A_acc[idx] + B_acc[idx];
     });
   }).wait();

  sycl::host_accessor C_acc(C_buff);
  return C_acc[0] == 1;
}

int main() {
  sycl::event e;
  auto b = e.get_backend();
  if (b == sycl::backend::ext_oneapi_cuda) {
    printf("CUDA\n");
  }
  printf("main\n");
  sycl::context sycl_ctx;
  sycl::queue Q{sycl_ctx, sycl::default_selector_v};
  /* sycl::context sycl_ctx = Q.get_context(); */
  printf("context\n");
  CUdevice Q_dev =
      sycl::get_native<sycl::backend::ext_oneapi_cuda>(Q.get_device());
  std::vector<CUcontext> cu_ctxs =
      sycl::get_native<sycl::backend::ext_oneapi_cuda>(sycl_ctx);
  printf("contexts\n");
  CUcontext cu_ctx = nullptr;
  for (auto ctx : cu_ctxs) {
    cuCtxSetCurrent(ctx);
    printf("cuCtxSetCurrent\n");
    CUdevice dev;
    cuCtxGetDevice(&dev);
    printf("cuCtxGetDevice\n");
    if (dev == Q_dev) {
      cu_ctx = ctx;
      break;
    }
  }
  printf("loop\n");
  assert(cu_ctx && "No context for SYCL device");

  printf("okay\n");
  // Get native cuda device
  CUdevice cu_dev;
  CUDA_CHECK(cuDeviceGet(&cu_dev, 0));
  auto sycl_dev = sycl::make_device<sycl::backend::ext_oneapi_cuda>(cu_dev);
  auto native_dev = sycl::get_native<sycl::backend::ext_oneapi_cuda>(sycl_dev);
  auto sycl_dev2 =
      sycl::make_device<sycl::backend::ext_oneapi_cuda>(native_dev);

  check_type<sycl::device>(sycl_dev);
  check_type<CUdevice>(native_dev);
  assert(native_dev == cu_dev);
  assert(sycl_dev == sycl_dev2);

  // Create sycl queue with new device and submit some work
  {
    sycl::queue new_Q(sycl_dev);
    assert(check_queue(new_Q));
  }



  printf("devuce\n");

  // Create new event
  CUevent cu_event;

  CUDA_CHECK(cuCtxSetCurrent(cu_ctx));
  CUDA_CHECK(cuEventCreate(&cu_event, CU_EVENT_DEFAULT));

  auto sycl_event =
      sycl::make_event<sycl::backend::ext_oneapi_cuda>(cu_event, sycl_ctx);
  printf("sycl_event\n");
  auto native_event =
      sycl::get_native<sycl::backend::ext_oneapi_cuda>(sycl_event);
  printf("native_event\n");

  check_type<sycl::event>(sycl_event);
  check_type<CUevent>(native_event);

  // Check sycl queue with sycl_ctx still works
  {
    sycl::queue new_Q(sycl_ctx, sycl::default_selector_v);
  printf("queue\n");
    assert(check_queue(new_Q));
  printf("assert\n");
  }

  printf("event\n");

  // Check has_native_event
  {
    auto e = Q.submit([&](sycl::handler &cgh) { cgh.single_task([] {}); });
    assert(sycl::ext::oneapi::cuda::has_native_event(e));
    printf("queue submit?\n");
  }

  {
    auto e = Q.submit([&](sycl::handler &cgh) { cgh.host_task([] {}); });
    printf("host task submitted\n");
    assert(!sycl::ext::oneapi::cuda::has_native_event(e));
  }

  printf("after event\n");

  // Create new queue
  CUstream cu_queue;
  CUDA_CHECK(cuCtxSetCurrent(cu_ctx));
  CUDA_CHECK(cuStreamCreate(&cu_queue, CU_STREAM_DEFAULT));

  printf("sycl_queue\n");
  auto sycl_queue =
      sycl::make_queue<sycl::backend::ext_oneapi_cuda>(cu_queue, sycl_ctx);
  printf("native_queue\n");
  auto native_queue = sycl::get_native<sycl::backend::ext_oneapi_cuda>(sycl_queue);

  printf("check_type\n");
  check_type<sycl::queue>(sycl_queue);
  check_type<CUstream>(native_queue);

  printf("check_queue\n");
  // Submit some work to new queue
  assert(check_queue(sycl_queue));
  printf("After check_queue\n");

  // Create new queue with Q's native type and submit some work
  {
    CUstream Q_native_stream =
        sycl::get_native<sycl::backend::ext_oneapi_cuda>(Q);
    sycl::queue new_Q = sycl::make_queue<sycl::backend::ext_oneapi_cuda>(
        Q_native_stream, sycl_ctx);
    assert(check_queue(new_Q));
  }

  printf("after check Q native\n");

  // Check Q still works
  assert(check_queue(Q));
}
