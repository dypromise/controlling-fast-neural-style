extern "C" {
#include "lauxlib.h"
#include "lua.h"
#include "lualib.h"
}

#include "THC.h"
#include "luaT.h"

#include "curand_kernel.h"
#include <assert.h>
#include <float.h>
#include <getopt.h>
#include <math_constants.h>
#include <math_functions.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define TB 256
#define EPS 0.1

#undef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#undef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))

__host__ __device__ int clamp(int x, int x_max,
                              int x_min) { // assume x_max >= x_min
  if (x > x_max) {
    return x_max;
  } else if (x < x_min) {
    return x_min;
  } else {
    return x;
  }
}

__host__ __device__ int cuMax(int a, int b) {
  if (a > b) {
    return a;
  } else {
    return b;
  }
}
__host__ __device__ int cuMin(int a, int b) {
  if (a < b) {
    return a;
  } else {
    return b;
  }
}

__device__ float
MycuRand(curandState &state) { // random number in cuda, between 0 and 1

  return curand_uniform(&state);
}

__device__ void
InitcuRand(curandState &state) { // random number in cuda, between 0 and 1

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  curand_init(i, 0, 0, &state);
}

THCState *getCutorchState(lua_State *L) {
  lua_getglobal(L, "cutorch");
  lua_getfield(L, -1, "getState");
  lua_call(L, 0, 1);
  THCState *state = (THCState *)lua_touserdata(L, -1);
  lua_pop(L, 2);
  return state;
}

void checkCudaError(lua_State *L) {
  cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    luaL_error(L, cudaGetErrorString(status));
  }
}

THCudaTensor *new_tensor_like(THCState *state, THCudaTensor *x) {
  THCudaTensor *y = THCudaTensor_new(state);
  THCudaTensor_resizeAs(state, y, x);
  return y;
}

__global__ void histogram_nomask_kernel(float *I, float *minI, float *maxI,
                                        int nbins, int c, int h, int w,
                                        float *hist) {
  int _id = blockIdx.x * blockDim.x + threadIdx.x;
  int size = h * w;

  if (_id < c * size) {
    int id = _id % size, dc = _id / size;

    float val = I[_id];
    float _minI = minI[dc];
    float _maxI = maxI[dc];

    if (_minI == _maxI) {
      _minI -= 1;
      _maxI += 1;
    }

    if (_minI <= val && val <= _maxI) {
      int idx = MIN((val - _minI) / (_maxI - _minI) * nbins, nbins - 1);
      int index = dc * nbins + idx;
      atomicAdd(&hist[index], 1.0f);
    }
  }
  return;
}

int histogram_nomask(lua_State *L) {
  THCState *state = getCutorchState(L);
  THCudaTensor *I = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");
  int nbins = luaL_checknumber(L, 2);
  THCudaTensor *minI =
      (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *maxI =
      (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");

  int c = THCudaTensor_size(state, I, 0);
  int h = THCudaTensor_size(state, I, 1);
  int w = THCudaTensor_size(state, I, 2);

  THCudaTensor *hist = THCudaTensor_new(state);
  THCudaTensor_resize2d(state, hist, c, nbins);
  THCudaTensor_zero(state, hist);

  histogram_nomask_kernel<<<(c * h * w - 1) / TB + 1, TB>>>(
      THCudaTensor_data(state, I), THCudaTensor_data(state, minI),
      THCudaTensor_data(state, maxI), nbins, c, h, w,
      THCudaTensor_data(state, hist));
  checkCudaError(L);

  luaT_pushudata(L, hist, "torch.CudaTensor");
  return 1;
}

__global__ void histogram_kernel(float *I, float *minI, float *maxI,
                                 float *mask, int nbins, int c, int h, int w,
                                 float *hist) {
  int _id = blockIdx.x * blockDim.x + threadIdx.x;
  int size = h * w;

  if (_id < c * size) {
    int id = _id % size, dc = _id / size;

    if (mask[id] < EPS)
      return;

    float val = I[_id];

    float _minI = minI[dc];
    float _maxI = maxI[dc];

    if (_minI == _maxI) {
      _minI -= 1;
      _maxI += 1;
    }

    if (_minI <= val && val <= _maxI) {
      int idx = MIN((val - _minI) / (_maxI - _minI) * nbins, nbins - 1);
      int index = dc * nbins + idx;
      atomicAdd(&hist[index], 1.0f);
    }
  }

  return;
}

int histogram(lua_State *L) {
  THCState *state = getCutorchState(L);
  THCudaTensor *I = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");
  int nbins = luaL_checknumber(L, 2);
  THCudaTensor *minI =
      (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *maxI =
      (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");
  THCudaTensor *mask =
      (THCudaTensor *)luaT_checkudata(L, 5, "torch.CudaTensor");

  int c = THCudaTensor_size(state, I, 0);
  int h = THCudaTensor_size(state, I, 1);
  int w = THCudaTensor_size(state, I, 2);

  THCudaTensor *hist = THCudaTensor_new(state);
  THCudaTensor_resize2d(state, hist, c, nbins);
  THCudaTensor_zero(state, hist);

  histogram_kernel<<<(c * h * w - 1) / TB + 1, TB>>>(
      THCudaTensor_data(state, I), THCudaTensor_data(state, minI),
      THCudaTensor_data(state, maxI), THCudaTensor_data(state, mask), nbins, c,
      h, w, THCudaTensor_data(state, hist));
  checkCudaError(L);

  luaT_pushudata(L, hist, "torch.CudaTensor");
  return 1;
}

__global__ void hist_remap_nomask_kernel(float *I, float *histJ, float *cumJ,
                                         float *_minJ, float *_maxJ, int nbins,
                                         float *_sortI, int *_idxI, float *R,
                                         int c, int h, int w) {
  int _id = blockIdx.x * blockDim.x + threadIdx.x;
  int size = h * w;

  if (_id < c * size) {
    // _id = dc * size + id
    int id = _id % size, dc = _id / size;

    float minJ = _minJ[dc];
    float maxJ = _maxJ[dc];
    float stepJ = (maxJ - minJ) / nbins;

    int idxI = _idxI[_id] - 1;
    int offset = 0;
    int cdf = id - offset;

    int s = 0;
    int e = nbins - 1;
    int m = (s + e) / 2;
    int binIdx = -1;

    while (s <= e) {
      // special handling for range boundary
      float cdf_e =
          m == nbins - 1 ? cumJ[dc * nbins + m] + 0.5f : cumJ[dc * nbins + m];
      float cdf_s = m == 0 ? -0.5f : cumJ[dc * nbins + m - 1];

      if (cdf >= cdf_e) {
        s = m + 1;
        m = (s + e) / 2;
      } else if (cdf < cdf_s) {
        e = m - 1;
        m = (s + e) / 2;
      } else {
        binIdx = m;
        break;
      }
    }

    float hist = histJ[dc * nbins + binIdx];
    float cdf_e = cumJ[dc * nbins + binIdx];
    float cdf_s = cdf_e - hist;
    float ratio = MIN(MAX((cdf - cdf_s) / (hist + 1e-8), 0.0f), 1.0f);
    float activation = minJ + (static_cast<float>(binIdx) + ratio) * stepJ;
    R[dc * size + idxI] = activation;
  }
  return;
}

int hist_remap_nomask(lua_State *L) {
  THCState *state = getCutorchState(L);
  THCudaTensor *I = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");
  THCudaTensor *histJ =
      (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *cumJ =
      (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *minJ =
      (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");
  THCudaTensor *maxJ =
      (THCudaTensor *)luaT_checkudata(L, 5, "torch.CudaTensor");
  int nbins = luaL_checknumber(L, 6);
  THCudaTensor *sortI =
      (THCudaTensor *)luaT_checkudata(L, 7, "torch.CudaTensor");
  THCudaIntTensor *idxI =
      (THCudaIntTensor *)luaT_checkudata(L, 8, "torch.CudaIntTensor");
  THCudaTensor *R = (THCudaTensor *)luaT_checkudata(L, 9, "torch.CudaTensor");

  int c = THCudaTensor_size(state, I, 0);
  int h = THCudaTensor_size(state, I, 1);
  int w = THCudaTensor_size(state, I, 2);

  hist_remap_nomask_kernel<<<(c * h * w - 1) / TB + 1, TB>>>(
      THCudaTensor_data(state, I), THCudaTensor_data(state, histJ),
      THCudaTensor_data(state, cumJ), THCudaTensor_data(state, minJ),
      THCudaTensor_data(state, maxJ), nbins, THCudaTensor_data(state, sortI),
      THCudaIntTensor_data(state, idxI), THCudaTensor_data(state, R), c, h, w);
  checkCudaError(L);
  return 0;
}

__global__ void hist_remap_kernel(float *I, int nI, float *mI, float *histJ,
                                  float *cumJ, float *_minJ, float *_maxJ,
                                  int nbins, float *_sortI, int *_idxI,
                                  float *R, int c, int h, int w) {
  int _id = blockIdx.x * blockDim.x + threadIdx.x;
  int size = h * w;

  if (_id < c * size) {
    // _id = dc * size + id
    int id = _id % size, dc = _id / size;

    float minJ = _minJ[dc];
    float maxJ = _maxJ[dc];
    float stepJ = (maxJ - minJ) / nbins;

    int idxI = _idxI[_id] - 1;
    if (mI[idxI] < EPS)
      return;
    int offset = h * w - nI;

    int cdf = id - offset;

    int s = 0;
    int e = nbins - 1;
    int m = (s + e) / 2;
    int binIdx = -1;

    while (s <= e) {
      // special handling for range boundary
      float cdf_e =
          m == nbins - 1 ? cumJ[dc * nbins + m] + 0.5f : cumJ[dc * nbins + m];
      float cdf_s = m == 0 ? -0.5f : cumJ[dc * nbins + m - 1];

      if (cdf >= cdf_e) {
        s = m + 1;
        m = (s + e) / 2;
      } else if (cdf < cdf_s) {
        e = m - 1;
        m = (s + e) / 2;
      } else {
        binIdx = m;
        break;
      }
    }

    float hist = histJ[dc * nbins + binIdx];
    float cdf_e = cumJ[dc * nbins + binIdx];
    float cdf_s = cdf_e - hist;
    float ratio = MIN(MAX((cdf - cdf_s) / (hist + 1e-8), 0.0f), 1.0f);
    float activation = minJ + (static_cast<float>(binIdx) + ratio) * stepJ;
    R[dc * size + idxI] = activation;
  }

  return;
}

int hist_remap(lua_State *L) {
  THCState *state = getCutorchState(L);
  THCudaTensor *I = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");
  int nI = luaL_checknumber(L, 2);
  THCudaTensor *mI = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *histJ =
      (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");
  THCudaTensor *cumJ =
      (THCudaTensor *)luaT_checkudata(L, 5, "torch.CudaTensor");
  THCudaTensor *minJ =
      (THCudaTensor *)luaT_checkudata(L, 6, "torch.CudaTensor");
  THCudaTensor *maxJ =
      (THCudaTensor *)luaT_checkudata(L, 7, "torch.CudaTensor");
  int nbins = luaL_checknumber(L, 8);
  THCudaTensor *sortI =
      (THCudaTensor *)luaT_checkudata(L, 9, "torch.CudaTensor");
  THCudaIntTensor *idxI =
      (THCudaIntTensor *)luaT_checkudata(L, 10, "torch.CudaIntTensor");
  THCudaTensor *R = (THCudaTensor *)luaT_checkudata(L, 11, "torch.CudaTensor");

  int c = THCudaTensor_size(state, I, 0);
  int h = THCudaTensor_size(state, I, 1);
  int w = THCudaTensor_size(state, I, 2);

  hist_remap_kernel<<<(c * h * w - 1) / TB + 1, TB>>>(
      THCudaTensor_data(state, I), nI, THCudaTensor_data(state, mI),
      THCudaTensor_data(state, histJ), THCudaTensor_data(state, cumJ),
      THCudaTensor_data(state, minJ), THCudaTensor_data(state, maxJ), nbins,
      THCudaTensor_data(state, sortI), THCudaIntTensor_data(state, idxI),
      THCudaTensor_data(state, R), c, h, w);
  checkCudaError(L);

  return 0;
}

static const struct luaL_Reg funcs[] = {
    {"histogram", histogram},                 // compute histogram
    {"histogram_nomask", histogram_nomask},   // compute histogram
    {"hist_remap", hist_remap},               // histogram remapping
    {"hist_remap_nomask", hist_remap_nomask}, // histogram remapping

    {NULL, NULL}};

extern "C" int luaopen_libcuda_utils(lua_State *L) {
  luaL_openlib(L, "cuda_utils", funcs, 0);
  return 1;
}