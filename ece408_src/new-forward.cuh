
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#define TILE_WIDTH 784

#include <mxnet/base.h>



namespace mxnet
{
namespace op
{




__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K) {

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */
	extern __shared__ float shmem[];
	float* X_shared = &shmem[0];
	float* W_shared = &shmem[H * W * C];

    int H_out = H - K + 1;
    int W_out = W - K + 1;

    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a
    // #define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
    #define x4d(i2,i1,i0) X_shared[(i2)*(H * W) + (i1)*(W) + i0]
    #define k4d(i3,i2,i1,i0) W_shared[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]

    /*
        Your code here!
    */
    int b = blockIdx.x;
    int tx = threadIdx.x;
    // float acc = 0.;

    for (int i = 0; i*TILE_WIDTH < M*C*K*K; i++){
        if (i*TILE_WIDTH + tx < M*C*K*K)
            W_shared[i*TILE_WIDTH + tx] = k[i*TILE_WIDTH + tx];
    }

    for (int i = 0; i*TILE_WIDTH < C*H*W; i++){
        if (i*TILE_WIDTH + tx < C*H*W)
            X_shared[i*TILE_WIDTH + tx] = x[b*C*H*W + i*TILE_WIDTH + tx];
    }
    __syncthreads();

    for (int i = 0; i*TILE_WIDTH < M*H_out*W_out; i++){
        if (i*TILE_WIDTH + tx < M*H_out*W_out) {
            float acc = 0.;
            // acc = w(M, CKK) * x(CKK, 1)
            int w_out = (i*TILE_WIDTH + tx) % W_out;
            int h_out = (i*TILE_WIDTH + tx) / W_out % H_out;
            int m = (i*TILE_WIDTH + tx) / W_out / H_out;
            for (int c = 0; c < C; c++) {
                for (int p = 0; p < K; p++) {
                    for (int q = 0; q < K; q++) {
                        acc += k4d(m, c, p, q) * x4d(c, h_out + p, w_out + q);
                    }
                }
            }
            y[b*M*H_out*W_out + i*TILE_WIDTH + tx] = acc;
        }
    }
    
    // #undef y4d
    #undef x4d
    #undef k4d
}




/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template<>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {
    

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Missing an ECE408 GPU implementation!";

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    cudaStream_t s = y.stream_->stream_;

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0];  // batch
    const int M = y.shape_[1];  // output channel
    const int C = x.shape_[1];  // input channel
    const int H = x.shape_[2];  // input height
    const int W = x.shape_[3];  // input width
    const int K = w.shape_[3];  // kernel size
    // const int H_out = H - K + 1;
    // const int W_out = W - K + 1;

    // Set the kernel dimensions
    dim3 gridDim(B, 1, 1);
    dim3 blockDim(TILE_WIDTH, 1, 1);
    size_t shmem_size = sizeof(float)* (H*W*C + K*K*C*M);

    // Call the kernel
    forward_kernel<<<gridDim, blockDim, shmem_size, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}


/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    assert( 0 && "No forward implementation for other datatypes needed for ECE408");
}

}
}

#endif