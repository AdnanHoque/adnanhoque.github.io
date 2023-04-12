---
title: "SSCST - Part 1"
date: 2023-04-11
layout: post
---

Part one is dedicated to a discussion around the Apple Neural Engine. We will take a deep dive into the first principle discussed in [1]. Part 2, will examine the Self-Supervised Chewing Spectogram Architecture (SSCST), the audio preprocessing steps, and the deployment steps for an audio transformer that will be fine tuned to do chewing detection. The architecture and use-case is for research purposes and should not be used for commercial applications without prior permission. 

# 1 Transformers on Apple Silicon

The insights in this section are developed from [1], the authors develop four principles to accelerate Transformers runnning on Apple Silicon. I'll discuss some of my thoughts on their first principle below.

# 1.1 Principle 1 — Picking the Right Data Format

This is the first principle discussed in [1]  and it boils down to representing the input seqence with a (B, S, 1, C) format as opposed to the more commonly used (B, S, C). Let's explore how this tensor layout might impact performance by first examining how arrays are stored in one-dimensional linear memory.

PyTorch Tensors are row-major. This means that the tensors are stored in such a way that the fastest-changing dimension is the column dimension. Elements in this dimension can be accessed contiguously. Here is a simple example to illustrate what it means for a tensor to be row-major:

Suppose we have a 2D Matrix A with dimensions (3, 4):

~~~

A = [[a11, a12, a13, a14],
    [a21, a22, a23, a24],
    [a31, a32, a33, a34]]
~~~

In a row-major layout, the elements are stored in memory sequentially by rows:

~~~
[a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34]
~~~

The fastest-changing dimension is the last dimension (columns in this example). As you can see, when traversing the linear memory, the column indices change first (a11, a12, a13, a14) before moving to the next row. This is why elements that are contiguous in the last dimension (columns) are also contiguous in memory.

Let's consider the implications of this on different architectures:

1. For a convolution layers in CNNs, the operation involves iterating over the spatial dimensions (height and width) of the input feature map and the channels. The kernel is applied at each spatial location, sliding across the height and width dimensions, and computing a weighted sum of the input channels. In this case, we iterate through the channel dimension and access the spatial dimension contiguously. 

2. Fully connected layers: In fully connected layers, the input and output neurons are connected through a weight matrix. When computing the dot product between the input vector and the weight matrix, your algorithm will typically iterate through and access the elements in the input vector and weight matrix contiguously.

3. Pooling layers in a CNN: In pooling layers, the operation is applied over the spatial dimensions (height and width) of the input feature map. Your algorithm is likely to iterate through the spatial dimensions and access the channel dimension contiguously.

4. Transformers: In a transformer architecture, the self-attention mechanism processes the input sequence. Your algorithm will likely iterate through the sequence length dimension and access the elements in the input sequence contiguously.

To develop an intuition for this principle let's examine convolutional layers. As we are accessing the spatial dimension contiguously it would make sense for our input tensor to have a shape of (B, H, W, C). 

Consider the following example to further illustrate this:

Assume we have a single image with 3 channels (R, G, B) and 3x3 dimensions, so our tensor has the shape (1, 3, 3, 3). In NCHW format, the memory layout would look like this:

~~~
[R00 R01 R02 R10 R11 R12 R20 R21 R22 G00 G01 G02 G10 G11 G12 G20 G21 G22 B00 B01 B02 B10 B11 B12 B20 B21 B22]
~~~

For the first position in the image (0, 0), the convolution operation needs to access R00, R01, R10, R11, G00, G01, G10, G11, B00, B01, B10, and B11. Notice that these values are stored consecutively in memory which allows us to take advantage of coalesced memory access and cache locality. 

# 1.2 Coalesced Memory Access


For NVIDIA GPUs this works in the following way, when threads in a warp access consecutive memory locations, the GPU hardware can combine these requests into a single transaction. This means that the data needed by all threads in the warp is fetched from memory in a single operation, reducing the number of memory transactions and improving overall performance.

For example, consider a warp (a group of 32 threads executing the same instructio), and each thread reads one element from memory. If these elements are stored consecutively in memory, the GPU can fetch them in a single transaction:

~~~
Memory:   A0 A1 A2 A3 ... A31
           
           |  |  |  |      |

Threads:  T0 T1 T2 T3 ... T31

~~~

In this case, thread T0 reads A0, thread T1 reads A1, and so on. The hardware can fetch all these elements in a single memory transaction.

# 1.3. Cache Locality 

Cache locality refers to the tendency of a program to access data that is close in memory to data that has been recently accessed. There are two types of cache locality: spatial locality and temporal locality. Spatial locality means that when a piece of data is accessed, it is likely that nearby data will be accessed soon. Temporal locality means that when a piece of data is accessed, it is likely that the same data will be accessed again soon. By accessing data with good cache locality, a program can take advantage of the cache hierarchy (L1, L2, and L3 caches) in modern CPUs and GPUs, which can significantly improve performance by reducing the latency of memory accesses.

Again, let's illustrate this whole process with an example. Consider a 3x3 input RGB Image and 2x2 kernel.

Input Image:
~~~
Input R (IR)       Input G (IG)       Input B (IB)
[ IR00 IR01 IR02 ] [ IG00 IG01 IG02 ] [ IB00 IB01 IB02 ]
[ IR10 IR11 IR12 ] [ IG10 IG11 IG12 ] [ IB10 IB11 IB12 ]
[ IR20 IR21 IR22 ] [ IG20 IG21 IG22 ] [ IB20 IB21 IB22 ]
~~~

Kernel:

~~~
Kernel (K)
[ K00 K01 ]
[ K10 K11 ]
~~~

The sequence of events are as follows,

1. The GPU begins by fetching the first 2x2 region of the red channel (IR00, IR01, IR10, IR11) along with the kernel values (K00, K01, K10, K11).

~~~
Fetched data from memory:
Input R:  IR00, IR01, IR10, IR11
Kernel:   K00, K01, K10, K11
~~~

2. The  GPU calculates the convolution for this region on the red channel

~~~
Convolution R (CR):
CR00 = IR00 * K00 + IR01 * K01 + IR10 * K10 + IR11 * K11
~~~

3. The GPU moves the kernel one step to the right and fetches the next 2x2 region of the red channel (IR01, IR02, IR11, IR12).

~~~
Convolution R (CR):
CR01 = IR01 * K00 + IR02 * K01 + IR11 * K10 + IR12 * K11
~~~

Since IR01 and IR11 are already in the cache from the previous step, only IR02 and IR12 need to be fetched from memory.

# Remarks

In Part 2, we'll continue to develop the intuition for an optimal tensor layout for the transformer as we did for the CNN. We'll take a look at my Audio Transformer implementation mostly based on [1] and [3] but modified in certain places for my use-case, and hardware platform. Training was mainly done on Colab Pro with a single NVIDIA A100 GPU and an Apple M2 Max Pro (38 core GPU + 16 core neural engine) laptop. I was interested to learn more about the compute abilities of the M2 and to test out the "Metal Acceleration" the result of PyTorch's collaboration with Apple which saw accelerated GPU training being enabled using Apple's Metal Performance Shaders (MPS) [4]. Great for fine-tuning type tasks and protyping locally. The GPU on the M2 Max is rated for 15.6 TFlops which means it's comparable to an Nvidia RTX 3080 (without concurrent integer operations and Max TDP = 350 Watts), for a low-powered machine (Max TDP = 79 Watts) the compute capibilities of this machine are very impressive.

On the appropriateness of using this type of transformers architecture for Audio Tasks. Transformers have acheieved SOTA performance in different domains and various tasks. However, it's been shown that Transformers will only outperform CNNs when the training volume size exceeds 100 milion samples. Thus, training from scratch will yield poor results on small datasets. Due to the quadratic scaling of the self-attention mechanism with the input size, compute is also a consideration for the many researchers that do not have access to the increasing number of super-clusters owned by industrial AI labs. However, it is difficult to ignore the SOTA results in every domain the transformer has been introduced, which has in turn led to the convergence of deep leanrning architectures we have witneessed over the past few years. In part 2 I'll also discuss the Mel Spectogram, masking strategy that I employed and other training strategies for foundation models, training and deployment pipeline, quantization as well and lastly neural engine performance on iPhone.

# Links
[1] [https://machinelearning.apple.com/research/neural-engine-transformers](https://machinelearning.apple.com/research/neural-engine-transformers)
[2] [https://arxiv.org/pdf/2010.11929.pdf](https://arxiv.org/pdf/2010.11929.pdf)
[3] [Audio Spectogram Transformer, Yuan Dong et al.](https://arxiv.org/pdf/2104.01778.pdf)
[4] [https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/)