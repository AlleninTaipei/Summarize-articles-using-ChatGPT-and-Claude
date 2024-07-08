# Smart-Infinity: Enhancing LLM Training with Near-Storage Processing

* [Source](https://arxiv.org/html/2403.06664v1):arXiv:2403.06664v1 11 Mar 2024
* [HPCA 2024 Best paper Award Winner - Honorable Mention](https://hpca-conf.org/2024/)
* [Smart-Infinity: Fast Large Language Model Training using Near-Storage Processing on a Real System](https://github.com/AIS-SNU/smart-infinity)


## Intruduction

* The rapid advancements in Large Language Models (LLMs) are driven by increasing parameter sizes, leading to substantial memory capacity requirements. Typically, this necessitates the use of numerous GPUs. **A popular solution to mitigate GPU memory limitations is storage-offloaded training,** which uses host memory and storage as an extended memory hierarchy. However, this approach is hampered by the significant bandwidth bottleneck of storage devices compared to GPU memory.

* Introduces a novel approach to improve the efficiency of training large language models (LLMs) by addressing the storage bandwidth bottleneck inherent in traditional storage-offloaded training methods.

## Background

|Background|Offloading Solutions|
|-|-|
|Host Memory-Offloaded Training|Optimizer states are offloaded to host memory.
||CPU updates parameters due to the low computational intensity of these operations.|
||Suitable for LLMs that exceed GPU memory capacity but can still fit within the host memory limits.|
|Storage-Offloaded Training|For larger models that exceed both GPU and host memory capacities, optimizer states are offloaded to storage devices (e.g., NVMe SSDs).|
||This method involves storing optimizer states and gradients in storage while keeping activations and model parameters in host memory. Although this approach mitigates GPU memory limitations, it introduces significant data transfer overhead due to the limited bandwidth of storage devices compared to GPU memory.|

|Step|Dataflow in Storage-Offloaded Training|
|-|-|
|Forward Pass|Step 1: GPU loads mixed precision parameters of a block.|
||Step 2: GPU processes the block forward.|
||Step 3: GPU checkpoints the activation of the block to host memory.|
||This process repeats for all blocks, ensuring that activations of the entire model are checkpointed.|
|Backward Pass|Step 1: GPU loads mixed precision parameters and activations for a block.|
||Step 2: GPU processes the block backward.|
||Step 3: Gradients are sent to host memory.|
||Step 4: Host offloads the created gradients to storage devices.|
|Update Procedure|Step 1: Gradients and optimizer states are uploaded from storage to host memory.|
||Step 2: CPU loads gradients and optimizer states.|
||Step 3: CPU updates parameters.|
||Step 4: CPU replaces mixed precision parameters with updated ones.|
||Step 5: CPU offloads updated optimizer states to storage.|
||The update procedure repeats for all blocks, followed by the next forward pass iteration.|

## Motivation

* The motivational study demonstrates that data transfer overhead in storage-offloaded LLM training cannot be easily mitigated due to the limited resource of the existing system structure. Therefore, when we use CSDs as an alternative, our primary goal is minimizing the data transfer between storage devices and host memory through the shared system interconnect, and we need to fully utilize the aggregated internal bandwidth from multiple CSDs and the computational ability of the accelerator in each CSD.
* [Samsung SmartSSD® Computational Storage Drives, Powered by Xilinx](https://www.anandtech.com/show/16245/xilinx-and-samsung-launch-smartssd-computational-storage-drive)

|CSDs|Computational Storage Devices (CSDs) integrate computational engines (e.g., lightweight FPGA accelerators) near storage devices, enabling significant latency and bandwidth benefits while reducing the host workload. This concept, often referred to as near-storage processing (NSP), has been extensively studied and developed over the years.|
|-|-|
|On-Device Accelerators|CSDs are equipped with accelerators, such as lightweight FPGAs, that perform computations near the storage. This reduces the need for frequent data transfers between the storage and the host, enhancing overall efficiency.|
|Internal PCIe Switch|CSDs include an internal PCIe switch, allowing direct peer-to-peer (P2P) communication between the SSD and the accelerator. This bypasses the host, avoiding redundant data traffic through the system interconnect, and provides a private inner path for data transfers.|
|Latency Reduction|By placing computation closer to the storage, CSDs reduce the latency associated with data transfers between the storage and the host.|
|Bandwidth Optimization|The internal PCIe switch facilitates efficient data movement, improving bandwidth utilization and reducing bottlenecks.|
|Host Workload Reduction|Offloading computations to the accelerators on CSDs alleviates the processing burden on the host CPU.|
|Single CSD Configuration|When a single CSD is used, there is no significant bandwidth boost since both storage-to-FPGA and storage-to-host traffic use the same PCIe lanes.|
|Multiple CSD Configuration|When multiple CSDs are deployed, the internal bandwidth aggregates linearly with the number of CSDs. This is because each CSD operates independently, utilizing its internal PCIe switch and accelerators, while the shared system interconnect bandwidth remains constant.|

|Smart-Infinity Solution||
|-|-|
|Near-Storage Processing|Smart-Infinity utilizes computational storage devices (CSDs), specifically near-storage accelerators, to perform parameter updates directly on the storage side. This reduces the data transfer required between storage and host memory.|
|SmartUpdate|This component performs parameter updates on custom accelerators within the CSDs, significantly cutting down storage traffic. By handling updates near storage, SmartUpdate alleviates the bottleneck caused by storage bandwidth limitations.|
|Efficient Data Transfer Handling|The system employs a data transfer handler structure that overlaps data transfers with fixed memory consumption, enhancing system integration and efficiency.|
|Gradient Compression/Decompression|To further reduce traffic, gradients are compressed on the GPU and decompressed on the storage accelerators. This method scales well with multiple CSDs and helps in alleviating shared channel bottlenecks.|

|Implementation and Results|Integration with PyTorch: Smart-Infinity is fully integrated with PyTorch and is ready for use on real systems. The implementation is available on GitHub.|
|-|-|
|Performance Gains|The proposed method achieves up to a 2.11x speedup over the baseline in mixed-precision LLM training, demonstrating its effectiveness in reducing training time.|
|Technical Contributions|Smart-Infinity Framework: Introduces a method for performing update phases of LLM training in custom CSD accelerators, reducing storage bandwidth bottlenecks.|
|Data Transfer Handler|Proposes an efficient structure to utilize storage bandwidth effectively and hide latency.|
|Gradient Compression|Utilizes accelerator-assisted gradient compression/decompression to enhance scalability and performance.|
|Real-System Integration|Provides a practical, ready-to-use solution integrated with PyTorch for real-world applications.|
|Experimental Validation|Training Time Breakdown: Analyzes the time spent on different phases of storage-offloaded LLM training and identifies the update phase as the most time-consuming due to storage access overhead.|
|RAID0 Limitations|Speedup saturates after utilizing more than four SSDs. Shows that increasing the number of SSDs in a RAID0 configuration eventually hits a bottleneck due to limited system interconnect bandwidth, highlighting the need for solutions like Smart-Infinity.|

## Smart-Infinity

* Smart-Infinity is an advanced system designed to optimize the training of large language models (LLMs) by reducing communication overhead and leveraging computational storage devices (CSDs).

* Smart-Infinity enhances LLM training by reducing interconnect traffic, leveraging the internal bandwidth of CSDs, and incorporating gradient compression. It is flexible and easily integrates with existing training frameworks, making it a powerful tool for efficient model training.

* This system is particularly effective with the Adam optimizer and involves several innovative components: **SmartUpdate, internal data transfer handler, and SmartComp.**

|SmartUpdate|Description|
|-|-|
|Function|Offloads the parameter update procedure from the CPU to the FPGA within CSDs.|
|Benefits|Reduces communication traffic between storage and host memory, utilizing the high internal bandwidth of CSDs.|
|Traffic Reduction|Minimizes the communication volume from 8M (baseline) to 4M (2M read + 2M write).|

|Internal Data Transfer Handler|Description|
|-|-|
|Function|Optimizes internal data transfers by overlapping buffer allocations and transfers.|
|Benefits|Enhances throughput and avoids out-of-memory errors.|
|Implementation|Uses preallocated buffers and separate threads to manage non-urgent data transfers.|

|SmartComp|Description|
|-|-|
|Function|Compresses gradients before storage and decompresses them using FPGA before the update phase.|
|Benefits|Reduces storage write traffic by a factor of the compression ratio (c%) and mitigates system interconnect bottlenecks.|
|Compression Method|Uses magnitude-based compression to select high-magnitude gradient elements.|

## Accelerator Architecture

|Smart-Infinity's accelerator architecture|Designed to handle various algorithms|
|-|-|
|Updater|Loads gradients, optimizer states, and target parameters.|
||Utilizes SIMD AXPBY units for averaging operations.|
||Updates target parameters and swaps updated data to storage via the data transfer handler.|
|Decompressor|Initializes gradient buffer with zero.|
||Loads and maps compressed gradients using indices and values.|
||Repeats the process until all elements are decompressed.|
||Decompressed gradients are used by the updater.|
|Flexibility|The accelerator can be extended to support other optimizers and compression methods, ensuring versatility and adaptability for different training scenarios.|

## [Implementation](https://github.com/AIS-SNU/smart-infinity.)

* Smart-Infinity enhances the LLM training process by providing a customizable, efficient, and seamless integration with DeepSpeed and PyTorch. It leverages high-level synthesis for user-defined logic, optimized hardware interaction, and automatic compilation for ease of use, making it a practical solution for advanced LLM training workflows.

|Key Components|Implementation|
|-|-|
|Parameter Update Feature|Smart-Infinity focuses on replacing the parameter update feature of DeepSpeed ZeRO-Infinity.|
||This replacement allows for more efficient parameter updates using custom logic implemented in C/C++.|
|Build Process|Uses `disutils` to build C/C++ implementations as additional Python modules.|
||Compilation is automatic when the DeepSpeed engine is initialized, enabling Smart-Infinity with a simple option specification.|
||No modifications are required for existing LLM training codes implemented with DeepSpeed to run with Smart-Infinity.|
|Customization via High-Level Synthesis (HLS)|Users can customize the decompressor/updater logic using HLS codes.|
||Smart-Infinity provides HLS templates for implementing custom compression or weight update logic, including performance analyzers and sanity checkers.|
|Middleware|Smart-Infinity interacts with the DeepSpeed runtime engine, supporting forward and backward execution in mixed precision.|
||During backward execution, generated gradients are offloaded and modified to be located in the corresponding SSD for FPGA execution via internal P2P communication.|
||Updated parameters from Smart-Infinity are passed back to the DeepSpeed runtime engine.|
||Uses `pybind11` to enable direct communication with PyTorch applications by exposing Python types into C++.|
|FPGA Integration|Utilizes Xilinx OpenCL extensions to interact with attached FPGAs during runtime.|
||Identifies which FPGA device is connected to a specific SSD via internal PCIe switch during device initialization.|
||Pre-allocates OpenCL buffer for data transfer between SSD and FPGA device memory using `CL_MEM_EXT_PTR_XILINX` flag.|
||Supports standard Linux file access system calls (`pread`/`pwrite`) for direct P2P data transfer.|
|User Level|Shows the design flow for users to customize the Smart-Infinity decompressor/updater.|
||Users can modify the logic using provided HLS templates.|
|Middleware Level|Describes the interaction between Smart-Infinity and the DeepSpeed runtime engine.|
||Highlights the support for mixed precision execution and the handling of gradient offloading and parameter updates.|
|Hardware Level|Illustrates how the Smart-Infinity engine interacts with hardware components like FPGAs and SSDs.|
||Emphasizes the use of OpenCL for efficient data transfer and buffer management.|

## Evaluation

* The environment includes up to 10 SAMSUNG SmartSSD devices, each with a 4TB NVMe SSD connected to a Kintex UltraScale+ KU15P FPGA via PCIe Gen3.0 x4. The FPGA resources are approximately 522K LUTs, 984 BRAMs, 1968 DSPs, and 4GB DDR4 DRAM. We composed a software RAID via Linux mdadm and connected SmartSSDs via PCIe expansion.

|environment|
|-|
|GPU NVIDIA A5000, A100 (40GB), A4000|
|CPU Xeon(R) Gold 6342, 2 × 48C 96T|
|Memory 32 × 32GB DDR4-3200|
|SSD SAMSUNG SmartSSD, 4TB|
|PCIe Expansion H3 Falcon 4109|
|Ubuntu 20.04 LTS|
|Python / PyTorch 3.9 / 1.12.1|
|CUDA / OpenCL	11.6.2 / 2.2|
|Vitis / XRT 2023.1 / 2.12.427|
|GPT-2, BERT, BLOOM, ViT|
|DeepSpeed	0.9.3|

## Discussion

### An Alternative Scenario: Multi-GPU Congested Topology

* Using tensor parallelism slightly reduces forward ('FW') and backward ('BW') time with more GPUs. However, the remotely placed GPUs generate extra model and activation transfer traffic, sharing the same PCIe interconnect with the CSDs. This incurs overhead during the 'BW+Grad. Offload' phase, compared to the default topology, without significantly affecting the 'Update+Opt. Upload/Offload' phase where GPUs are idle. Consequently, the speedup observed is smaller than the default setup, indicating that PCIe topology structure impacts performance. Nonetheless, Smart-Infinity still achieves a 1.66× to 1.86× speedup with ten CSDs, demonstrating its extension to PCIe lane-limited environments with multiple GPUs.

### Applying Smart-Infinity to Model Compression

* In this work, we demonstrated Smart-Infinity's utility through fine-tuning. Various model compression methods, such as quantization, pruning, or low-rank decomposition, which require fine-tuning to recover from compression-induced accuracy drops, can also benefit from Smart-Infinity.

* These applications are expected to enhance Smart-Infinity's speedup further. As discussed in Section IV-C, Smart-Infinity's current bottleneck is the upstream model transfer from the CSD to the host. For model compression, Smart-Infinity can perform compression and upload the compressed model, reducing this bottleneck.

* However, achieving further speedup presents challenges. For instance, quantization often uses backpropagation with a straight-through estimator (STE) and variable floating-point quantization intervals per layer. This requires GPUs to use floating-point models instead of integers. To address this, CSDs must derive per-layer quantization intervals, convert models to integers, and send parameters and intervals upstream. GPUs then convert integer weights to floating points for STE. Pruning and low-rank decomposition present similar issues. Consequently, CSDs must handle more complex compression tasks using lightweight FPGA, and efficient GPU kernels for decompression must be developed to avoid new bottlenecks. We leave this for future work, anticipating more innovative approaches for model compression with Smart-Infinity.

* Smart-Infinity employs PCIe switch-based storage expansion systems to accommodate multiple CSDs. As modern workloads like LLMs, recommendation systems, big data analytics, and bioinformatics demand more memory and storage capacity, new proposals advocate for shared storage and memory across multiple servers. For large capacity needs, these systems often use switches to increase available memory or storage slots, pooling resources among multiple servers to meet dynamic capacity requirements over time. Some even allow direct GPU-initiated NVMe access. Smart-Infinity is well-suited for this trend. Expansion using switches increases the number of storage device slots but not the raw link bandwidth to the host. More devices for capacity worsen the link bandwidth bottleneck. Smart-Infinity, using CSDs, leverages increased internal bandwidth and computational capability with more capacity. As system architecture evolves toward resource sharing, solutions like Smart-Infinity are expected to become prevalent.

### Related Work

#### Near-Data Processing

* Near-data processing has been explored for DRAM, SRAM, NVRAM, and storage. Early research integrated DRAM with logic in the 90s. With the rise of 3D stacked memories, logic dies stacked with memory dies were utilized for various applications. Recently, the discontinuity of 3D stacked memory led academia and industry back to DDR variants, sometimes modifying die-internal circuitry, using buffer chips on DIMMs, or both, supported by commercial products.

* Near-data processing for storage devices began with separate accelerators near the IO subsystem. Early ideas utilized embedded cores for computational approaches, focusing on database and big data workloads, with many follow-up approaches. However, embedded cores often lacked computational power, prompting dedicated accelerators. ASICs were used for genome sequence analysis, private information retrieval, and ML queries. FPGAs enhanced flexibility for similar issues. Notable works include GradPIM and OptimStore, targeting DNN training but assuming dedicated memory/storage dies, making them less practical and only implemented in simulators without considering real system integration issues.

* Commercial products with FPGA accelerators in SSD packages for near-storage processing include SmartSSD, used for DB workloads and sorting, and deployed in commercial clouds for DB scans.  Some works use CSD for near-storage processing to accelerate neural network training, mainly focusing on preprocessing training data or embedding tables to reduce storage overhead. To our knowledge, Smart-Infinity is the first to use multiple CSDs to mitigate system interconnect bottlenecks in DNN training, building on SmartSSD but not limited to specific products.

#### DNN Training Acceleration

* As deep learning models grow and training times lengthen, many efforts aim to reduce training time. Distributed training uses multiple workers. Data parallelism replicates the entire model for each worker to process training batches independently. When model size exceeds worker memory limits, model parallelism splits the model across workers. Pipelining addresses low worker utilization in model parallelism by dividing minibatches into smaller microbatches and overlapping them.

* In distributed training, server communication becomes a bottleneck. Gradient compression reduces communication overhead, effectively reducing training time while maintaining accuracy, even in large models. Gradient compression includes gradient sparsification and low-rank decomposition. Sparsification sends portions of gradients based on magnitude, with proven convergence and comparable accuracy. Low-rank decomposition reduces communication volume by factorizing gradient matrices into smaller low-rank matrices, using the power iteration method to reduce decomposition overhead. Both approaches use error compensation, memorizing errors from compression and adding them to gradients at the next step found that error compensation doesn't apply to the Adam optimizer due to nonlinearity, so it preconditions the variance term after a warm-up period to equate it to momentum SGD.

## Conclusion

* Smart-Infinity, a novel CSD-based framework to accelerate storage-offloaded LLM training. Using FPGA accelerators in each CSD, Smart-Infinity efficiently addresses the transfer bottleneck in LLM training. We introduce transfer optimizations and CSD-assisted gradient compression to enhance the system's speedup. Smart-Infinity is fully integrated into PyTorch with off-the-shelf products, making it a ready-to-use solution. Experimental results show significant speedup and good scalability with an increased number of CSDs.


