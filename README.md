# Nvidia CUDA + Rust 

This project demonstrates how to run a simple CUDA kernel (`saxpy`) from Rust using the [`cudarc`](https://crates.io/crates/cudarc) and [`nvtx`](https://crates.io/crates/nvtx) crates. We also profile execution with NVIDIA Nsight Systems (`nsys`) and Nsight Compute (`ncu`).

---

## Prerequisites

* **Hardware**: NVIDIA GPU (RTX 3070 or similar) with CUDA compute capability ≥ 8.6
* **OS**: Arch Linux
* **Rust**: Installed via [rustup](https://rustup.rs/)
* **CUDA Toolkit**: Version 13.0.1 (installed manually)
* **Nsight Systems**: Version 2025.3.2
* **Nsight Compute**: Version 2025.3.1

---

## Setting up CUDA on Arch Linux

Arch doesn’t ship `nsys` and `ncu` directly. We had to grab the Debian `.deb` installer from NVIDIA and unpack it manually.

### 1. Install `dpkg` so Arch can handle `.deb` files

```bash
sudo pacman -S dpkg
```

### 2. Download the CUDA Debian 12 local installer

```bash
wget https://developer.download.nvidia.com/compute/cuda/13.0.1/local_installers/cuda-repo-debian12-13-0-local_13.0.1-580.82.07-1_amd64.deb
```

### 3. Extract the `.deb` without installing

```bash
dpkg-deb -x cuda-repo-debian12-13-0-local_13.0.1-580.82.07-1_amd64.deb extracted/
```

### 4. Move binaries into `/opt`

The `.deb` gives us Nsight tools under `extracted/opt`. Place them under `/opt` so they’re in expected locations:

```bash
sudo cp -r extracted/opt/* /opt/
```

### 5. Add tools to PATH

```bash
export PATH=/opt/nsight-systems/2025.3.2/target-linux-x64:$PATH
export PATH=/opt/nsight-compute/2025.3.1:$PATH
```

Add those lines to your `~/.zshrc` or `~/.bashrc` for persistence.

Now `nsys` and `ncu` should be available:

```bash
nsys --version
ncu --version
```

---

## Building and Running the Project

### Clone and build

```bash
git clone https://github.com/asoldo/nvidia_cuda
cd nvidia_cuda
cargo build --release
```

### Run normally

```bash
./target/release/nvidia_cuda
```

You should see:

```
SAXPY complete on 1000000 elements. Example out[12345] = 30862.5
```

### Profile with Nsight Systems

```bash
nsys profile -o saxpy-nvtx ./target/release/nvidia_cuda
nsys stats saxpy-nvtx.nsys-rep
```

### Profile with Nsight Compute

```bash
sudo ncu ./target/release/nvidia_cuda
```

---

## Project Layout

* `Cargo.toml` – dependencies (`cudarc`, `nvtx`, `anyhow`)
* `src/main.rs` – SAXPY kernel in CUDA C, compiled at runtime with NVRTC, launched from Rust, wrapped with NVTX ranges for profiling

---

## Notes

* `nvtx` ranges allow clear separation of **H2D copies**, **kernel execution**, and **D2H copy** in `nsys` reports.
* `ncu` provides GPU occupancy, memory throughput, and bottleneck analysis.
* The SAXPY kernel here is minimal: `out[i] = a * x[i] + y[i]`.

---

## Example Outputs

**Nsight Compute (ncu)**:

* Memory Throughput: ~86%
* Achieved Occupancy: ~53%
* Suggested Local Speedup: ~20–33%

**Nsight Systems (nsys)**:

* NVTX timeline clearly shows H2D copies → kernel launch → D2H copy.

---

## saxpy-nvtx.nsys-rep 

```sh
NOTICE: Existing SQLite export found: saxpy-nvtx.sqlite
        It is assumed file was previously exported from: saxpy-nvtx.nsys-rep
        Consider using --force-export=true if needed.

Processing [saxpy-nvtx.sqlite] with [/opt/nsight-systems/2025.3.2/host-linux-x64/reports/nvtx_sum.py]... 

 ** NVTX Range Summary (nvtx_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)   Style       Range    
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  -------  -------------
     60.6          976,403          1  976,403.0  976,403.0   976,403   976,403          0.0  PushPop  :D2H copy    
     38.3          616,135          1  616,135.0  616,135.0   616,135   616,135          0.0  PushPop  :H2D copies  
      1.1           17,908          1   17,908.0   17,908.0    17,908    17,908          0.0  PushPop  :saxpy kernel

Processing [saxpy-nvtx.sqlite] with [/opt/nsight-systems/2025.3.2/host-linux-x64/reports/osrt_sum.py]... 

 ** OS Runtime Summary (osrt_sum):

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)            Name         
 --------  ---------------  ---------  -------------  -------------  -----------  -----------  ------------  ----------------------
     47.2      229,316,112         22   10,423,459.6    4,859,633.0        3,617   68,057,026  16,305,656.2  p oll                  
     29.1      141,388,640        626      225,860.4       10,434.0        1,000    8,230,568     551,464.5  i octl                 
     22.6      110,019,055          1  110,019,055.0  110,019,055.0  110,019,055  110,019,055           0.0  p thread_cond_timedwait
      0.4        2,133,249         50       42,665.0        6,680.0        3,026      824,847     142,135.3  m map64                
      0.2        1,119,247         18       62,180.4       58,193.0        6,671      299,171      64,905.5  s em_timedwait         
      0.1          475,093         18       26,394.1        5,255.5        1,807      126,720      42,735.0  m map                  
      0.1          471,367         15       31,424.5        1,787.0        1,173      224,307      76,954.7  c lose                 
      0.1          450,848          2      225,424.0      225,424.0       98,767      352,081     179,120.0  p thread_join          
      0.0          210,696         10       21,069.6        4,126.0        1,460      172,833      53,378.3  m unmap                
      0.0          165,946         44        3,771.5        3,154.0        1,549       21,982       2,936.3  o pen64                
      0.0          144,706         28        5,168.1        2,420.5        1,137       33,872       7,680.8  f open                 
      0.0           94,154          3       31,384.7       30,912.0       22,062       41,180       9,567.8  p thread_create        
      0.0           83,890          1       83,890.0       83,890.0       83,890       83,890           0.0  p thread_cond_wait     
      0.0           66,549         19        3,502.6        3,531.0        1,035        8,257       1,784.9  f close                
      0.0           25,012          1       25,012.0       25,012.0       25,012       25,012           0.0  f gets                 
      0.0           16,301          5        3,260.2        1,349.0        1,035        8,605       3,247.8  f read                 
      0.0           14,646          5        2,929.2        3,098.0        1,520        3,682         889.5  o pen                  
      0.0           11,539          7        1,648.4        1,283.0        1,010        3,484         855.8  w rite                 
      0.0            8,923          1        8,923.0        8,923.0        8,923        8,923           0.0  c onnect               
      0.0            8,234          2        4,117.0        4,117.0        4,050        4,184          94.8  s ocket                
      0.0            7,030          2        3,515.0        3,515.0        3,004        4,026         722.7  p ipe2                 
      0.0            4,108          2        2,054.0        2,054.0        1,529        2,579         742.5  r ead                  
      0.0            3,606          1        3,606.0        3,606.0        3,606        3,606           0.0  p thread_kill          
      0.0            3,516          1        3,516.0        3,516.0        3,516        3,516           0.0  m protect              
      0.0            1,970          1        1,970.0        1,970.0        1,970        1,970           0.0  l isten                
      0.0            1,856          1        1,856.0        1,856.0        1,856        1,856           0.0  f cntl                 
      0.0            1,796          1        1,796.0        1,796.0        1,796        1,796           0.0  s ignal                
      0.0            1,636          1        1,636.0        1,636.0        1,636        1,636           0.0  p thread_cond_broadcast
      0.0            1,535          1        1,535.0        1,535.0        1,535        1,535           0.0  b ind                  
      0.0            1,344          1        1,344.0        1,344.0        1,344        1,344           0.0  p thread_cond_signal   
      0.0            1,193          1        1,193.0        1,193.0        1,193        1,193           0.0  f write                

Processing [saxpy-nvtx.sqlite] with [/opt/nsight-systems/2025.3.2/host-linux-x64/reports/cuda_api_sum.py]... 

 ** CUDA API Summary (cuda_api_sum):

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)   Min (ns)  Max (ns)   StdDev (ns)          Name 
       
 --------  ---------------  ---------  -----------  ---------  --------  ---------  -----------  -------------
-------
     63.4        3,169,271          3  1,056,423.7    1,544.0       719  3,167,008  1,827,819.7  cuMemAllocAsy nc     
     19.4          972,567          1    972,567.0  972,567.0   972,567    972,567          0.0  cuMemcpyDtoHA sync_v2
     11.9          592,908          2    296,454.0  296,454.0   293,745    299,163      3,831.1  cuMemcpyHtoDA sync_v2
      1.4           70,910          1     70,910.0   70,910.0    70,910     70,910          0.0  cuStreamDestr oy_v2  
      1.2           62,322          1     62,322.0   62,322.0    62,322     62,322          0.0  cuModuleLoadD ata    
      0.7           33,764          3     11,254.7    7,681.0     3,424     22,659     10,103.2  cuMemsetD8Asy nc     
      0.5           23,225         21      1,106.0      355.0       132     14,218      3,028.8  cuStreamWaitE vent   
      0.4           17,728          1     17,728.0   17,728.0    17,728     17,728          0.0  cuModuleUnloa d      
      0.3           16,932          1     16,932.0   16,932.0    16,932     16,932          0.0  cuStreamCreat e      
      0.2           11,780          1     11,780.0   11,780.0    11,780     11,780          0.0  cuLaunchKerne l      
      0.2            8,581          9        953.4      692.0       259      3,139        884.0  cuEventRecord
      0.1            5,049          1      5,049.0    5,049.0     5,049      5,049          0.0  cuCtxSynchron ize    
      0.1            5,026          6        837.7      222.0       149      3,474      1,316.1  cuEventCreate
      0.1            3,909          3      1,303.0      696.0       669      2,544      1,074.8  cuMemFreeAsyn c      
      0.1            3,368          6        561.3      132.5       115      2,669      1,032.9  cuEventDestro y_v2   
      0.0            1,952          1      1,952.0    1,952.0     1,952      1,952          0.0  cuStreamSynch ronize 
      0.0            1,893          1      1,893.0    1,893.0     1,893      1,893          0.0  cuInit       
      0.0              490          1        490.0      490.0       490        490          0.0  cuCtxSetCurre nt     

Processing [saxpy-nvtx.sqlite] with [/opt/nsight-systems/2025.3.2/host-linux-x64/reports/cuda_gpu_kern_sum.py]
... 

 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)  Name 
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  -----
    100.0           30,272          1  30,272.0  30,272.0    30,272    30,272          0.0  saxpy

Processing [saxpy-nvtx.sqlite] with [/opt/nsight-systems/2025.3.2/host-linux-x64/reports/cuda_gpu_mem_time_sum .py]... 

 ** CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):

 Time (%)  Total Time (ns)  Count  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)           Operation  
        
 --------  ---------------  -----  ---------  ---------  --------  --------  -----------  -------------------- --------
     60.1          819,739      1  819,739.0  819,739.0   819,739   819,739          0.0  [CUDA memcpy Device- to-Host]
     37.6          513,278      2  256,639.0  256,639.0   250,783   262,495      8,281.6  [CUDA memcpy Host-to -Device]
      2.3           31,840      3   10,613.3   10,816.0     9,984    11,040        556.4  [CUDA memset]       

Processing [saxpy-nvtx.sqlite] with [/opt/nsight-systems/2025.3.2/host-linux-x64/reports/cuda_gpu_mem_size_sum .py]... 

 ** CUDA GPU MemOps Summary (by Size) (cuda_gpu_mem_size_sum):

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)           Operation          
 ----------  -----  --------  --------  --------  --------  -----------  ----------------------------
     12.000      3     4.000     4.000     4.000     4.000        0.000  [CUDA memset]               
      8.000      2     4.000     4.000     4.000     4.000        0.000  [CUDA memcpy Host-to-Device]
      4.000      1     4.000     4.000     4.000     4.000        0.000  [CUDA memcpy Device-to-Host]

Processing [saxpy-nvtx.sqlite] with [/opt/nsight-systems/2025.3.2/host-linux-x64/reports/openmp_sum.py]... 
SKIPPED: saxpy-nvtx.sqlite does not contain OpenMP event data.

Processing [saxpy-nvtx.sqlite] with [/opt/nsight-systems/2025.3.2/host-linux-x64/reports/opengl_khr_range_sum.py]... 
SKIPPED: saxpy-nvtx.sqlite does not contain KHR Extension (KHR_DEBUG) data.

Processing [saxpy-nvtx.sqlite] with [/opt/nsight-systems/2025.3.2/host-linux-x64/reports/opengl_khr_gpu_range_sum.py]... 
SKIPPED: saxpy-nvtx.sqlite does not contain GPU KHR Extension (KHR_DEBUG) data.

Processing [saxpy-nvtx.sqlite] with [/opt/nsight-systems/2025.3.2/host-linux-x64/reports/vulkan_marker_sum.py]... 
SKIPPED: saxpy-nvtx.sqlite does not contain Vulkan Debug Extension (Vulkan Debug Util) data.

Processing [saxpy-nvtx.sqlite] with [/opt/nsight-systems/2025.3.2/host-linux-x64/reports/vulkan_gpu_marker_sum.py]... 
SKIPPED: saxpy-nvtx.sqlite does not contain GPU Vulkan Debug Extension (GPU Vulkan Debug markers) data.

Processing [saxpy-nvtx.sqlite] with [/opt/nsight-systems/2025.3.2/host-linux-x64/reports/dx11_pix_sum.py]... 
SKIPPED: saxpy-nvtx.sqlite does not contain DX11 CPU debug markers.

Processing [saxpy-nvtx.sqlite] with [/opt/nsight-systems/2025.3.2/host-linux-x64/reports/dx12_gpu_marker_sum.py]... 
SKIPPED: saxpy-nvtx.sqlite does not contain DX12 GPU debug markers.

Processing [saxpy-nvtx.sqlite] with [/opt/nsight-systems/2025.3.2/host-linux-x64/reports/dx12_pix_sum.py]... 
SKIPPED: saxpy-nvtx.sqlite does not contain DX12 CPU debug markers.

Processing [saxpy-nvtx.sqlite] with [/opt/nsight-systems/2025.3.2/host-linux-x64/reports/wddm_queue_sum.py]... 
SKIPPED: saxpy-nvtx.sqlite does not contain WDDM context data.

Processing [saxpy-nvtx.sqlite] with [/opt/nsight-systems/2025.3.2/host-linux-x64/reports/um_sum.py]... 
SKIPPED: saxpy-nvtx.sqlite does not contain CUDA Unified Memory CPU page faults data.

Processing [saxpy-nvtx.sqlite] with [/opt/nsight-systems/2025.3.2/host-linux-x64/reports/um_total_sum.py]... 
SKIPPED: saxpy-nvtx.sqlite does not contain CUDA Unified Memory CPU page faults data.

Processing [saxpy-nvtx.sqlite] with [/opt/nsight-systems/2025.3.2/host-linux-x64/reports/um_cpu_page_faults_sum.py]... 
SKIPPED: saxpy-nvtx.sqlite does not contain CUDA Unified Memory CPU page faults data.

Processing [saxpy-nvtx.sqlite] with [/opt/nsight-systems/2025.3.2/host-linux-x64/reports/openacc_sum.py]... 
SKIPPED: saxpy-nvtx.sqlite does not contain OpenACC event data.

Processing [saxpy-nvtx.sqlite] with [/opt/nsight-systems/2025.3.2/host-linux-x64/reports/syscall_sum.py]... 
SKIPPED: saxpy-nvtx.sqlite does not contain syscall data.

```

## NCU Output

```
==PROF== Connected to process 1574528 (.../nvidia_cuda/target/release/nvidia_cuda)
==PROF== Profiling "saxpy" - 0: 0%....50%....100% - 8 passes
SAXPY complete on 1000000 elements. Example out[12345] = 30862.5
==PROF== Disconnected from process 1574528
[1574528] nvidia_cuda@127.0.0.1
  saxpy (977, 1, 1)x(1024, 1, 1), Context 1, Stream 13, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         6.79
    SM Frequency                    Ghz         1.50
    Elapsed Cycles                cycle       46,791
    Memory Throughput                 %        86.67
    DRAM Throughput                   %        86.67
    Duration                         us        31.23
    L1/TEX Cache Throughput           %        15.28
    L2 Cache Throughput               %        26.82
    SM Active Cycles              cycle    41,255.37
    Compute (SM) Throughput           %        11.70
    ----------------------- ----------- ------------

    INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of the device.   
          To further improve performance, work will likely need to be shifted from the most utilized to another unit.   
          Start by analyzing DRAM in the Memory Workload Analysis section.                                              

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                 1,024
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                    977
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte            8.19
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              46
    Stack Size                                                 1,024
    Threads                                   thread       1,000,448
    # TPCs                                                        23
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                               21.24
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block            4
    Block Limit Shared Mem                block            8
    Block Limit Warps                     block            1
    Theoretical Active Warps per SM        warp           32
    Theoretical Occupancy                     %        66.67
    Achieved Occupancy                        %        51.94
    Achieved Active Warps Per SM           warp        24.93
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 22.09%                                                                                    
          The difference between calculated theoretical (66.7%) and measured achieved occupancy (51.9%) can be the      
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can   
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices   
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on     
          optimizing occupancy.                                                                                         
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Local Speedup: 33.33%                                                                                    
          The 8.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the       
          hardware maximum of 12. This kernel's theoretical occupancy (66.7%) is limited by the number of warps within  
          each block.                                                                                                   

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle      183,704
    Total DRAM Elapsed Cycles        cycle    1,695,744
    Average L1 Active Cycles         cycle    41,255.37
    Total L1 Elapsed Cycles          cycle    2,137,302
    Average L2 Active Cycles         cycle    41,188.62
    Total L2 Elapsed Cycles          cycle    1,407,648
    Average SM Active Cycles         cycle    41,255.37
    Total SM Elapsed Cycles          cycle    2,137,302
    Average SMSP Active Cycles       cycle    38,970.04
    Total SMSP Elapsed Cycles        cycle    8,549,208
    -------------------------- ----------- ------------
```
