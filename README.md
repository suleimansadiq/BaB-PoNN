# BaB-PoNN: Posit Neural Network Branch-and-Bound Robustness Verification

This repository contains a **single branch-and-bound (BaB) robustness-verification framework**, instantiated for **two fixed posit-8 neural-network architectures** trained on MNIST:

- **MNIST MLP (posit-8)**
- **LeNet-5 (posit-8)**

Both variants implement **posit-8 arithmetic with quire-8 accumulation**, using the **SoftPosit** library. Robustness is measured in **input space** under a joint perturbation budget:

- `kmax` — maximum number of pixels that may be changed  
- `bmax` — maximum total number of bit flips across all changed pixels  

A **counterexample** is any perturbed image that satisfies `(kmax, bmax)` and changes the model’s **clean posit-8 prediction**.

---

## 1. Repository Structure

A single BaB framework is instantiated for two architectures.

### 1.1 `MNIST-MLP/`

BaB instantiation for a fixed posit-8 MNIST MLP.

Contains:

- `posit_bab_verify_mlp_replay*.c` — standalone MLP verifier  
- Posit-8 weights and bias `.bin` files  
- MNIST data:
  - `mnist_images_u8.bin`
  - `mnist_labels_u8.bin`
- `driver/` — shell scripts for batch / parallel verification runs  
  - Make them executable before use, e.g. `chmod +x driver/*.sh`

### 1.2 `LeNet5-8/`

BaB instantiation for a fixed posit-8 LeNet-5.

Contains:

- `posit_bab_verify_lenet5_replay*.c` — standalone LeNet-5 verifier  
- Posit-8 conv + FC layer `.bin` files  
- MNIST data:
  - `mnist_images_u8.bin`
  - `mnist_labels_u8.bin`
- `driver/` — shell scripts for batch / parallel verification runs  
  - Again, make them executable: `chmod +x driver/*.sh`

Each verifier C file is a **self-contained executable**: it loads the posit-8 model, reads MNIST, and runs the same BaB robustness framework specialized to that architecture.

---

## 2. Dependencies

The code assumes the following libraries and tools are installed and available to your C toolchain:

- **C toolchain**  
  - e.g. `gcc`, `make`, C11 support
- **SoftPosit** for posit and quire arithmetic  
  - Project page: <https://gitlab.com/cerlane/SoftPosit>
- **GMP (GNU Multiple Precision Arithmetic Library)**  
  - Used for integer / arithmetic bookkeeping
- Standard system libraries:
  - `libm` (math)
  - `pthread` (POSIX threads)

Include and library paths (e.g. `-I/usr/local/include`, `-L/usr/local/lib`) may need to be adapted to your local installation.

---

## 3. Required Data Files

Each architecture folder expects the following binary dumps to be present.

### 3.1 MNIST binary dumps

- `mnist_images_u8.bin`  
  - 10,000 MNIST test images  
  - each image is 28×28, stored as 784 bytes (`uint8`)

- `mnist_labels_u8.bin`  
  - 10,000 labels in `{0,…,9}` (`uint8`)

### 3.2 Posit-8 model binaries

Exact filenames may vary slightly by directory, but typical LeNet-5 files are:

- `conv1_W_p8.bin`, `conv1_b_p8.bin`  
- `conv2_W_p8.bin`, `conv2_b_p8.bin`  
- `fc1_W_p8.bin`, `fc1_b_p8.bin`  
- `fc2_W_p8.bin`, `fc2_b_p8.bin`  
- `fc3_W_p8.bin`, `fc3_b_p8.bin`  

Typical MLP files:

- `fc1_W_p8.bin`, `fc1_b_p8.bin`  
- `fc2_W_p8.bin`, `fc2_b_p8.bin`  
- `fc3_W_p8.bin`, `fc3_b_p8.bin`  

These contain posit-8 encoded weights and biases and are read directly by the C code.

---

## 4. Building the Verifiers

### 4.1 Stack size

The BaB recursion can be deep. It is recommended to increase the stack limit before compiling/running, e.g.:

```bash
ulimit -s unlimited

4.2 Example build: LeNet5-8

From inside LeNet5-8/:
ulimit -s unlimited

gcc -O3 -std=c11 posit_bab_verify_lenet5_replay_nolace.c \
    -o posit_bab_verify_lenet5_replay_nolace \
    -I/usr/local/include -L/usr/local/lib \
    -march=native -flto -fomit-frame-pointer -DNDEBUG \
    -lgmp -lm -pthread -lsoftposit
Adjust include/library paths and the SoftPosit library name if needed for your system.

4.3 Example build: MNIST-MLP

From inside MNIST-MLP/:
ulimit -s unlimited

gcc -O3 -std=c11 posit_bab_verify_mlp_replay_nolace.c \
    -o posit_bab_verify_mlp_replay_nolace \
    -I/usr/local/include -L/usr/local/lib \
    -march=native -flto -fomit-frame-pointer -DNDEBUG \
    -lgmp -lm -pthread -lsoftposit

5. Running the Verifiers

Both architecture-specific verifiers share the same high-level CLI.

5.1 Example run (LeNet5-8)
./posit_bab_verify_lenet5_replay_nolace \
    --idx 26 \
    --kmax 2 \
    --bmax 4 \
    --topx 200 \
    --widen 1.5 \
    --depth 500000000 \
    --timelimit 100000 \
    --greedy \
    --verbose 2
    
This run:

uses MNIST image index 26

defines budgets (kmax=2, bmax=4)

restricts to the topx=200 most influential symbolic pixels

applies a widening factor 1.5 to bounds

uses an aggressive depth limit and time limit

enables greedy warm-start heuristics

logs at verbosity level 2

6. Command-Line Options (Summary)

The exact options are parsed via getopt_long in the C code. Common flags:

--idx <N>
MNIST test index (0–9999).

--kmax <K>
Maximum number of pixels allowed to change (K budget).

--bmax <B>
Maximum total number of bit flips across all changed pixels (B budget).

--xrc <r0-r1,c0-c1>
Restrict perturbations to a rectangular region (rows r0–r1, cols c0–c1).

--topx <X>
Restrict symbolic pixels to the X most influential candidates.

--widen <W>
Widening factor for optimistic bounds (W ≥ 1.0).

--depth <D>
Recursion depth limit for BaB.

--timelimit <T>
Wall-clock time limit in seconds.

--nodelimit <N>
Stop after visiting N BaB nodes.

--idlelimit <S>
Stop if no meaningful improvement in the bound for S seconds.

--idle-eps <eps>
Threshold for what counts as a meaningful improvement.

--greedy
Enable both greedy byte-level and bit-level warm starts.

--greedy-byte / --greedy-bit
Enable only the corresponding greedy strategy.

--no-greedy, --no-greedy-byte, --no-greedy-bit
Disable the corresponding greedy mode(s).

--verbose <1|2|3>
Logging verbosity (info / debug / trace).

--progress <N>
Periodic progress reporting (if used in your driver).

--rank-fast
Use fast influence-based ranking.

--roi-heur <H> or --roi-heur <HxW>
Automatically choose an H×W ROI block with largest influence.

--no-root-bound
Skip the initial root UNSAT bound, forcing full BaB exploration.

7. Driver Scripts (Batch / Parallel Verification)

Each architecture folder contains a driver/ directory with scripts such as:

run_local_robustness_mlp.sh

run_local_robustness_lenet5.sh

run_global_robustness_*.sh

These scripts typically:

compile the verifier binary (if needed)

loop over a range of MNIST indices

launch multiple verification jobs in parallel

store logs under a logs/ directory

aggregate results into a summary .csv

Before using the scripts:
cd driver
chmod +x *.sh
./*.sh

cd driver
chmod +x *.sh
./run_local_robustness_lenet5.sh

9. Citation

If you use this framework in your research, please cite the associated paper.

A full BibTeX entry will be provided once the final paper and venue details are available.


