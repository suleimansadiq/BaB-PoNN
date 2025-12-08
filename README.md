# BaB-PoNN: Posit-8 Branch-and-Bound Robustness Verification

This repository contains a **single branch-and-bound (BaB) robustness-verification framework**, instantiated for **two fixed posit-8 neural-network architectures** trained on MNIST:

- **MNIST MLP (posit-8)**
- **LeNet-5 (posit-8)**

Both variants implement **posit-8 arithmetic with quire-8 accumulation**, using the **SoftPosit** library.  
The verifier checks **input-space robustness** under a **joint perturbation budget**:

- `kmax` — maximum number of pixels that may be changed  
- `bmax` — maximum total number of bit flips across all changed pixels  

A **counterexample** is any perturbed image that satisfies both budgets and changes the model’s **clean posit-8 prediction**.

---

## 1. Repository Structure

A **single BaB framework** is implemented and instantiated for **two architectures**.

### 1.1 `MNIST-MLP/`

Contains the BaB instantiation for a fixed posit-8 MNIST MLP.

Includes:

- `posit_bab_verify_mlp_replay*.c` — standalone MLP verifier
- Posit-8 weights and bias `.bin` files
- MNIST data:
  - `mnist_images_u8.bin`
  - `mnist_labels_u8.bin`
- `driver/` folder with shell scripts for batch verification runs  
  - Scripts must be made executable, for example:
    ```bash
    chmod +x driver/*.sh
    ```

### 1.2 `LeNet5-8/`

Contains the BaB instantiation for a fixed posit-8 LeNet-5.

Includes:

- `posit_bab_verify_lenet5_replay*.c` — standalone LeNet-5 verifier
- Posit-8 conv + FC layer binaries
- MNIST data:
  - `mnist_images_u8.bin`
  - `mnist_labels_u8.bin`
- `driver/` folder with batch / parallel scripts  
  - Also make executable with:
    ```bash
    chmod +x driver/*.sh
    ```

Each verifier C file is a **self-contained executable**: it loads the posit-8 model, reads MNIST, and runs the same BaB robustness framework specialized to that architecture.

---

## 2. Dependencies

### 2.1 GCC / build tools

On Ubuntu / Debian:

```bash
sudo apt update
sudo apt install build-essential
2.2 SoftPosit (posit-8 + quire-8 arithmetic)
The framework uses SoftPosit for posit and quire arithmetic.

Project:

https://gitlab.com/cerlane/SoftPosit

Typical installation:

bash
Copy code
git clone https://gitlab.com/cerlane/SoftPosit.git
cd SoftPosit
make
sudo make install
This typically installs:

softposit.h → /usr/local/include

libsoftposit.a → /usr/local/lib

2.3 GMP (GNU Multiple Precision Arithmetic Library)
Required by the search framework (integer bookkeeping etc.):

bash
Copy code
sudo apt install libgmp-dev
The verifiers also link against:

-lm (math)

-pthread (threads)

3. Required Data Files
Each architecture folder expects the following files to be present.

3.1 MNIST binary dumps
mnist_images_u8.bin

10,000 MNIST test images

each image is 28×28, stored as 784 bytes (uint8)

mnist_labels_u8.bin

10,000 labels in {0,…,9} (uint8)

3.2 Posit-8 model binaries
Exact filenames may vary slightly, but typical LeNet-5 files are:

conv1_W_p8.bin, conv1_b_p8.bin

conv2_W_p8.bin, conv2_b_p8.bin

fc1_W_p8.bin, fc1_b_p8.bin

fc2_W_p8.bin, fc2_b_p8.bin

fc3_W_p8.bin, fc3_b_p8.bin

Typical MLP files:

fc1_W_p8.bin, fc1_b_p8.bin

fc2_W_p8.bin, fc2_b_p8.bin

fc3_W_p8.bin, fc3_b_p8.bin

These files contain posit-8 encoded weights/biases and are read directly by the C code.

4. Building the Verifiers
4.1 Increase stack size (recommended)
The BaB recursion can be deep. Before compiling or running, it is recommended to run:

bash
Copy code
ulimit -s unlimited
This should be done in the same shell from which you invoke gcc and the verifier binaries.

4.2 Example build: LeNet5-8 verifier
From inside LeNet5-8/:

bash
Copy code
ulimit -s unlimited

gcc -O3 -std=c11 posit_bab_verify_lenet5_replay_nolace.c \
    -o posit_bab_verify_lenet5_replay_nolace \
    -I/usr/local/include -L/usr/local/lib \
    -march=native -flto -fomit-frame-pointer -DNDEBUG \
    -lgmp -lm -pthread -l:libsoftposit.a
Depending on your system, you may need -lsoftposit instead of -l:libsoftposit.a.

4.3 Example build: MNIST-MLP verifier
From inside MNIST-MLP/:

bash
Copy code
ulimit -s unlimited

gcc -O3 -std=c11 posit_bab_verify_mlp_replay_nolace.c \
    -o posit_bab_verify_mlp_replay_nolace \
    -I/usr/local/include -L/usr/local/lib \
    -march=native -flto -fomit-frame-pointer -DNDEBUG \
    -lgmp -lm -pthread -l:libsoftposit.a
5. Running the Verifiers
Both architecture-specific verifiers share the same high-level CLI.

5.1 Example run (LeNet5-8)
bash
Copy code
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
This will:

Load MNIST image index 26 and its label

Compute the clean posit-8 logits and top-1 prediction (y_ref)

Search for a perturbed input within (kmax=2, bmax=4) that changes the prediction

Use influence-based pixel ranking with topx=200 and widening factor 1.5

Stop if the recursion depth hits 500000000 or the time limit hits 100000 seconds

Use greedy warm-start heuristics (--greedy)

Log progress at verbosity level 2

6. Command-Line Options (Summary)
The exact set of options is parsed via getopt_long in the C files.
Common options include:

--idx <N>
MNIST test index (0–9999).

--kmax <K>
Max number of pixels allowed to change (K budget).

--bmax <B>
Max total number of bit flips across all changed pixels (B budget).

--xrc <r0-r1,c0-c1>
Restrict perturbations to a rectangular region of the image (rows r0–r1, columns c0–c1).

--topx <X>
Restrict symbolic pixels to the X most influential ones (based on influence metric).

--widen <W>
Widening factor applied to optimistic bounds (W >= 1.0).

--depth <D>
Recursion depth limit for the BaB search.

--timelimit <T>
Wall-clock time limit in seconds.

--nodelimit <N>
Stop after visiting N BaB nodes.

--idlelimit <S>
Stop if no improvement in bounds occurs for S seconds.

--idle-eps <eps>
Small epsilon for deciding whether an improvement is significant.

--greedy
Enable greedy warm-start with both byte-level and bit-level strategies.

--greedy-byte
Only greedy byte-level warm start.

--greedy-bit
Only greedy bit-level warm start.

--no-greedy, --no-greedy-byte, --no-greedy-bit
Disable the corresponding greedy modes.

--verbose <1|2|3>
Verbosity level:

1: info

2: debug

3: trace

--progress <N>
Periodic progress logging every N nodes (if implemented in the driver).

--rank-fast
Use a fast influence-based ranking routine for candidate pixels.

--roi-heur <H> or --roi-heur <HxW>
Automatically choose an H×W block with largest influence as the region of interest.

--no-root-bound
Disable the initial root-bound UNSAT check (forces full BaB exploration).

7. Driver Scripts (Batch / Parallel Verification)
Each architecture folder contains a driver/ directory, with scripts such as:

run_local_robustness_mlp.sh

run_local_robustness_lenet5.sh

run_global_robustness_*.sh

These scripts typically:

Compile the corresponding verifier binary (if needed).

Loop over a range of MNIST indices.

Launch multiple verification jobs in parallel (controlled by MAX_JOBS or similar).

Store individual logs under a logs/ directory.

Aggregate results into a summary .csv file.

Before using the scripts:

bash
Copy code
cd driver
chmod +x *.sh
Then, for example:

bash
Copy code
./run_local_robustness_lenet5.sh
8. License
This codebase is made available under the MIT License, suitable for research and academic use, redistribution, and extension.
See the LICENSE file in the repository root for full terms.

9. Citation
If you use this framework in your research or publications, please cite the associated paper.

A full BibTeX entry can be added once the final paper and venue details are available.


