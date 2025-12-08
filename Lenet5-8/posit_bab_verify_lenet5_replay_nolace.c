/*
 * posit_bab_verify_lenet5_replay_nolace.c
 *
 * Posit-8 branch-and-bound (BaB) robustness verifier for a fixed LeNet-5
 * classifier on MNIST, using exact posit-8 arithmetic with quire-8
 * accumulation via SoftPosit.
 *
 * Author:    Suleiman Sadiq
 * Affiliation: University of Reading, Department of Computer Science
 *
 * --------------------------------------------------------------------------
 * HIGH-LEVEL PURPOSE
 * --------------------------------------------------------------------------
 * This tool performs *input-space* robustness verification for a trained
 * posit-8 LeNet-5 model on MNIST under a *mixed* perturbation budget:
 *
 *   - K-budget (kmax): maximum number of pixels that may be changed.
 *   - B-budget (bmax): maximum total number of bit flips across all changed
 *                      pixels (in the 8-bit input byte representation).
 *
 * The adversary operates directly on the 8-bit MNIST image (uint8_t in
 * [0..255]). Robustness is measured *with respect to the clean model
 * prediction* y_ref (top-1 class of the unperturbed forward pass):
 *
 *   - Let y_ref be argmax_d f(x)_d on the original image x.
 *   - The adversary chooses a perturbed image x' by flipping bits in up to
 *     kmax pixels, with at most bmax bit flips in total.
 *   - A counterexample is any x' within these budgets for which:
 *
 *           argmax_d f(x')_d != y_ref.
 *
 * The tool either:
 *   - finds such an x' (OUTCOME_SAT / "Counterexample"), or
 *   - proves that no such x' exists within the budgets, possibly up to
 *     time / node / depth / idle limits (OUTCOME_UNSAT or TIMEOUT-like
 *     outcomes).
 *
 * The search is performed by a custom BaB (branch-and-bound) engine over
 * the space of *bit flips* on a selected subset of pixels ("symbolic pixels").
 *
 * --------------------------------------------------------------------------
 * NETWORK ARCHITECTURE
 * --------------------------------------------------------------------------
 * The verifier assumes a fixed LeNet-5-style architecture with hardcoded
 * layer dimensions and quantized posit-8 weights/biases:
 *
 *   - Input: 784-dimensional (28x28 grayscale image, uint8_t [0..255])
 *   - Conv1: 1x28x28 -> 6x28x28, kernel 5x5, stride 1, padding via 32x32
 *             zero-centered normalized input (P8_NORM), ReLU activation,
 *             followed by 2x2 max pooling (stride 2) -> 6x14x14.
 *   - Conv2: 6x14x14 -> 16x10x10, kernel 5x5, stride 1, ReLU activation,
 *             followed by 2x2 max pooling (stride 2) -> 16x5x5.
 *   - FC1: 400 -> 120, ReLU.
 *   - FC2: 120 -> 84, ReLU.
 *   - FC3: 84 -> 10 logits (no softmax).
 *
 * All internal arithmetic in the forward pass uses:
 *   - posit8_t for data (SoftPosit posit-8),
 *   - quire8_t for accumulation (q8_fdp_add / q8_to_p8),
 *   - exact SoftPosit conversion for normalization and double logging.
 *
 * --------------------------------------------------------------------------
 * NUMERICAL SEMANTICS
 * --------------------------------------------------------------------------
 * - Inputs are loaded as uint8_t in [0..255] from MNIST and normalized to
 *   roughly [-1, 1] using:
 *
 *        x_norm = (v - 127.5) / 127.5,   v in {0,...,255},
 *
 *   then converted to posit-8 using convertDoubleToP8 (cached in P8_NORM).
 *
 * - Weights and biases are stored as raw posit-8 bytes (uint8_t) and are
 *   *not* re-quantized at runtime. They are interpreted as posit8_t by
 *   constructing a posit8_t with the same underlying .v value.
 *
 * - All dot-products are evaluated via quire8_t accumulators:
 *       q  = q8_clr(...)
 *       q  = q8_fdp_add(q, w, x)   // fused dot-product add
 *       sum = q8_to_p8(q)
 *       out = p8_add(sum, bias)
 *
 * - ReLU is implemented as p8_relu (compare with zero in posit domain).
 *
 * - All logging of logits uses a cached double conversion table P8_DBL:
 *       p8_to_double_fast(posit8_t x) -> double
 *
 * --------------------------------------------------------------------------
 * EXTERNAL DEPENDENCIES
 * --------------------------------------------------------------------------
 * 1) SoftPosit (C library for posit arithmetic)
 *    - Required for posit8_t, quire8_t, and associated operations:
 *        convertDoubleToP8, convertP8ToDouble,
 *        p8_add, p8_sub, q8_clr, q8_fdp_add, q8_to_p8, etc.
 *    - The header "softposit.h" must be available in the include path.
 *    - The static or shared library (e.g., libsoftposit.a) must be available
 *      in the library path.
 *
 * 2) GMP (GNU Multiple Precision Arithmetic Library)
 *    - Required by some SoftPosit builds (depending on configuration).
 *
 * 3) POSIX and standard C libraries
 *    - <stdint.h>, <stdio.h>, <stdlib.h>, <string.h>, <getopt.h>,
 *      <time.h>, <math.h>, <assert.h>, <pthread> (via -pthread link).
 *
 * 4) Linux-like environment
 *    - CLOCK_MONOTONIC (clock_gettime) used for timing.
 *    - ulimit(1) used to allow deep recursion via stack growth.
 *
 * --------------------------------------------------------------------------
 * BUILD INSTRUCTIONS
 * --------------------------------------------------------------------------
 * The typical build (adjust paths as needed):
 *
 *   # Ensure stack is large enough for BaB recursion (recommended)
 *   ulimit -s unlimited
 *
 *   # Compile (example with SoftPosit installed under /usr/local)
 *   gcc -O3 -std=c11 posit_bab_verify_lenet5_replay_nolace.c \
 *       -o posit_bab_verify_lenet5_replay_nolace \
 *       -I/usr/local/include -L/usr/local/lib \
 *       -march=native -flto -fomit-frame-pointer -DNDEBUG \
 *       -lgmp -lm -pthread -l:softposit.a
 *
 * Notes:
 *   - -march=native and -flto are optional but recommended for performance.
 *   - -fomit-frame-pointer and -DNDEBUG remove extra overhead / asserts.
 *   - -lgmp is required if SoftPosit was built with GMP support.
 *   - The linker flag -l:softposit.a explicitly links the SoftPosit static
 *     library. Adjust to -lsoftposit or a shared lib name as needed.
 *
 * --------------------------------------------------------------------------
 * RUNTIME DATA FILES
 * --------------------------------------------------------------------------
 * The code uses a fixed base directory:
 *
 *   #define DIR_BASE "/home/suleiman/Documents/codes2/bnb/lenet5/"
 *
 * You *must* update DIR_BASE to point to a directory that actually contains
 * the following binary files. Each file is read via must_read(), which
 * enforces exact byte counts (short reads terminate the program).
 *
 * 1) MNIST images and labels (uint8_t, little-endian, flat binary):
 *
 *    F_IMGS  = DIR_BASE "mnist_images_u8.bin"
 *      - Shape: [10000][784]
 *      - Layout: row-major, 10k images, each 28x28 = 784 bytes.
 *      - Values: uint8_t in [0..255].
 *
 *    F_LBLS  = DIR_BASE "mnist_labels_u8.bin"
 *      - Shape: [10000]
 *      - Each label is a single uint8_t digit in [0..9].
 *
 * 2) LeNet-5 posit-8 weights and biases (as raw posit8_t bytes):
 *
 *    F_C1_W  = DIR_BASE "conv1_W_p8.bin"
 *      - Shape: 5*5*1*C1_OUT  (5x5 kernel, 1 input channel, 6 output channels)
 *      - Layout: ((kr*5 + kc)*1 + ic)*C1_OUT + oc
 *
 *    F_C1_b  = DIR_BASE "conv1_b_p8.bin"
 *      - Shape: [C1_OUT] (6 biases)
 *
 *    F_C2_W  = DIR_BASE "conv2_W_p8.bin"
 *      - Shape: 5*5*C1_OUT*C2_OUT  (5x5, 6 input channels, 16 output)
 *      - Layout: (((kr*5 + kc)*C1_OUT + ic)*C2_OUT) + oc
 *
 *    F_C2_b  = DIR_BASE "conv2_b_p8.bin"
 *      - Shape: [C2_OUT] (16 biases)
 *
 *    F_FC1_W = DIR_BASE "fc1_W_p8.bin"
 *      - Shape: 400*FC1_OUT  (400 -> 120)
 *      - Layout: input-major, i*FC1_OUT + j
 *
 *    F_FC1_b = DIR_BASE "fc1_b_p8.bin"
 *      - Shape: [FC1_OUT] (120 biases)
 *
 *    F_FC2_W = DIR_BASE "fc2_W_p8.bin"
 *      - Shape: FC1_OUT*FC2_OUT (120 -> 84)
 *
 *    F_FC2_b = DIR_BASE "fc2_b_p8.bin"
 *      - Shape: [FC2_OUT] (84 biases)
 *
 *    F_FC3_W = DIR_BASE "fc3_W_p8.bin"
 *      - Shape: FC2_OUT*NUM_CLASSES (84 -> 10)
 *
 *    F_FC3_b = DIR_BASE "fc3_b_p8.bin"
 *      - Shape: [NUM_CLASSES] (10 biases)
 *
 * All of these are treated as raw posit-8 bytes. No headers or metadata
 * are expected; the program reads exactly sizeof(array) bytes for each.
 *
 * --------------------------------------------------------------------------
 * ROBUSTNESS SPECIFICATION
 * --------------------------------------------------------------------------
 * Let:
 *   - x      be the original MNIST image at index --idx.
 *   - y_true be its label from F_LBLS (for logging only).
 *   - y_ref  be the top-1 class of the clean posit-8 forward pass (argmax).
 *
 * Define the *clean margin* at input x as:
 *   margin(x) = max_{d != y_ref} f(x)_d - f(x)_{y_ref}.
 *
 * A perturbed image x' is allowed if:
 *   1) It differs from x in at most kmax pixels (K-budget),
 *   2) The total number of bit flips across all 8-bit pixels is <= bmax.
 *
 * The search tries to drive margin(x') > 0 (change of top-1 class).
 * If such x' is found, the run is SAT and a witness is printed.
 * If the BaB bounds (plus budgets) guarantee margin(x') <= 0 for all
 * admissible x', the run is UNSAT (no counterexample).
 *
 * --------------------------------------------------------------------------
 * COMMAND-LINE INTERFACE (OPTIONS & FLAGS)
 * --------------------------------------------------------------------------
 * All options are long-form only (no short flags), parsed via getopt_long.
 * Defaults are chosen for small local robustness runs unless overridden.
 *
 *   --idx <N>              (int, default: 0)
 *       Index of the MNIST test image to verify (0 <= N < 10000).
 *
 *   --kmax <K>             (int, default: 2)
 *       Maximum number of pixels that may be changed (K-budget).
 *       Count is based on how many distinct pixels differ from the original
 *       image after perturbations.
 *
 *   --bmax <B>             (int, default: 16)
 *       Maximum total number of bit flips across all changed pixels
 *       (B-budget). This counts individual bit flips in the 8-bit representation
 *       of the pixel intensities.
 *
 *   --xrc <spec>           (string, default: none)
 *       Region-of-interest (ROI) restriction for candidate pixels. The
 *       specification syntax is:
 *
 *         - "r0-r1,c0-c1" : rectangular block in rows [r0..r1], cols [c0..c1].
 *                            One RRect is expanded into per-row ranges.
 *         - "r,c"         : single pixel at row r, col c.
 *
 *       Multiple --xrc flags can be passed; each adds to the union of allowed
 *       pixels. If no X-spec is provided, the default is the full [0..27]^2
 *       grid. Xspecs are converted to an internal list Xspecs[].
 *
 *   --topx <N>             (int, default: 0)
 *       After ranking candidate pixels by an influence score, keep only the
 *       top N pixels as "symbolic pixels" (symPix). If N <= 0 or N >= number
 *       of available candidates, all candidates are kept. This effectively
 *       controls the subnetwork / patch size.
 *
 *   --widen <W>            (double, default: 2.0)
 *       Multiplicative widening factor used in the optimistic residual
 *       bound. The BaB bound uses:
 *           upper = margin + W * (optimistic_gain),
 *       where W >= 1.0 (values < 1.0 are clamped to 1.0). Larger W makes
 *       the bound looser (less pruning, more conservative).
 *
 *   --depth <D>            (int, default: auto = 8 * nSym)
 *       Maximum recursion depth for the BaB search. If not specified,
 *       the depth limit is set to 8 * nSym, corresponding to at most one
 *       branch decision per bit of each symbolic pixel.
 *
 *   --timelimit <T>        (double, seconds, default: 0.0 = disabled)
 *       Wall-clock time limit. Once elapsed time exceeds T, the search
 *       stops with OUTCOME_TIMEOUT (internally BABS_TIME).
 *
 *   --nodelimit <N>        (long, default: 0 = disabled)
 *       Maximum number of BaB nodes to explore. Each recursive call to
 *       bab_search increments g_nodes_seen; if g_nodes_seen > N, the search
 *       terminates with OUTCOME_NODE.
 *
 *   --idlelimit <T>        (double, seconds, default: 0.0 = disabled)
 *       Idle limit for bound improvement. The solver tracks the best
 *       upper bound g_best_upper_seen and the last time it improved by
 *       more than --idle-eps. If no improvement above --idle-eps is
 *       observed over T seconds, the search terminates with OUTCOME_IDLE.
 *
 *   --idle-eps <eps>       (double, default: 1e-6)
 *       Minimum improvement in the best upper bound that counts as "progress"
 *       for idle-limit purposes.
 *
 *   --greedy               (flag, default: OFF)
 *       Enable both greedy-byte and greedy-bit warm starts. This is a
 *       convenience flag equivalent to enabling both:
 *          --greedy-byte --greedy-bit
 *
 *   --greedy-byte          (flag, default: OFF)
 *       Enable greedy *byte-level* warm start:
 *         - Repeatedly test changing one pixel at a time to any of the 256
 *           byte values that stays within (kmax, bmax).
 *         - At each step, choose the change that maximizes a heuristic
 *           margin-reduction gain.
 *         - If a misclassification is found, the run terminates SAT with
 *           "witness_found_greedy_byte".
 *
 *   --greedy-bit           (flag, default: OFF)
 *       Enable greedy *bit-level* warm start:
 *         - Modelled via a BaBNode, repeatedly choose the bit flip with
 *           highest locally estimated gain using quick_gate_bounds.
 *         - If misclassification is found within budgets, terminates SAT
 *           with "witness_found_greedy_bit".
 *
 *   --no-greedy            (flag)
 *       Disable both greedy-byte and greedy-bit warm starts, even if
 *       previously enabled on the command line.
 *
 *   --no-greedy-byte       (flag)
 *       Disable only the greedy-byte warm start.
 *
 *   --no-greedy-bit        (flag)
 *       Disable only the greedy-bit warm start.
 *
 *   --verbose <v>          (int, default: 1)
 *       Set verbosity level:
 *         v = 1 : INFO-level (LOG_INFO)   -> high-level progress messages.
 *         v = 2 : DEBUG-level (LOG_DEBUG)-> includes pruning / node info.
 *         v = 3 : TRACE-level (LOG_TRACE)-> detailed bound updates, etc.
 *
 *   --progress <N>         (int, default: 0)
 *       Currently reserved / not actively used in the code for periodic
 *       progress printing. Included for potential future extensions.
 *
 *   --rank-fast            (flag, default: OFF)
 *       Control how pixel influence is computed for ranking:
 *         - OFF (default): uses quick_gate_bounds and per-pixel K-gain
 *           (TMP_k_gain) to rank symbolic pixels.
 *         - ON  : uses pre_rank_pixels_fast_u8, which for each candidate
 *                 pixel evaluates the effect of setting that pixel to 0
 *                 and 255 and scores the margin change. Cheaper than a
 *                 full 256-valued sweep.
 *
 *   --roi-heur <spec>      (string, default: disabled)
 *       Automatic ROI block selection heuristic:
 *
 *         spec can be:
 *           - "H"         : HxH block (square),
 *           - "H W"       : HxW block (space separated),
 *           - "HxW" or "HXW" : HxW block (letter 'x' or 'X').
 *
 *       The tool:
 *         1) Computes a per-pixel influence score (pre_rank_pixels_fast_u8),
 *         2) Slides an HxW window over the 28x28 grid,
 *         3) Chooses the block with the highest sum of influence,
 *         4) Restricts Xspec to that block before ranking and BaB.
 *
 *       If provided, this overrides any explicit --xrc setting by resetting
 *       Xspec to the selected block.
 *
 *   --no-root-bound        (flag, default: root bound enabled)
 *       By default, the solver performs a *root-bound* UNSAT check before
 *       entering the full BaB recursion:
 *         - It evaluates the current margin at the root,
 *         - Computes an optimistic residual bound using
 *           optimistic_residual_bound_disjoint,
 *         - If margin + rem <= 0, it concludes that no counterexample
 *           exists within budgets and terminates early with UNSAT
 *           ("unsat_by_root_bound").
 *
 *       Passing --no-root-bound disables this initial root UNSAT check;
 *       BaB is then started from the root regardless of the initial bound.
 *
 * --------------------------------------------------------------------------
 * EXIT CODES & STATUS LINE
 * --------------------------------------------------------------------------
 * The program exits via status_and_exit() with a unified "STATUS" line:
 *
 *   STATUS: <OutcomeStr> | best_upper_margin=<v> | elapsed=<t>s |
 *           pixels_changed=<K> | total_bit_flips=<B> |
 *           avg_hamming_per_pixel=<H> [ | patch_rows=... | patch_cols=... ]
 *           [ | note=<note> ]
 *
 * OutcomeStr corresponds to the Outcome enum:
 *   OUTCOME_SAT        -> "Counterexample"
 *   OUTCOME_UNSAT      -> "No counterexample"
 *   OUTCOME_TIMEOUT    -> "TIMEOUT"
 *   OUTCOME_IDLE       -> "TIMEOUT" (idle-stop treated as timeout flavour)
 *   OUTCOME_DEPTH      -> "No counterexample" (stopped by depth limit)
 *   OUTCOME_NODE       -> "No counterexample" (stopped by node limit)
 *
 * Process exit codes:
 *   0  = Run completed normally (any Outcome).
 *   1  = Internal safety check failed (require_or_die triggered).
 *   2  = Bad or invalid command-line arguments (die_args).
 *   3  = I/O error (missing or truncated files, die_io).
 *
 * The STATUS note field indicates the terminating cause, such as:
 *   - "witness_found_greedy_byte"
 *   - "witness_found_greedy_bit"
 *   - "witness_found_bab"
 *   - "unsat_by_root_bound"
 *   - "stopped_by_time_limit"
 *   - "stopped_by_idle_limit"
 *   - "stopped_by_node_limit"
 *   - "stopped_by_depth_limit"
 *
 * When a counterexample is found, the tool prints:
 *   - Clean logits (hex, bits, double),
 *   - Perturbed logits (hex, bits, double),
 *   - Detailed pixel-level changes (hex, bits, doubles, Hamming distance),
 *   - New prediction vs clean ref vs true label,
 *   - A replay forward pass (stage-2) to confirm the result.
 *
 * --------------------------------------------------------------------------
 * TYPICAL USAGE EXAMPLES
 * --------------------------------------------------------------------------
 * 1) Basic local robustness check on a single image (idx=26):
 *
 *   ./posit_bab_verify_lenet5_replay_nolace \
 *       --idx 26 \
 *       --kmax 2 \
 *       --bmax 4 \
 *       --topx 2 \
 *       --widen 1.0 \
 *       --verbose 2 \
 *       --depth 500000000 \
 *       --timelimit 100000 \
 *       --greedy \
 *       --no-root-bound
 *
 * 2) Global robustness with default budgets (no ROI restriction):
 *
 *   ./posit_bab_verify_lenet5_replay_nolace \
 *       --idx 990 \
 *       --kmax 2 \
 *       --bmax 4 \
 *       --topx 0 \
 *       --widen 1.0 \
 *       --verbose 2
 *
 * 3) ROI-limited run on a 28x28 full patch, but using automatic ROI heuristic:
 *
 *   ./posit_bab_verify_lenet5_replay_nolace \
 *       --idx 0 \
 *       --kmax 2 \
 *       --bmax 4 \
 *       --roi-heur 14x14 \
 *       --topx 50 \
 *       --widen 1.5 \
 *       --rank-fast \
 *       --verbose 2
 *
 * 4) Explicitly limiting to a 10x10 block near the center:
 *
 *   ./posit_bab_verify_lenet5_replay_nolace \
 *       --idx 123 \
 *       --kmax 1 \
 *       --bmax 4 \
 *       --xrc 9-18,9-18 \
 *       --topx 30 \
 *       --widen 1.2
 *
 * 5) Pure BaB search without greedy warm starts and with strong bounding:
 *
 *   ./posit_bab_verify_lenet5_replay_nolace \
 *       --idx 42 \
 *       --kmax 2 \
 *       --bmax 8 \
 *       --topx 20 \
 *       --widen 1.0 \
 *       --no-greedy \
 *       --verbose 3
 *
 * --------------------------------------------------------------------------
 * IMPLEMENTATION NOTES
 * --------------------------------------------------------------------------
 * - The "symbolic pixel" list symPix[] is derived from:
 *     a) the ROI (either via --xrc or --roi-heur), and
 *     b) an influence ranking (quick_gate_bounds or pre_rank_pixels_fast_u8),
 *        optionally truncated by --topx.
 *
 * - The "active patch" (g_patch_r0..r1, g_patch_c0..c1) is inferred from
 *   symPix[] and reported in the STATUS line for convenience.
 *
 * - quick_gate_bounds:
 *     * Computes an optimistic per-pixel swing in logits (over all 256
 *       possible byte values).
 *     * Fills pxSwing[t].inc/dec for each pixel t and class.
 *     * Derives:
 *         - TMP_k_gain[t]     : best K-based margin gain if we fully
 *                               change pixel t (arbitrary byte).
 *         - TMP_b_gain[t][bit]: best single-bit margin gain for bit "bit"
 *                               of pixel t.
 *       These are then used both for bounding (optimistic_residual_bound)
 *       and for greedy-bit decisions / BaB branching heuristic.
 *
 * - optimistic_residual_bound_disjoint:
 *     * Given current K/B usage and a set of candidate per-pixel K/B gains,
 *       computes an optimistic upper bound on residual gain from remaining
 *       budget, assuming disjoint usage of pixel-level and bit-level gains.
 *
 * - BaB search:
 *     * At each node:
 *         1) Evaluate current prediction and margin.
 *         2) If misclassified: return SAT and log witness.
 *         3) Compute local bounds via quick_gate_bounds.
 *         4) Compute upper bound = margin + widened residual bound.
 *         5) If upper <= 0: prune (UNSAT branch).
 *         6) If depth limit exhausted: stop (DEPTH).
 *         7) Otherwise branch on the best bit (highest TMP_b_gain).
 *
 * - Greedy warm starts:
 *     * Are purely optional heuristics. They can quickly find a witness in
 *       some cases, reducing BaB work. They must obey (kmax, bmax), and any
 *       result is confirmed via a replay forward pass.
 *
 * --------------------------------------------------------------------------
 * IMPORTANT: BEFORE RUNNING
 * --------------------------------------------------------------------------
 * 1) Ensure DIR_BASE is set to the correct directory containing all required
 *    MNIST and LeNet-5 posit dumps.
 * 2) Confirm that the files have the exact sizes expected by must_read().
 * 3) Install and link SoftPosit properly (include path and library path).
 * 4) Use "ulimit -s unlimited" before running for deep BaB recursion.
 * 5) Be aware that some configurations (large kmax/bmax, large topx, strict
 *    widen, high depth) can lead to very long runtimes unless pruned by
 *    bounds or timelimit/node/idle limits.
 */


#define _GNU_SOURCE
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <time.h>
#include <math.h>

#include "softposit.h"

/* ---- LeNet-5 fixed shapes ---- */
enum {
    C1_OUT = 6,         /* conv1 out channels */
    C2_OUT = 16,        /* conv2 out channels */
    FC1_OUT = 120,
    FC2_OUT = 84,
    NUM_CLASSES = 10
};

/* ---------- Outcomes & exit handling ---------- */
typedef enum {
    OUTCOME_SAT = 0,        /* witness found */
    OUTCOME_UNSAT = 1,      /* proof within budgets (incl. root bound) */
    OUTCOME_TIMEOUT = 2,    /* wall-clock time limit hit */
    OUTCOME_IDLE = 3,       /* idle-limit stop */
    OUTCOME_DEPTH = 4,      /* depth-limit stop */
    OUTCOME_NODE = 5        /* node-limit stop */
} Outcome;

/* Exit codes:
 *   0 = ran ok (any outcome)
 *   2 = invalid/bad args
 *   3 = I/O error (missing files / short reads)
 *   1 = safety check failed (internal invariant)
 */

static double g_run_t0 = 0.0;
static double g_best_upper_seen = -1e300; /* best margin+rem seen overall */
static int    g_last_K_used = 0;          /* pixels changed in last SAT */
static int    g_last_B_used = 0;          /* total bit flips in last SAT */

/* Active patch/ROI (derived from symPix) */
static int g_patch_has = 0;
static int g_patch_r0  = 0;
static int g_patch_r1  = 27;
static int g_patch_c0  = 0;
static int g_patch_c1  = 27;

static inline double now_s(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

static void status_and_exit(Outcome oc, const char* note){
    double elapsed = now_s() - g_run_t0;
    const char* s;
    switch (oc){
        case OUTCOME_SAT:     s = "Counterexample";    break;
        case OUTCOME_UNSAT:   s = "No counterexample"; break;
        case OUTCOME_TIMEOUT: s = "TIMEOUT";           break;
        case OUTCOME_IDLE:    s = "TIMEOUT";           break;
        case OUTCOME_DEPTH:   s = "No counterexample"; break;
        case OUTCOME_NODE:    s = "No counterexample"; break;
        default:              s = "UNKNOWN";           break;
    }
    double avg_ham = (g_last_K_used>0)
                   ? ((double)g_last_B_used / (double)g_last_K_used)
                   : 0.0;

    fprintf(stdout,
        "STATUS: %s | best_upper_margin=%.6f | elapsed=%.3fs | "
        "pixels_changed=%d | total_bit_flips=%d | avg_hamming_per_pixel=%.6f",
        s, g_best_upper_seen, elapsed,
        g_last_K_used, g_last_B_used, avg_ham);

    if (g_patch_has){
        fprintf(stdout,
                " | patch_rows=%d-%d | patch_cols=%d-%d",
                g_patch_r0, g_patch_r1, g_patch_c0, g_patch_c1);
    }

    if (note && *note){
        fprintf(stdout, " | note=%s", note);
    }
    fprintf(stdout, "\n");
    fflush(stdout);
    exit(0);
}

static void die_args(const char* msg){
    fprintf(stderr, "Bad args: %s\n", msg?msg:"");
    fflush(stderr);
    exit(2);
}
static void die_io(const char* fn, const char* msg){
    if (fn) fprintf(stderr, "I/O error on %s: %s\n", fn, msg?msg:"");
    else    fprintf(stderr, "I/O error: %s\n", msg?msg:"");
    fflush(stderr);
    exit(3);
}
static void require_or_die(int cond, const char* why){
    if (!cond){
        fprintf(stderr, "SAFETY CHECK FAILED: %s\n", why?why:"");
        fflush(stderr);
        exit(1);
    }
}

/* ---------- Logging ---------- */
typedef enum { LOG_INFO=1, LOG_DEBUG=2, LOG_TRACE=3 } LogLevel;
static LogLevel g_verbosity = LOG_INFO;

#define LOGI(fmt,...) do{ \
    if(g_verbosity>=LOG_INFO){ \
        fprintf(stdout,"[%7.3fs] " fmt "\n", now_s()-g_run_t0, ##__VA_ARGS__); \
        fflush(stdout); \
    } \
}while(0)

#define LOGD(fmt,...) do{ \
    if(g_verbosity>=LOG_DEBUG){ \
        fprintf(stdout,"[%7.3fs] " fmt "\n", now_s()-g_run_t0, ##__VA_ARGS__); \
        fflush(stdout); \
    } \
}while(0)

#define LOGT(fmt,...) do{ \
    if(g_verbosity>=LOG_TRACE){ \
        fprintf(stdout,"[%7.3fs] " fmt "\n", now_s()-g_run_t0, ##__VA_ARGS__); \
        fflush(stdout); \
    } \
}while(0)

/* ---------- Pretty helpers ---------- */
static inline void bits8_str(uint8_t v, char out[9]){
    for (int i=7;i>=0;i--) out[7-i] = ((v>>i)&1)?'1':'0';
    out[8]='\0';
}

/* ---- normalization helper ---- */
static inline double byte_to_norm_double(uint8_t v){
    return ((double)v - 127.5) / 127.5;
}

static void print_change_header(void){
    puts("  idx  pix  row col    old_hex        old_bits     old_double     new_hex        new_bits     new_double  ham");
}
static void print_change_row(int idx, int pix, uint8_t oldv, uint8_t newv){
    int row = pix / 28, col = pix % 28;
    char oldb[9], newb[9]; bits8_str(oldv, oldb); bits8_str(newv, newb);
    double oldd = byte_to_norm_double(oldv);
    double newd = byte_to_norm_double(newv);
    printf("  %3d %4d  %3d %3d    0x%02x        %s   %+.6f     0x%02x        %s   %+.6f  %3d\n",
           idx, pix, row, col, oldv, oldb, oldd, newv, newb, newd,
           __builtin_popcount((unsigned)(oldv ^ newv)));
}

/* ---------- Files (paths) ---------- */
#define DIR_BASE "/home/suleiman/Documents/codes2/bnb/lenet5/"

#define F_IMGS  DIR_BASE "mnist_images_u8.bin"
#define F_LBLS  DIR_BASE "mnist_labels_u8.bin"
#define F_C1_W  DIR_BASE "conv1_W_p8.bin"
#define F_C1_b  DIR_BASE "conv1_b_p8.bin"
#define F_C2_W  DIR_BASE "conv2_W_p8.bin"
#define F_C2_b  DIR_BASE "conv2_b_p8.bin"
#define F_FC1_W DIR_BASE "fc1_W_p8.bin"
#define F_FC1_b DIR_BASE "fc1_b_p8.bin"
#define F_FC2_W DIR_BASE "fc2_W_p8.bin"
#define F_FC2_b DIR_BASE "fc2_b_p8.bin"
#define F_FC3_W DIR_BASE "fc3_W_p8.bin"
#define F_FC3_b DIR_BASE "fc3_b_p8.bin"

/* ---------- CLI / Globals ---------- */
static int g_idx=0;
static int g_kmax=2, g_bmax=16, g_topx=0;
static double g_widen = 2.0;
static int    g_depth_limit_cli = -1;   /* -1 = auto (8 * nSym) */
static double g_time_limit_s = 0.0;
static long   g_node_limit   = 0;
static long   g_nodes_seen   = 0;
static double g_idle_limit_s = 0.0;
static double g_idle_eps     = 1e-6;
static double g_last_progress_t = 0.0;
static int    g_greedy_en    = 0;      /* OFF unless --greedy */
static int    g_greedy_byte  = 0;
static int    g_greedy_bit   = 0;
static int    g_progress_every = 0;
static int    g_rank_fast    = 0;

/* ROI heuristics */
static int    g_roi_auto = 0;
static int    g_roi_h    = 0;
static int    g_roi_w    = 0;

/* root bound toggle */
static int    g_no_root_bound = 0;

/* ---------- IO helpers ---------- */
static inline void must_read(const char*fn, void*buf, size_t sz){
    FILE*f=fopen(fn,"rb");
    if(!f) die_io(fn,"open failed");
    size_t n=fread(buf,1,sz,f);
    fclose(f);
    if(n!=sz){
        char msg[96];
        snprintf(msg,sizeof msg,"short read (%zu/%zu)", (size_t)n, sz);
        die_io(fn,msg);
    }
}

/* ---------- Posit8 + Quire8 helpers & caches ---------- */
static posit8_t P8_NORM[256];  /* (v-127.5)/127.5 -> posit8 */
static double   P8_DBL[256];   /* cached convertP8ToDouble */

static void build_norm_table_zero_centered(void){
    for(int v=0; v<256; v++){
        double xd=((double)v - 127.5)/127.5;
        P8_NORM[v]=convertDoubleToP8(xd);
    }
}
static void build_p8_double_table(void){
    for(int v=0; v<256; ++v)
        P8_DBL[v] = convertP8ToDouble((posit8_t){ .v=(uint8_t)v });
}
static inline double p8_to_double_fast(posit8_t x){ return P8_DBL[x.v]; }

static inline int p8_is_NaR(posit8_t x){ return x.v==0x80; }
static inline int p8_gt_pure(posit8_t a, posit8_t b){
    posit8_t d = p8_sub(a,b);
    if (p8_is_NaR(d)) return 0;
    return (d.v!=0x00) && ((d.v & 0x80u)==0);
}
static inline posit8_t p8_relu(posit8_t x){
    posit8_t z={.v=0x00};
    return p8_gt_pure(x,z)? x : z;
}
static inline posit8_t p8_max(posit8_t a, posit8_t b){
    return p8_gt_pure(a,b) ? a : b;
}

/* ---------- LeNet-5 forward (posit8 + quire8) ---------- */
static void forward_posit8_quire_logits_lenet5(
    const uint8_t img_u8[784],
    const uint8_t * restrict C1_W, const uint8_t * restrict C1_b,
    const uint8_t * restrict C2_W, const uint8_t * restrict C2_b,
    const uint8_t * restrict FC1_W, const uint8_t * restrict FC1_b,
    const uint8_t * restrict FC2_W, const uint8_t * restrict FC2_b,
    const uint8_t * restrict FC3_W, const uint8_t * restrict FC3_b,
    posit8_t outL[NUM_CLASSES])
{
    /* 32x32 padded input */
    posit8_t Xpad[32*32];
    for (int i=0;i<32*32;i++) Xpad[i] = (posit8_t){ .v = 0x00 };
    for (int r=0;r<28;r++){
        for (int c=0;c<28;c++){
            Xpad[(r+2)*32 + (c+2)] = P8_NORM[ img_u8[r*28+c] ];
        }
    }

    /* Conv1: 5x5, inC=1, outC=C1_OUT -> 28x28xC1_OUT */
    posit8_t C1[28*28*C1_OUT];
    posit8_t A1[28*28*C1_OUT];
    posit8_t P1[14*14*C1_OUT];

    for (int orow=0; orow<28; orow++){
        for (int ocol=0; ocol<28; ocol++){
            for (int oc=0; oc<C1_OUT; oc++){
                quire8_t q = q8_clr((quire8_t){0});
                for (int kr=0; kr<5; kr++){
                    for (int kc=0; kc<5; kc++){
                        posit8_t x = Xpad[(orow+kr)*32 + (ocol+kc)];
                        uint8_t w8 = C1_W[((kr*5 + kc)*1 + 0)*C1_OUT + oc];
                        q = q8_fdp_add(q, (posit8_t){ .v = w8 }, x);
                    }
                }
                posit8_t sum = q8_to_p8(q);
                posit8_t acc = p8_add(sum, (posit8_t){ .v = C1_b[oc] });
                C1[(orow*28 + ocol)*C1_OUT + oc] = acc;
            }
        }
    }
    for (int i=0;i<28*28*C1_OUT;i++) A1[i] = p8_relu(C1[i]);

    /* MaxPool 2x2 stride 2 -> 14x14xC1_OUT */
    for (int or=0; or<14; or++){
        for (int oc=0; oc<14; oc++){
            for (int ch=0; ch<C1_OUT; ch++){
                int r0 = or*2, c0 = oc*2;
                posit8_t v00 = A1[( (r0+0)*28 + (c0+0) )*C1_OUT + ch];
                posit8_t v01 = A1[( (r0+0)*28 + (c0+1) )*C1_OUT + ch];
                posit8_t v10 = A1[( (r0+1)*28 + (c0+0) )*C1_OUT + ch];
                posit8_t v11 = A1[( (r0+1)*28 + (c0+1) )*C1_OUT + ch];
                posit8_t m0 = p8_max(v00, v01);
                posit8_t m1 = p8_max(v10, v11);
                P1[(or*14 + oc)*C1_OUT + ch] = p8_max(m0, m1);
            }
        }
    }

    /* Conv2: 5x5, inC=C1_OUT, outC=C2_OUT -> 10x10xC2_OUT */
    posit8_t C2[10*10*C2_OUT];
    posit8_t A2[10*10*C2_OUT];
    posit8_t P2[5*5*C2_OUT];

    for (int orow=0; orow<10; orow++){
        for (int ocol=0; ocol<10; ocol++){
            for (int oc=0; oc<C2_OUT; oc++){
                quire8_t q = q8_clr((quire8_t){0});
                for (int ic=0; ic<C1_OUT; ic++){
                    for (int kr=0; kr<5; kr++){
                        for (int kc=0; kc<5; kc++){
                            posit8_t x = P1[((orow+kr)*14 + (ocol+kc))*C1_OUT + ic];
                            uint8_t w8 = C2_W[ (((kr*5 + kc)*C1_OUT + ic)*C2_OUT) + oc ];
                            q = q8_fdp_add(q, (posit8_t){ .v = w8 }, x);
                        }
                    }
                }
                posit8_t sum = q8_to_p8(q);
                posit8_t acc = p8_add(sum, (posit8_t){ .v = C2_b[oc] });
                C2[(orow*10 + ocol)*C2_OUT + oc] = acc;
            }
        }
    }
    for (int i=0;i<10*10*C2_OUT;i++) A2[i] = p8_relu(C2[i]);

    /* MaxPool 2x2 stride 2 -> 5x5xC2_OUT */
    for (int or=0; or<5; or++){
        for (int oc=0; oc<5; oc++){
            for (int ch=0; ch<C2_OUT; ch++){
                int r0 = or*2, c0 = oc*2;
                posit8_t v00 = A2[( (r0+0)*10 + (c0+0) )*C2_OUT + ch];
                posit8_t v01 = A2[( (r0+0)*10 + (c0+1) )*C2_OUT + ch];
                posit8_t v10 = A2[( (r0+1)*10 + (c0+0) )*C2_OUT + ch];
                posit8_t v11 = A2[( (r0+1)*10 + (c0+1) )*C2_OUT + ch];
                posit8_t m0 = p8_max(v00, v01);
                posit8_t m1 = p8_max(v10, v11);
                P2[(or*5 + oc)*C2_OUT + ch] = p8_max(m0, m1);
            }
        }
    }

    /* FC1: 400 -> 120 */
    posit8_t FC0[5*5*C2_OUT];
    for (int i=0;i<5*5*C2_OUT;i++) FC0[i] = P2[i];

    posit8_t F1[FC1_OUT];
    for (int j=0;j<FC1_OUT;j++){
        quire8_t q = q8_clr((quire8_t){0});
        for (int i=0;i<5*5*C2_OUT;i++){
            posit8_t x = FC0[i];
            uint8_t w8 = FC1_W[i*FC1_OUT + j];
            q = q8_fdp_add(q, (posit8_t){ .v = w8 }, x);
        }
        posit8_t sum = q8_to_p8(q);
        F1[j] = p8_relu( p8_add(sum, (posit8_t){ .v = FC1_b[j] }) );
    }

    /* FC2: 120 -> 84 */
    posit8_t F2[FC2_OUT];
    for (int j=0;j<FC2_OUT;j++){
        quire8_t q = q8_clr((quire8_t){0});
        for (int i=0;i<FC1_OUT;i++){
            posit8_t x = F1[i];
            uint8_t w8 = FC2_W[i*FC2_OUT + j];
            q = q8_fdp_add(q, (posit8_t){ .v = w8 }, x);
        }
        posit8_t sum = q8_to_p8(q);
        F2[j] = p8_relu( p8_add(sum, (posit8_t){ .v = FC2_b[j] }) );
    }

    /* FC3: 84 -> 10 logits */
    for (int j=0; j<NUM_CLASSES; j++){
        quire8_t q = q8_clr((quire8_t){0});
        for (int i=0;i<FC2_OUT;i++){
            posit8_t x = F2[i];
            uint8_t w8 = FC3_W[i*NUM_CLASSES + j];
            q = q8_fdp_add(q, (posit8_t){ .v = w8 }, x);
        }
        posit8_t sum = q8_to_p8(q);
        outL[j] = p8_add(sum, (posit8_t){ .v = FC3_b[j] });
    }
}

static int argmax10_pure(const posit8_t L[NUM_CLASSES]){
    int best=0;
    for (int d=1; d<NUM_CLASSES; d++)
        if (p8_gt_pure(L[d], L[best])) best=d;
    return best;
}

static void print_logits_all(const char* title, const posit8_t L[NUM_CLASSES]){
    printf("%s (hex):", title);
    for (int d=0; d<NUM_CLASSES; d++) printf("  0x%02x", L[d].v);
    puts("");
    printf("%s (bits):", title);
    for (int d=0; d<NUM_CLASSES; d++){
        char b[9]; bits8_str(L[d].v, b); printf(" %s", b);
    }
    puts("");
    printf("%s (double):", title);
    for (int d=0; d<NUM_CLASSES; d++) printf(" %+.6f", p8_to_double_fast(L[d]));
    puts("");
}

/* ---------- Replay helper ---------- */
static void replay_forward_and_log(
    const uint8_t *img_u8,
    const uint8_t *C1_W,const uint8_t *C1_b,
    const uint8_t *C2_W,const uint8_t *C2_b,
    const uint8_t *FC1_W,const uint8_t *FC1_b,
    const uint8_t *FC2_W,const uint8_t *FC2_b,
    const uint8_t *FC3_W,const uint8_t *FC3_b,
    int yref, int true_label, const char* note)
{
    puts("----- REPLAY FORWARD PASS (stage-2) -----");
    posit8_t Lr[NUM_CLASSES];
    forward_posit8_quire_logits_lenet5(img_u8,
        C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,FC2_W,FC2_b,FC3_W,FC3_b, Lr);
    print_logits_all("replay logits", Lr);
    int pred_r = argmax10_pure(Lr);
    fprintf(stdout,
            "[%7.3fs] replay prediction: %d (clean ref %d, true label %d)%s%s\n",
            now_s()-g_run_t0, pred_r, yref, true_label,
            (note && *note) ? " | note=" : "",
            (note && *note) ? note : "");
    puts("----- END REPLAY -----");
}

/* ---------- Region selection (Xspec) ---------- */
typedef struct { int r0,r1,c0,c1; } RRect;
static RRect Xspecs[4096];
static int   Xn = 0;

static void add_Xspec_rc(const char*s){
    int r0,r1,c0,c1;
    if (sscanf(s,"%d-%d,%d-%d",&r0,&r1,&c0,&c1)==4){
        if(r0<0) r0=0; if(r1>27) r1=27;
        if(c0<0) c0=0; if(c1>27) c1=27;
        if (r0>r1 || c0>c1) return;
        for(int r=r0;r<=r1;r++){
            int a=r*28+c0, b=r*28+c1;
            if (Xn<4096) Xspecs[Xn++] = (RRect){ .r0=a, .r1=b, .c0=0, .c1=0 };
        }
    } else {
        int r,c;
        if (sscanf(s,"%d,%d",&r,&c)==2){
            if (r>=0 && r<28 && c>=0 && c<28){
                int idx=r*28+c;
                if (Xn<4096) Xspecs[Xn++] = (RRect){ .r0=idx, .r1=idx, .c0=0, .c1=0 };
            }
        }
    }
}
static int in_X(int i){
    for(int t=0;t<Xn;t++) if (i>=Xspecs[t].r0 && i<=Xspecs[t].r1) return 1;
    return 0;
}
static void reset_Xspec_to_list(const int *pix, int n){
    Xn=0;
    for(int i=0;i<n;i++){
        int p=pix[i];
        if (p>=0 && p<784 && Xn<4096)
            Xspecs[Xn++] = (RRect){ .r0=p, .r1=p, .c0=0, .c1=0 };
    }
}

/* ---------- FAST RANK helper (optional pre-ranking) ----------
 * Cheap influence probe: per pixel, try extremes {0,255} and score margin change.
 */
static void pre_rank_pixels_fast_u8(
    const uint8_t *img_u8,
    const uint8_t *C1_W,const uint8_t *C1_b,
    const uint8_t *C2_W,const uint8_t *C2_b,
    const uint8_t *FC1_W,const uint8_t *FC1_b,
    const uint8_t *FC2_W,const uint8_t *FC2_b,
    const uint8_t *FC3_W,const uint8_t *FC3_b,
    int yref,
    const int *cand, int nCand,
    double *out_infl)
{
    posit8_t Lc[NUM_CLASSES];
    forward_posit8_quire_logits_lenet5(img_u8,
        C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,FC2_W,FC2_b,FC3_W,FC3_b, Lc);

    double Ly = p8_to_double_fast(Lc[yref]);
    double br = -1e300;
    for (int d=0; d<NUM_CLASSES; ++d){
        if (d==yref) continue;
        double vd = p8_to_double_fast(Lc[d]);
        if (vd > br) br = vd;
    }

    for (int t=0; t<nCand; ++t){
        int p = cand[t];
        double best_gain = 0.0;

        uint8_t tmp[784];
        memcpy(tmp, img_u8, 784);

        /* v = 0 */
        tmp[p] = 0;
        posit8_t L2[NUM_CLASSES];
        forward_posit8_quire_logits_lenet5(tmp,
            C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,FC2_W,FC2_b,FC3_W,FC3_b, L2);
        double Ly2 = p8_to_double_fast(L2[yref]);
        double br2 = -1e300;
        for (int d=0; d<NUM_CLASSES; ++d){
            if (d==yref) continue;
            double vd = p8_to_double_fast(L2[d]);
            if (vd > br2) br2 = vd;
        }
        double g0 = (br2 - br) + (Ly - Ly2);
        if (g0 > best_gain) best_gain = g0;

        /* v = 255 */
        memcpy(tmp, img_u8, 784);
        tmp[p] = 255;
        forward_posit8_quire_logits_lenet5(tmp,
            C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,FC2_W,FC2_b,FC3_W,FC3_b, L2);
        Ly2 = p8_to_double_fast(L2[yref]);
        br2 = -1e300;
        for (int d=0; d<NUM_CLASSES; ++d){
            if (d==yref) continue;
            double vd = p8_to_double_fast(L2[d]);
            if (vd > br2) br2 = vd;
        }
        double g1 = (br2 - br) + (Ly - Ly2);
        if (g1 > best_gain) best_gain = g1;

        out_infl[t] = (best_gain > 0.0 ? best_gain : 0.0);
    }
}

/* ---------- ROI heuristic selector ---------- */
static void choose_roi_block_heuristic(
    int h, int w,
    const uint8_t img[784],
    const uint8_t *C1_W,const uint8_t *C1_b,
    const uint8_t *C2_W,const uint8_t *C2_b,
    const uint8_t *FC1_W,const uint8_t *FC1_b,
    const uint8_t *FC2_W,const uint8_t *FC2_b,
    const uint8_t *FC3_W,const uint8_t *FC3_b,
    int yref,
    int *out_r0, int *out_c0, int *out_r1, int *out_c1)
{
    int    cand[784];
    double infl[784];
    for (int i=0;i<784;i++) cand[i]=i;

    pre_rank_pixels_fast_u8(img,
        C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,FC2_W,FC2_b,FC3_W,FC3_b,
        yref, cand, 784, infl);

    double best_score = -1.0;
    int best_r0 = 0, best_c0 = 0;

    for (int r0=0; r0<=28-h; ++r0){
        for (int c0=0; c0<=28-w; ++c0){
            double s = 0.0;
            for (int dr=0; dr<h; ++dr){
                int r = r0 + dr;
                int base = r * 28;
                for (int dc=0; dc<w; ++dc){
                    int c = c0 + dc;
                    int pix = base + c;
                    s += infl[pix];
                }
            }
            if (s > best_score){
                best_score = s;
                best_r0 = r0;
                best_c0 = c0;
            }
        }
    }

    *out_r0 = best_r0;
    *out_c0 = best_c0;
    *out_r1 = best_r0 + h - 1;
    *out_c1 = best_c0 + w - 1;
}

/* ---------- Bounds (quick-gate style) ---------- */
typedef struct { double inc[NUM_CLASSES]; double dec[NUM_CLASSES]; } SwingPerClass;

/* Global temp buffers (reused) */
static SwingPerClass TMP_pxSwing[2048];
static double        TMP_k_gain[2048];
static double        TMP_b_gain[2048][8];
static double        TMP_Lc_opt[NUM_CLASSES];

static void quick_gate_bounds(
    const uint8_t *img,
    const uint8_t *C1_W,const uint8_t *C1_b,
    const uint8_t *C2_W,const uint8_t *C2_b,
    const uint8_t *FC1_W,const uint8_t *FC1_b,
    const uint8_t *FC2_W,const uint8_t *FC2_b,
    const uint8_t *FC3_W,const uint8_t *FC3_b,
    int yref,
    const int *symPix,int nSym,
    SwingPerClass *pxSwing,
    double *k_gain,
    double b_gain[][8],
    double *Lc_opt)
{
    posit8_t Lc[NUM_CLASSES];
    forward_posit8_quire_logits_lenet5(img,
        C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,FC2_W,FC2_b,FC3_W,FC3_b, Lc);
    for(int d=0; d<NUM_CLASSES; d++)
        Lc_opt[d]=p8_to_double_fast(Lc[d]);

    for(int t=0;t<nSym;t++){
        int p = symPix[t];

        double minv[NUM_CLASSES], maxv[NUM_CLASSES];
        for(int d=0; d<NUM_CLASSES; d++){
            minv[d]=+1e300;
            maxv[d]=-1e300;
        }
        posit8_t Ltmp[NUM_CLASSES];

        uint8_t tmp[784];
        memcpy(tmp, img, 784);
        uint8_t orig = tmp[p];

        for (int v=0; v<256; v++){
            tmp[p] = (uint8_t)v;
            forward_posit8_quire_logits_lenet5(tmp,
                C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,FC2_W,FC2_b,FC3_W,FC3_b, Ltmp);
            for(int d=0; d<NUM_CLASSES; d++){
                double lv = p8_to_double_fast(Ltmp[d]);
                if (lv<minv[d]) minv[d]=lv;
                if (lv>maxv[d]) maxv[d]=lv;
            }
        }
        tmp[p] = orig;

        for(int d=0; d<NUM_CLASSES; d++){
            double inc = maxv[d]-Lc_opt[d];
            double dec = Lc_opt[d]-minv[d];
            pxSwing[t].inc[d]= (inc<0?0:inc);
            pxSwing[t].dec[d]= (dec<0?0:dec);
        }

        /* single-bit gains */
        for(int bit=0; bit<8; bit++){
            tmp[p] = img[p] ^ (1u<<bit);
            forward_posit8_quire_logits_lenet5(tmp,
                C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,FC2_W,FC2_b,FC3_W,FC3_b, Ltmp);
            double best_gain=0.0;
            for(int d=0; d<NUM_CLASSES; d++){
                if (d==yref) continue;
                double gain = (p8_to_double_fast(Ltmp[d]) - Lc_opt[d])
                            + (Lc_opt[yref] - p8_to_double_fast(Ltmp[yref]));
                if (gain>best_gain) best_gain=gain;
            }
            b_gain[t][bit] = (best_gain>0?best_gain:0.0);
        }
        tmp[p] = orig;

        double kbest=0.0;
        for(int d=0; d<NUM_CLASSES; d++){
            if (d==yref) continue;
            double s = pxSwing[t].inc[d] + pxSwing[t].dec[yref];
            if (s>kbest) kbest=s;
        }
        k_gain[t]=kbest;
    }
}

/* Disjoint K/B optimistic bound with widening. */
static double optimistic_residual_bound_disjoint(
    const uint8_t *pixel_changed_flags, int changed_bytes, int changed_bits,
    const int* symPix, int nSym,
    const double *k_gain, double b_gain[][8],
    int kmax, int bmax, double widen)
{
    int Kleft = kmax - changed_bytes;
    int Bleft = bmax - changed_bits;
    if (Kleft<=0 && Bleft<=0) return 0.0;

    typedef struct { int t; double g; } Kg;
    Kg *kg = (Kg*)malloc(sizeof(Kg)*nSym);
    int kgN=0;
    for (int t=0; t<nSym; t++){
        if (!pixel_changed_flags[t]) {
            double g = k_gain[t];
            if (g<0) g=0;
            kg[kgN++] = (Kg){ t, g };
        }
    }
    for (int i=0;i<kgN;i++)
        for(int j=i+1;j<kgN;j++)
            if (kg[j].g > kg[i].g){
                Kg tmp=kg[i]; kg[i]=kg[j]; kg[j]=tmp;
            }

    double k_part = 0.0;
    int takeK = (Kleft < kgN ? Kleft : kgN);
    int *picked = (int*)calloc(nSym, sizeof(int));
    for (int i=0; i<takeK; i++){
        k_part += kg[i].g;
        picked[ kg[i].t ] = 1;
    }
    free(kg);

    int total_bits = nSym*8;
    double *bg = (double*)malloc(sizeof(double)*total_bits);
    int bgN=0;
    for (int t=0; t<nSym; t++){
        int eligible = pixel_changed_flags[t] || !picked[t];
        if (!eligible) continue;
        for (int b=0; b<8; b++){
            double g = b_gain[t][b];
            if (g > 0) bg[bgN++] = g;
        }
    }
    for (int i=0;i<bgN;i++)
        for(int j=i+1;j<bgN;j++)
            if (bg[j] > bg[i]){
                double tmp=bg[i]; bg[i]=bg[j]; bg[j]=tmp;
            }
    double b_part = 0.0;
    int takeB = (Bleft < bgN ? Bleft : bgN);
    for (int i=0;i<takeB;i++) b_part += bg[i];

    free(bg);
    free(picked);

    double bound = (k_part + b_part) * (widen>1.0 ? widen : 1.0);
    return bound;
}

/* ---------- BaB node ---------- */
typedef struct {
    uint8_t img_cur[784];
    uint8_t pixel_changed_flags[2048]; /* by index in symPix[] */
    int     changed_bytes;             /* K used */
    int     changed_bits;              /* B used */
    uint8_t considered_bit[2048][8];   /* branching avoid revisits */
} BaBNode;

static int node_apply_bitflip(BaBNode *nd,
                              int pix_global_idx, int sym_pos, int bit,
                              int kmax, int bmax, uint8_t orig_byte)
{
    uint8_t old  = nd->img_cur[pix_global_idx];
    uint8_t newv = old ^ (1u<<bit);

    int additional_bit  = ((old ^ newv)&(1u<<bit))? 1:0;
    int additional_byte = (!nd->pixel_changed_flags[sym_pos] && (newv!=orig_byte)) ? 1 : 0;

    if (nd->changed_bits  + additional_bit  > bmax) return 0;
    if (nd->changed_bytes + additional_byte > kmax) return 0;

    nd->img_cur[pix_global_idx] = newv;
    nd->changed_bits  += additional_bit;
    if (newv!=orig_byte) nd->pixel_changed_flags[sym_pos]=1;

    int K=0;
    for(int i=0;i<2048;i++) K += nd->pixel_changed_flags[i];
    nd->changed_bytes = K;
    return 1;
}

/* ---------- (Optional) Greedy warm starts (default OFF) ---------- */
static int greedy_byte_warm_start(
    const uint8_t *img_orig,
    const uint8_t *C1_W,const uint8_t *C1_b,
    const uint8_t *C2_W,const uint8_t *C2_b,
    const uint8_t *FC1_W,const uint8_t *FC1_b,
    const uint8_t *FC2_W,const uint8_t *FC2_b,
    const uint8_t *FC3_W,const uint8_t *FC3_b,
    int yref, int true_label,
    const int* symPix, int nSym,
    int kmax, int bmax,
    int* outK, int* outB, int* outPred)
{
    uint8_t cur[784];
    memcpy(cur, img_orig, 784);
    uint8_t byte_changed[2048]={0};
    int K=0, B=0;

    posit8_t L_clean0[NUM_CLASSES];
    forward_posit8_quire_logits_lenet5(img_orig,
        C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,FC2_W,FC2_b,FC3_W,FC3_b, L_clean0);

    for (;;){
        posit8_t L[NUM_CLASSES];
        forward_posit8_quire_logits_lenet5(cur,
            C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,FC2_W,FC2_b,FC3_W,FC3_b, L);
        int pred = argmax10_pure(L);
        if (pred != yref){
            LOGI("Greedy BYTE warm start: SAT (K=%d,B=%d).", K,B);
            print_logits_all("clean logits", L_clean0);
            print_logits_all("post-perturb logits", L);
            printf("changes: %d bytes, %d bit flips\n", K,B);
            if (K){
                print_change_header();
                for(int t=0;t<nSym;t++){
                    int p=symPix[t];
                    uint8_t old=img_orig[p], nw=cur[p];
                    if (old!=nw) print_change_row(t, p, old, nw);
                }
            }
            printf("new prediction: %d (clean ref %d, true %d)\n",
                   pred, yref, true_label);

            require_or_die(K<=kmax, "K_used exceeds kmax (greedy byte)");
            require_or_die(B<=bmax, "B_used exceeds bmax (greedy byte)");
            require_or_die(pred!=yref, "greedy pred_after == clean ref");

            if (outK) *outK=K;
            if (outB) *outB=B;
            if (outPred) *outPred=pred;

            replay_forward_and_log(cur,
                C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,FC2_W,FC2_b,FC3_W,FC3_b,
                yref,true_label,"witness_found_greedy_byte");
            return 1;
        }
        if (K>=kmax && B>=bmax) return 0;

        double best_gain = 0.0;
        int    best_t=-1;
        uint8_t best_v=0;

        for (int t=0; t<nSym; t++){
            int p = symPix[t];
            uint8_t old = cur[p];
            for (int v=0; v<256; v++){
                uint8_t nv = (uint8_t)v;
                if (nv == old) continue;
                int ham = __builtin_popcount((unsigned)(old ^ nv));
                int addK = byte_changed[t] ? 0 : 1;
                if (B + ham > bmax || K + addK > kmax) continue;

                uint8_t tmp[784];
                memcpy(tmp, cur, 784);
                tmp[p]=nv;
                posit8_t L2[NUM_CLASSES];
                forward_posit8_quire_logits_lenet5(tmp,
                    C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,FC2_W,FC2_b,FC3_W,FC3_b, L2);

                double Ly  = p8_to_double_fast(L[yref]);
                double Ly2 = p8_to_double_fast(L2[yref]);
                double br  = -1e300, br2=-1e300;
                for(int d=0; d<NUM_CLASSES; d++){
                    if (d==yref) continue;
                    double vd = p8_to_double_fast(L[d]);
                    double v2 = p8_to_double_fast(L2[d]);
                    if (vd>br)  br = vd;
                    if (v2>br2) br2 = v2;
                }
                double gain = (br2 - br) + (Ly - Ly2);
                if (gain > best_gain){
                    best_gain=gain; best_t=t; best_v=nv;
                }
            }
        }

        if (best_t<0 || best_gain <= 0.0) return 0;

        int p = symPix[best_t];
        uint8_t old = cur[p];
        int ham = __builtin_popcount((unsigned)(old ^ best_v));
        int addK = byte_changed[best_t] ? 0 : 1;
        cur[p]=best_v;
        B += ham;
        if (!byte_changed[best_t]){ byte_changed[best_t]=1; K++; }
    }
}

static int greedy_bit_warm_start(
    const uint8_t *img_orig,
    const uint8_t *C1_W,const uint8_t *C1_b,
    const uint8_t *C2_W,const uint8_t *C2_b,
    const uint8_t *FC1_W,const uint8_t *FC1_b,
    const uint8_t *FC2_W,const uint8_t *FC2_b,
    const uint8_t *FC3_W,const uint8_t *FC3_b,
    int yref, int true_label,
    const int* symPix, int nSym,
    int kmax, int bmax,
    int* outK, int* outB, int* outPred)
{
    BaBNode nd;
    memcpy(nd.img_cur, img_orig, 784);
    memset(nd.pixel_changed_flags, 0, sizeof(nd.pixel_changed_flags));
    memset(nd.considered_bit, 0, sizeof(nd.considered_bit));
    nd.changed_bits  = 0;
    nd.changed_bytes = 0;

    posit8_t L_clean0[NUM_CLASSES];
    forward_posit8_quire_logits_lenet5(img_orig,
        C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,FC2_W,FC2_b,FC3_W,FC3_b, L_clean0);

    for (;;){
        posit8_t L[NUM_CLASSES];
        forward_posit8_quire_logits_lenet5(nd.img_cur,
            C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,FC2_W,FC2_b,FC3_W,FC3_b, L);
        int pred = argmax10_pure(L);
        if (pred != yref){
            LOGI("Greedy BIT warm start: SAT (K=%d,B=%d).",
                 nd.changed_bytes, nd.changed_bits);
            print_logits_all("clean logits", L_clean0);
            print_logits_all("post-perturb logits", L);
            printf("changes: %d bytes, %d bit flips\n",
                   nd.changed_bytes, nd.changed_bits);
            if (nd.changed_bytes){
                print_change_header();
                for(int t=0;t<nSym;t++){
                    int p=symPix[t];
                    uint8_t old=img_orig[p], nw=nd.img_cur[p];
                    if (old!=nw) print_change_row(t, p, old, nw);
                }
            }
            printf("new prediction: %d (clean ref %d, true %d)\n",
                   pred, yref, true_label);

            require_or_die(nd.changed_bytes<=kmax,"K_used exceeds kmax (greedy bit)");
            require_or_die(nd.changed_bits <=bmax,"B_used exceeds bmax (greedy bit)");
            require_or_die(pred!=yref,"greedy pred_after == clean ref");

            if (outK) *outK=nd.changed_bytes;
            if (outB) *outB=nd.changed_bits;
            if (outPred) *outPred=pred;

            replay_forward_and_log(nd.img_cur,
                C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,FC2_W,FC2_b,FC3_W,FC3_b,
                yref,true_label,"witness_found_greedy_bit");
            return 1;
        }
        if (nd.changed_bytes>=kmax && nd.changed_bits>=bmax) return 0;

        quick_gate_bounds(nd.img_cur,
            C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,FC2_W,FC2_b,FC3_W,FC3_b,
            yref,
            symPix,nSym,
            TMP_pxSwing, TMP_k_gain, TMP_b_gain, TMP_Lc_opt);

        double best=-1.0;
        int best_t=-1, best_b=-1;
        for(int t=0;t<nSym;t++){
            for(int b=0;b<8;b++){
                double g = TMP_b_gain[t][b];
                if (g>best){
                    best=g; best_t=t; best_b=b;
                }
            }
        }
        if (best_t<0 || best<=0.0) return 0;

        int p = symPix[best_t];
        if (!node_apply_bitflip(&nd, p, best_t, best_b,
                                kmax,bmax, img_orig[p]))
            return 0;
    }
}

/* ---------- BaB search ---------- */
typedef enum {
    BABS_UNSAT = 0,
    BABS_SAT   = 1,
    BABS_TIME  = 2,
    BABS_IDLE  = 3,
    BABS_NODE  = 4,
    BABS_DEPTH = 5
} BabResult;

static BabResult bab_search(
    BaBNode *nd,
    const uint8_t *img_orig,
    const uint8_t *C1_W,const uint8_t *C1_b,
    const uint8_t *C2_W,const uint8_t *C2_b,
    const uint8_t *FC1_W,const uint8_t *FC1_b,
    const uint8_t *FC2_W,const uint8_t *FC2_b,
    const uint8_t *FC3_W,const uint8_t *FC3_b,
    int yref, int true_label,
    const int* symPix, int nSym,
    int kmax, int bmax,
    int depth_limit, double widen)
{
    double tnow = now_s();
    if (g_time_limit_s>0.0 && tnow-g_run_t0 > g_time_limit_s){
        LOGI("STOP: time limit.");
        return BABS_TIME;
    }
    if (g_idle_limit_s>0.0 && tnow-g_last_progress_t > g_idle_limit_s){
        LOGI("STOP: idle limit (no bound improvement).");
        return BABS_IDLE;
    }
    if (g_node_limit>0 && ++g_nodes_seen > g_node_limit){
        LOGI("STOP: node limit.");
        return BABS_NODE;
    }

    posit8_t L[NUM_CLASSES];
    forward_posit8_quire_logits_lenet5(nd->img_cur,
        C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,FC2_W,FC2_b,FC3_W,FC3_b, L);
    int pred = argmax10_pure(L);

    if (pred != yref){
        LOGI("Counterexample found: BaB witness (K=%d,B=%d).",
             nd->changed_bytes, nd->changed_bits);

        require_or_die(nd->changed_bytes<=kmax,"K_used exceeds kmax (BaB)");
        require_or_die(nd->changed_bits <=bmax,"B_used exceeds bmax (BaB)");
        require_or_die(pred!=yref,"pred_after == clean ref (BaB)");

        g_last_K_used = nd->changed_bytes;
        g_last_B_used = nd->changed_bits;

        posit8_t Lc0[NUM_CLASSES];
        forward_posit8_quire_logits_lenet5(img_orig,
            C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,FC2_W,FC2_b,FC3_W,FC3_b, Lc0);
        print_logits_all("clean logits", Lc0);
        print_logits_all("post-perturb logits", L);

        printf("changes: %d bytes, %d bit flips\n",
               nd->changed_bytes, nd->changed_bits);
        if (nd->changed_bytes){
            print_change_header();
            for(int t=0;t<nSym;t++){
                int p=symPix[t];
                uint8_t old=img_orig[p], nw=nd->img_cur[p];
                if (old!=nw) print_change_row(t, p, old, nw);
            }
        }
        printf("new prediction: %d (clean ref %d, true %d)\n",
               pred, yref, true_label);

        replay_forward_and_log(nd->img_cur,
            C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,FC2_W,FC2_b,FC3_W,FC3_b,
            yref,true_label,"witness_found_bab");
        return BABS_SAT;
    }

    /* margin at this node */
    double Ly = p8_to_double_fast(L[yref]);
    double best_rival=-1e300;
    for(int d=0; d<NUM_CLASSES; d++){
        if (d==yref) continue;
        double vd = p8_to_double_fast(L[d]);
        if (vd>best_rival) best_rival=vd;
    }
    double cur_margin = best_rival - Ly;

    /* local bounds */
    quick_gate_bounds(nd->img_cur,
        C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,FC2_W,FC2_b,FC3_W,FC3_b,
        yref,
        symPix,nSym,
        TMP_pxSwing, TMP_k_gain, TMP_b_gain, TMP_Lc_opt);

    double rem = optimistic_residual_bound_disjoint(
        nd->pixel_changed_flags, nd->changed_bytes, nd->changed_bits,
        symPix, nSym, TMP_k_gain, TMP_b_gain, kmax,bmax, widen);

    double upper = cur_margin + rem;
    if (upper > g_best_upper_seen + g_idle_eps){
        g_best_upper_seen = upper;
        g_last_progress_t = now_s();
        LOGT("progress: best_upper=%.6f (margin=%.6f, rem=%.6f)",
             g_best_upper_seen, cur_margin, rem);
    }

    if (cur_margin + rem <= 0){
        LOGD("PRUNE: margin=%.6f, optimistic_rem(widened)=%.6f -> cannot exceed 0.",
             cur_margin, rem);
        return BABS_UNSAT;
    }

    if (depth_limit<=0){
        LOGD("STOP: depth limit.");
        return BABS_DEPTH;
    }

    /* choose best bit */
    double best=-1.0;
    int best_t=-1, best_b=-1;
    for(int tt=0; tt<nSym; tt++){
        for(int b=0; b<8; b++){
            if (nd->considered_bit[tt][b]) continue;
            double g = TMP_b_gain[tt][b];
            if (g>best){
                best=g; best_t=tt; best_b=b;
            }
        }
    }
    if (best_t<0) return BABS_UNSAT;
    nd->considered_bit[best_t][best_b]=1;

    /* LEFT: apply flip */
    BaBNode left = *nd;
    {
        int p = symPix[best_t];
        if (node_apply_bitflip(&left, p, best_t, best_b,
                               kmax,bmax, img_orig[p])){
            BabResult r = bab_search(&left, img_orig,
                C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,FC2_W,FC2_b,FC3_W,FC3_b,
                yref,true_label,
                symPix,nSym,kmax,bmax, depth_limit-1, widen);
            if (r!=BABS_UNSAT) return r;
        }
    }

    /* RIGHT: skip flip */
    BaBNode right = *nd;
    BabResult rr = bab_search(&right, img_orig,
        C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,FC2_W,FC2_b,FC3_W,FC3_b,
        yref,true_label,
        symPix,nSym,kmax,bmax, depth_limit-1, widen);
    return rr;
}

/* ---------- Influence sort for topX ---------- */
typedef struct { int pix; double infl; } PixInfl;
static int cmp_infl_desc(const void*a,const void*b){
    double x=((const PixInfl*)a)->infl;
    double y=((const PixInfl*)b)->infl;
    return (x>y)?-1:(x<y)?1:0;
}

/* ---------- Main ---------- */
int main(int argc, char**argv){
    g_run_t0 = now_s();
    setvbuf(stdout, NULL, _IONBF, 0);
    g_last_progress_t = g_run_t0;
    g_best_upper_seen = -1e300;
    g_last_K_used = 0;
    g_last_B_used = 0;

    static struct option Lopt[] = {
        {"idx",         required_argument, 0, 'i'},
        {"kmax",        required_argument, 0, 'k'},
        {"bmax",        required_argument, 0, 'b'},
        {"xrc",         required_argument, 0, 'y'},
        {"topx",        required_argument, 0, 'p'},
        {"widen",       required_argument, 0, 'w'},
        {"depth",       required_argument, 0, 'd'},
        {"timelimit",   required_argument, 0, 'T'},
        {"nodelimit",   required_argument, 0, 'N'},
        {"idlelimit",   required_argument, 0, 2000},
        {"idle-eps",    required_argument, 0, 2001},
        {"greedy",      no_argument,       0, 1000},
        {"greedy-byte", no_argument,       0, 1001},
        {"greedy-bit",  no_argument,       0, 1002},
        {"no-greedy",   no_argument,       0, 1100},
        {"no-greedy-byte", no_argument,    0, 1101},
        {"no-greedy-bit",  no_argument,    0, 1102},
        {"verbose",     required_argument, 0, 'v'},
        {"progress",    required_argument, 0, 'g'},
        {"rank-fast",   no_argument,       0, 2100},
        {"roi-heur",    required_argument, 0, 2200},
        {"no-root-bound", no_argument,     0, 2300},
        {0,0,0,0}
    };

    int opt;
    while((opt=getopt_long(argc,argv,"",Lopt,0))!=-1){
        if(opt=='i') g_idx=atoi(optarg);
        if(opt=='k') g_kmax=atoi(optarg);
        if(opt=='b') g_bmax=atoi(optarg);
        if(opt=='y') add_Xspec_rc(optarg);
        if(opt=='p') g_topx=atoi(optarg);
        if(opt=='w') g_widen=strtod(optarg,NULL);
        if(opt=='d') g_depth_limit_cli=atoi(optarg);
        if(opt=='T') g_time_limit_s=strtod(optarg,NULL);
        if(opt=='N') g_node_limit=strtol(optarg,NULL,10);
        if(opt==2000) g_idle_limit_s=strtod(optarg,NULL);
        if(opt==2001) g_idle_eps=strtod(optarg,NULL);
        if(opt=='v'){
            int vv=atoi(optarg);
            if(vv<1) vv=1;
            if(vv>3) vv=3;
            g_verbosity=(LogLevel)vv;
        }
        if(opt=='g') g_progress_every=atoi(optarg);
        if(opt==1000) g_greedy_en=1;
        if(opt==1001) g_greedy_byte=1;
        if(opt==1002) g_greedy_bit =1;
        if(opt==1100){ g_greedy_en=0; g_greedy_byte=0; g_greedy_bit=0; }
        if(opt==1101){ g_greedy_byte=0; }
        if(opt==1102){ g_greedy_bit =0; }
        if(opt==2100){ g_rank_fast=1; }
        if(opt==2200){
            int h=0,w=0;
            if (sscanf(optarg,"%dx%d",&h,&w)==2 ||
                sscanf(optarg,"%dX%d",&h,&w)==2){
                /* HxW */
            } else if (sscanf(optarg,"%d %d",&h,&w)==2){
                /* H W */
            } else if (sscanf(optarg,"%d",&h)==1){
                w=h;
            } else {
                die_args("bad --roi-heur (expected N, HxW, or \"H W\")");
            }
            if (h<=0 || w<=0 || h>28 || w>28){
                die_args("bad --roi-heur size (must be in 1..28)");
            }
            g_roi_auto = 1;
            g_roi_h = h;
            g_roi_w = w;
        }
        if(opt==2300){
            g_no_root_bound = 1;
        }
    }

    if (g_kmax<0) die_args("kmax must be >=0");
    if (g_bmax<0) die_args("bmax must be >=0");
    if (g_widen<1.0) g_widen=1.0;

    LOGI("init: idx=%d kmax=%d bmax=%d topx=%d widen=%.3f rank_fast=%s",
         g_idx,g_kmax,g_bmax,g_topx,g_widen,
         g_rank_fast ? "on" : "off");

    /* Data */
    static uint8_t imgs[10000][784], lbl[10000];
    must_read(F_IMGS, imgs, 784u*10000u);
    must_read(F_LBLS, lbl, 10000u);

    static uint8_t C1_W[5*5*1*C1_OUT], C1_b[C1_OUT];
    static uint8_t C2_W[5*5*C1_OUT*C2_OUT], C2_b[C2_OUT];
    static uint8_t FC1_W[400*FC1_OUT], FC1_b[FC1_OUT];
    static uint8_t FC2_W[FC1_OUT*FC2_OUT], FC2_b[FC2_OUT];
    static uint8_t FC3_W[FC2_OUT*NUM_CLASSES], FC3_b[NUM_CLASSES];

    must_read(F_C1_W,  C1_W,  sizeof C1_W);
    must_read(F_C1_b,  C1_b,  sizeof C1_b);
    must_read(F_C2_W,  C2_W,  sizeof C2_W);
    must_read(F_C2_b,  C2_b,  sizeof C2_b);
    must_read(F_FC1_W, FC1_W, sizeof FC1_W);
    must_read(F_FC1_b, FC1_b, sizeof FC1_b);
    must_read(F_FC2_W, FC2_W, sizeof FC2_W);
    must_read(F_FC2_b, FC2_b, sizeof FC2_b);
    must_read(F_FC3_W, FC3_W, sizeof FC3_W);
    must_read(F_FC3_b, FC3_b, sizeof FC3_b);

    build_norm_table_zero_centered();
    build_p8_double_table();

    /* Clean logits */
    posit8_t Lc_p[NUM_CLASSES];
    forward_posit8_quire_logits_lenet5(imgs[g_idx],
        C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,FC2_W,FC2_b,FC3_W,FC3_b, Lc_p);
    print_logits_all("clean logits", Lc_p);

    int true_label = lbl[g_idx];
    int pred_clean = argmax10_pure(Lc_p);
    int yref = pred_clean; /* robustness is w.r.t. clean prediction */
    LOGI("clean prediction: %d (true label %d)", pred_clean, true_label);

    /* ROI heuristic if enabled */
    if (g_roi_auto){
        int r0,c0,r1,c1;
        choose_roi_block_heuristic(g_roi_h,g_roi_w,
                                   imgs[g_idx],
                                   C1_W,C1_b,C2_W,C2_b,
                                   FC1_W,FC1_b,FC2_W,FC2_b,FC3_W,FC3_b,
                                   yref,
                                   &r0,&c0,&r1,&c1);
        Xn = 0;
        char buf[64];
        snprintf(buf,sizeof buf,"%d-%d,%d-%d", r0,r1,c0,c1);
        add_Xspec_rc(buf);
        LOGI("heuristic ROI: %dx%d block at rows %d-%d, cols %d-%d",
             g_roi_h,g_roi_w,r0,r1,c0,c1);
    }

    /* Build candidate symbolic pixel list */
    int symPix_all[2048];
    int nSym_all=0;
    for(int i=0;i<784;i++) if (in_X(i)) symPix_all[nSym_all++]=i;
    if (nSym_all==0){
        for(int i=0;i<784;i++) symPix_all[nSym_all++]=i;
    }

    /* Influence ranking + topx trimming */
    PixInfl infl[2048];
    int inflN=0;
    if (g_rank_fast){
        double *score = (double*)malloc(sizeof(double)*nSym_all);
        pre_rank_pixels_fast_u8(imgs[g_idx],
            C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,FC2_W,FC2_b,FC3_W,FC3_b,
            yref,
            symPix_all,nSym_all,
            score);
        for (int t=0; t<nSym_all; ++t)
            infl[inflN++] = (PixInfl){ .pix=symPix_all[t], .infl=score[t] };
        free(score);
    } else {
        quick_gate_bounds(imgs[g_idx],
            C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,FC2_W,FC2_b,FC3_W,FC3_b,
            yref,
            symPix_all,nSym_all,
            TMP_pxSwing, TMP_k_gain, TMP_b_gain, TMP_Lc_opt);
        for(int t=0;t<nSym_all;t++)
            infl[inflN++] = (PixInfl){ .pix=symPix_all[t], .infl=TMP_k_gain[t] };
    }

    qsort(infl, inflN, sizeof(PixInfl), cmp_infl_desc);
    int take = (g_topx>0 && g_topx<inflN) ? g_topx : inflN;
    int *symPix = (int*)malloc(sizeof(int)*take);
    for(int i=0;i<take;i++) symPix[i]=infl[i].pix;

    LOGI("selection: nSym=%d (from %d), topx=%d, rank_fast=%s",
         take, nSym_all, g_topx, g_rank_fast ? "on" : "off");

    /* derive patch bbox */
    if (take > 0){
        int min_r=27,max_r=0,min_c=27,max_c=0;
        for (int i=0;i<take;i++){
            int p = symPix[i];
            int r = p / 28;
            int c = p % 28;
            if (r<min_r) min_r=r;
            if (r>max_r) max_r=r;
            if (c<min_c) min_c=c;
            if (c>max_c) max_c=c;
        }
        g_patch_has=1;
        g_patch_r0=min_r; g_patch_r1=max_r;
        g_patch_c0=min_c; g_patch_c1=max_c;
        LOGI("active patch: rows %d-%d, cols %d-%d (derived from symPix)",
             g_patch_r0,g_patch_r1,g_patch_c0,g_patch_c1);
    }

    reset_Xspec_to_list(symPix, take);

    /* Greedy warm starts (OFF by default, only if flags set) */
    int gK=0,gB=0,gPred=-1;
    if (g_greedy_en || g_greedy_byte){
        if (greedy_byte_warm_start(imgs[g_idx],
                                   C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,
                                   FC2_W,FC2_b,FC3_W,FC3_b,
                                   yref,true_label,
                                   symPix,take,
                                   g_kmax,g_bmax,
                                   &gK,&gB,&gPred)){
            g_last_K_used = gK;
            g_last_B_used = gB;
            status_and_exit(OUTCOME_SAT, "witness_found_greedy_byte");
        }
    }
    if (g_greedy_en || g_greedy_bit){
        if (greedy_bit_warm_start(imgs[g_idx],
                                  C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,
                                  FC2_W,FC2_b,FC3_W,FC3_b,
                                  yref,true_label,
                                  symPix,take,
                                  g_kmax,g_bmax,
                                  &gK,&gB,&gPred)){
            g_last_K_used = gK;
            g_last_B_used = gB;
            status_and_exit(OUTCOME_SAT, "witness_found_greedy_bit");
        }
    }

    LOGI("Proceeding to BaB (greedy %s).",
         (g_greedy_en||g_greedy_byte||g_greedy_bit)? "enabled" : "disabled");

    /* Root UNSAT bound (optional) */
    BaBNode root;
    memcpy(root.img_cur, imgs[g_idx], 784);
    memset(root.pixel_changed_flags, 0, sizeof(root.pixel_changed_flags));
    memset(root.considered_bit, 0, sizeof(root.considered_bit));
    root.changed_bits  = 0;
    root.changed_bytes = 0;

    if (!g_no_root_bound){
        posit8_t Lroot[NUM_CLASSES];
        forward_posit8_quire_logits_lenet5(root.img_cur,
            C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,FC2_W,FC2_b,FC3_W,FC3_b, Lroot);
        double Ly  = p8_to_double_fast(Lroot[yref]);
        double br  = -1e300;
        for(int d=0; d<NUM_CLASSES; d++){
            if (d==yref) continue;
            double vd = p8_to_double_fast(Lroot[d]);
            if (vd>br) br=vd;
        }
        double cur_margin = br - Ly;

        quick_gate_bounds(root.img_cur,
            C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,FC2_W,FC2_b,FC3_W,FC3_b,
            yref,
            symPix,take,
            TMP_pxSwing, TMP_k_gain, TMP_b_gain, TMP_Lc_opt);

        double rem = optimistic_residual_bound_disjoint(
            root.pixel_changed_flags, root.changed_bytes, root.changed_bits,
            symPix,take, TMP_k_gain, TMP_b_gain, g_kmax,g_bmax, g_widen);

        double upper0 = cur_margin + rem;
        if (upper0 > g_best_upper_seen + g_idle_eps){
            g_best_upper_seen = upper0;
            g_last_progress_t = now_s();
        }

        if (cur_margin + rem <= 0.0){
            LOGI("====== UNSAT by root bound ======");
            LOGI("margin=%.6f, optimistic_rem(widened)=%.6f (sum <= 0).",
                 cur_margin, rem);
            status_and_exit(OUTCOME_UNSAT, "unsat_by_root_bound");
        }
    } else {
        LOGI("Skipping root bound check (--no-root-bound).");
    }

    int depth_limit = (g_depth_limit_cli>0) ? g_depth_limit_cli : (8*take);
    LOGI("BaB search starting (K<=%d, B<=%d, nSym=%d, depth_limit=%d, widen=%.2f)...",
         g_kmax,g_bmax,take,depth_limit,g_widen);

    BabResult br = bab_search(&root, imgs[g_idx],
                              C1_W,C1_b,C2_W,C2_b,FC1_W,FC1_b,
                              FC2_W,FC2_b,FC3_W,FC3_b,
                              yref,true_label,
                              symPix,take,
                              g_kmax,g_bmax,
                              depth_limit,g_widen);

    switch (br){
        case BABS_SAT:
            status_and_exit(OUTCOME_SAT, "witness_found_bab");
            break;
        case BABS_UNSAT:
            LOGI("No counterexample within budgets (BaB): no misclassification found.");
            status_and_exit(OUTCOME_UNSAT,"no_counterexample_within_budgets");
            break;
        case BABS_TIME:
            LOGI("Stopped without proof (time limit).");
            status_and_exit(OUTCOME_TIMEOUT,"stopped_by_time_limit");
            break;
        case BABS_IDLE:
            LOGI("Stopped without proof (idle limit).");
            status_and_exit(OUTCOME_IDLE,"stopped_by_idle_limit");
            break;
        case BABS_NODE:
            LOGI("Stopped without proof (node limit).");
            status_and_exit(OUTCOME_NODE,"stopped_by_node_limit");
            break;
        case BABS_DEPTH:
            LOGI("Stopped without proof (depth limit).");
            status_and_exit(OUTCOME_DEPTH,"stopped_by_depth_limit");
            break;
        default:
            LOGI("Stopped for unknown reason.");
            status_and_exit(OUTCOME_NODE,"unknown_bab_result");
            break;
    }
}
