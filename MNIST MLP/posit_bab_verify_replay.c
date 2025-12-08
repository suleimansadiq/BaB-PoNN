/*
 * posit_bab_verify_replay.c
 *
 * Posit-8 branch-and-bound (BaB) robustness verifier for a fixed MNIST MLP.
 *
 * Author: Suleiman Sadiq
 * Affiliation: University of Reading, Department of Computer Science
 *
 * --------------------------------------------------------------------------
 * OVERVIEW
 * --------------------------------------------------------------------------
 * This tool searches for input-space adversarial examples for a trained
 * posit-8 MNIST MLP, under a joint budget on:
 *   - number of pixels that may be changed (kmax, a K budget), and
 *   - total number of bit flips across all changed pixels (bmax, a B budget).
 *
 * The network architecture is:
 *   - Input: 784-dimensional (28x28 grayscale image, uint8_t [0..255])
 *   - Hidden layer: 8 neurons, posit-8 weights and biases, ReLU activation
 *   - Output layer: 10 logits, posit-8 weights and biases (no softmax)
 *
 * All arithmetic in the forward pass is done in posit-8 with quire-8
 * accumulation, using SoftPosit.
 *
 * The verifier:
 *   - loads fixed binary files for images, labels, and posit-8 weights,
 *   - selects a subset of pixels to treat as symbolic (symPix),
 *   - explores the search space of allowed bit-flip patterns on those pixels
 *     using branch and bound,
 *   - checks whether the predicted class changes from the clean prediction,
 *   - enforces budgets on K (pixels changed) and B (bit flips),
 *   - proves UNSAT within budgets when search is fully explored,
 *   - or terminates early due to time, idle, depth, or node limits.
 *
 * It also supports several heuristics:
 *   - greedy warm-start search (optional),
 *   - influence-based symbolic pixel ranking (default),
 *   - fast influence pre-ranking based on 0/255 probes (rank-fast),
 *   - heuristic region-of-interest (ROI) block selection (roi-heur).
 *
 * --------------------------------------------------------------------------
 * BUILD INSTRUCTIONS
 * --------------------------------------------------------------------------
 * Requirements:
 *   - C compiler with C11 support (tested with gcc),
 *   - SoftPosit library and headers installed in /usr/local,
 *   - libgmp, libm, pthreads.
 *
 * Recommended build command:
 *
 *   ulimit -s unlimited
 *   gcc -O3 -std=c11 posit_bab_verify_replay.c -o posit_bab_verify_replay \
 *       -I/usr/local/include -L/usr/local/lib \
 *       -march=native -flto -fomit-frame-pointer -DNDEBUG \
 *       -lgmp -lm -pthread -l:softposit.a
 *
 * Notes:
 *   - "ulimit -s unlimited" is recommended before build and runs to avoid
 *     stack overflows in deep recursion.
 *   - -march=native and -flto are optional optimizations for local runs.
 *
 * --------------------------------------------------------------------------
 * REQUIRED INPUT FILES
 * --------------------------------------------------------------------------
 * The verifier expects the following binary files in the current directory:
 *
 *   F_IMGS: "mnist_images_u8.bin"
 *     - 10000 MNIST test images, each 28x28 = 784 bytes.
 *     - Layout: imgs[10000][784], uint8_t, row-major order.
 *
 *   F_LBLS: "mnist_labels_u8.bin"
 *     - 10000 MNIST labels, uint8_t each, values 0..9.
 *
 *   F_W1:   "W1_p8.bin"
 *     - First-layer weights, 784 x 8 bytes, uint8_t, posit-8 encoding.
 *     - Layout: W1[i*8 + j] is weight from input pixel i to hidden neuron j.
 *
 *   F_b1:   "b1_p8.bin"
 *     - First-layer biases, 8 bytes, uint8_t, posit-8 encoding.
 *
 *   F_W2:   "W2_p8.bin"
 *     - Second-layer weights, 8 x 10 bytes, uint8_t, posit-8 encoding.
 *     - Layout: W2[j*10 + d] is weight from hidden neuron j to logit d.
 *
 *   F_b2:   "b2_p8.bin"
 *     - Second-layer biases, 10 bytes, uint8_t, posit-8 encoding.
 *
 * If any file is missing or too short, the verifier aborts with an I/O error
 * and exit code 3.
 *
 * --------------------------------------------------------------------------
 * RUNTIME USAGE
 * --------------------------------------------------------------------------
 * Always set unlimited stack before running:
 *
 *   ulimit -s unlimited
 *
 * Basic invocation:
 *
 *   ./posit_bab_verify_replay [OPTIONS...]
 *
 * Example (small ROI, top-ranked symbolic pixels):
 *
 *   ./posit_bab_verify_replay \
 *       --idx 10 --xrc 6-10,6-10 --topx 64 \
 *       --kmax 20 --bmax 48 \
 *       --widen 8.0 --depth 4096 \
 *       --verbose 2 \
 *       --idlelimit 20 \
 *       --rank-fast
 *
 * Example (global robustness heuristic, full image considered, top-36 pixels
 * symbolic, moderate budgets):
 *
 *   ./posit_bab_verify_replay \
 *       --idx 11 \
 *       --xrc 0-27,0-27 \
 *       --topx 36 \
 *       --kmax 36 \
 *       --bmax 288 \
 *       --widen 8.0 \
 *       --verbose 2 \
 *       --idlelimit 1000 \
 *       --timelimit 15000 \
 *       --rank-fast
 *
 * Example (full global robustness, all pixels symbolic, no topx trimming):
 *
 *   ./posit_bab_verify_replay \
 *       --idx 11 \
 *       --xrc 0-27,0-27 \
 *       --kmax 36 \
 *       --bmax 288 \
 *       --widen 8.0 \
 *       --verbose 2 \
 *       --idlelimit 1000 \
 *       --timelimit 15000 \
 *       --rank-fast
 *
 * --------------------------------------------------------------------------
 * EXIT CODES
 * --------------------------------------------------------------------------
 * The process-level exit code is intentionally simple:
 *
 *   0  - The verifier ran successfully, regardless of SAT/UNSAT/timeout.
 *   1  - Internal safety check failed (invariant violation).
 *   2  - Bad command line arguments (parse error or invalid value).
 *   3  - I/O error (missing files or short reads).
 *
 * The logical outcome of the run (SAT, UNSAT, TIMEOUT, etc.) is reported in
 * the final STATUS line on stdout, not through the exit code.
 *
 * --------------------------------------------------------------------------
 * FINAL STATUS LINE FORMAT
 * --------------------------------------------------------------------------
 * At the end of each run, the verifier prints a single STATUS line:
 *
 *   STATUS: <Outcome> | best_upper_margin=<val> | elapsed=<sec>s |
 *           pixels_changed=<K> | total_bit_flips=<B> |
 *           avg_hamming_per_pixel=<B_over_K> |
 *           patch_rows=r0-r1 | patch_cols=c0-c1 | note=<tag>
 *
 * where:
 *
 *   Outcome
 *     "Counterexample"
 *       - A misclassification witness was found within (kmax, bmax) budgets.
 *
 *     "No counterexample"
 *       - The search proved that no misclassification exists within budgets
 *         (either through full BaB search or root bound) or terminated
 *         through depth or node limits without finding a violation.
 *
 *     "TIMEOUT"
 *       - The run ended due to wall clock time limit (timelimit) or idle limit.
 *
 *   best_upper_margin
 *     - The best (largest) upper bound on the adversarial margin seen so far.
 *       The margin is defined as:
 *         margin = best_rival_logit - true_class_logit.
 *       An upper bound greater than 0 indicates that, in principle, an attack
 *       could exist; a non-positive bound can prove robustness in the root.
 *
 *   elapsed
 *     - Total wall clock time in seconds from start of main to STATUS.
 *
 *   pixels_changed (K)
 *     - Number of pixels that have changed from their original uint8_t value
 *       in the final SAT witness. This is K <= kmax. If UNSAT or no witness,
 *       this is 0.
 *
 *   total_bit_flips (B)
 *     - Total number of bit flips across all changed pixels in the final
 *       SAT witness. This is B <= bmax. It may be larger than K when multiple
 *       bits are flipped per pixel.
 *
 *   avg_hamming_per_pixel
 *     - For SAT runs: total_bit_flips / pixels_changed. For UNSAT: 0.0.
 *
 *   patch_rows / patch_cols
 *     - Bounding box of the "active patch" that contains all symbolic pixels
 *       actually considered in the BaB search:
 *         rows r0..r1, cols c0..c1.
 *       This is derived from the final symPix list after ranking and topx
 *       trimming. For global runs, patch_rows and patch_cols typically span
 *       the region given by --xrc (for example 0-27, 0-27).
 *
 *   note
 *     - A short tag describing how the run ended, for example:
 *         "witness_found_bab"
 *         "witness_found_greedy_byte"
 *         "witness_found_greedy_bit"
 *         "unsat_by_root_bound"
 *         "no_counterexample_within_budgets"
 *         "stopped_by_time_limit"
 *         "stopped_by_idle_limit"
 *         "stopped_by_node_limit"
 *         "stopped_by_depth_limit"
 *
 * --------------------------------------------------------------------------
 * LOGGING CONVENTIONS
 * --------------------------------------------------------------------------
 * Logging uses three verbosity levels:
 *
 *   --verbose 1  (default)
 *     - High level information and STATUS.
 *
 *   --verbose 2
 *     - Detailed BaB events, patch selection, and witness details.
 *
 *   --verbose 3
 *     - Trace-level logs, including bound updates and pruning decisions.
 *
 * All log lines are prefixed with:
 *
 *   [  t.ttts] message...
 *
 * where t.ttt is the elapsed time in seconds from g_run_t0.
 *
 * The verifier also prints "clean logits" and "post-perturb logits" in three
 * representations (hex, bits, double) for any SAT witness, along with a
 * detailed per-pixel change table.
 *
 * --------------------------------------------------------------------------
 * FORWARD PASS AND LOGITS
 * --------------------------------------------------------------------------
 * The function:
 *
 *   forward_posit8_quire_logits(img, W1, b1, W2, b2, outL)
 *
 * implements the network forward pass:
 *
 *   - Input img: uint8_t[784], raw 0..255 image values.
 *   - Each byte is normalized to a zero-centered double in [-1,1] using
 *       (v - 127.5) / 127.5
 *     then quantized to posit-8 (P8_NORM table).
 *   - First layer: 784x8 weights (posit-8), quire-8 dot products, plus biases,
 *     followed by ReLU in posit-8.
 *   - Second layer: 8x10 weights (posit-8), quire-8 dot products, plus biases.
 *   - outL[10]: posit-8 logits.
 *
 * The argmax over posit-8 logits is computed with:
 *
 *   argmax10_pure(L)
 *
 * which compares logits using posit subtraction and sign checks.
 *
 * --------------------------------------------------------------------------
 * SYMBOLIC PIXELS AND REGIONS
 * --------------------------------------------------------------------------
 * The verifier maintains a set of candidate pixels "symPix" that may be
 * modified symbolically by the search. This set is constructed in three steps:
 *
 *   1. Region filter (Xspec) via --xrc and/or --roi-heur.
 *   2. Influence scoring (quick_gate_bounds or rank-fast).
 *   3. Optional top-k trimming via --topx.
 *
 * Step 1: Region specification via --xrc
 * --------------------------------------
 * Option:
 *
 *   --xrc R0-R1,C0-C1
 *     - Adds a rectangular region of pixels to the candidate list.
 *     - R0,R1,C0,C1 are integer coordinates, 0 <= R,C <= 27.
 *     - Pixels in rows R0..R1 and columns C0..C1 are included.
 *     - Values are clamped to [0,27].
 *
 *   --xrc R,C
 *     - Adds a single pixel at row R, column C to the candidate list.
 *
 * You may pass --xrc multiple times to combine regions and individual pixels.
 *
 * If no --xrc is given, the code falls back to "no region filter":
 * every pixel 0..783 is considered eligible at this stage.
 *
 * The internal representation uses an array of row ranges per row, but the
 * semantics for the user are simply:
 *
 *   in_X(i) is true if and only if pixel i is in at least one --xrc region.
 *
 * Step 2: Optional heuristic ROI block via --roi-heur
 * ---------------------------------------------------
 * Option:
 *
 *   --roi-heur HxW
 *   --roi-heur HxW (with lowercase x or uppercase X)
 *   --roi-heur "H W"
 *   --roi-heur H
 *
 *   - Enables an automatic heuristic ROI mode.
 *   - H is the block height in pixels (rows), W is the block width in pixels
 *     (columns). If only H is given, W = H.
 *   - 1 <= H,W <= 28.
 *
 * When roi-heur is enabled:
 *
 *   - The verifier computes a cheap influence score for each of the 784 pixels
 *     using pre_rank_pixels_fast_u8 (see rank-fast section).
 *   - It then searches all possible HxW windows on the 28x28 grid and
 *     selects the block with the maximum sum of pixel influence scores.
 *   - The selected HxW window is converted to a single --xrc region that
 *     overrides any earlier region specifications.
 *   - This yields a data-dependent ROI that focuses the BaB search on a
 *     high-influence patch around the digit.
 *
 * Step 3: Influence scoring and top-k trimming
 * --------------------------------------------
 * After determining the initial candidate set symPix_all from Xspec (or
 * from ROI selection), we assign an influence score to each candidate pixel.
 * There are two modes:
 *
 *   - Default mode (no --rank-fast):
 *       quick_gate_bounds(...) computes per-pixel k_gain[t] values,
 *       representing an optimistic effect on the adversarial margin when that
 *       pixel is allowed to change. These are used as influence scores.
 *
 *   - Fast ranking (--rank-fast):
 *       pre_rank_pixels_fast_u8(...) computes influence scores by probing
 *       each pixel at extreme values 0 and 255 and measuring how much the
 *       true-vs-rival margin can be worsened.
 *
 * The influence scores are then sorted in descending order, and we keep the
 * top "take" pixels:
 *
 *   take = (g_topx > 0 && g_topx < inflN) ? g_topx : inflN
 *
 * where inflN is the number of candidate pixels before trimming.
 *
 * Option:
 *
 *   --topx X
 *     - Sets g_topx = X.
 *     - If X > 0 and X < number_of_candidates, only the X most influential
 *       pixels are kept as symbolic.
 *     - If X == 0 or X >= number_of_candidates, all candidates are kept.
 *
 * After trimming, the symPix[] array of length "take" lists the global pixel
 * indices (0..783) that are symbolic for this run. The "active patch" printed
 * in the log is the bounding box of these symPix positions.
 *
 * Interpreting "global" vs "local" robustness:
 *   - If you set --xrc 0-27,0-27 and omit --topx, then all 784 pixels may be
 *     considered symbolic (global robustness).
 *   - If you add --topx 36, then all pixels are globally "considered" for
 *     ranking, but the BaB search only allows the top-36 to change. This is
 *     a heuristic global check with a symbol limit.
 *
 * --------------------------------------------------------------------------
 * BUDGET PARAMETERS: KMAX, BMAX, WIDEN, DEPTH
 * --------------------------------------------------------------------------
 * Option:
 *
 *   --kmax K
 *     - Maximum number of pixels that may change.
 *     - This is a constraint on the count of distinct pixels whose value
 *       differs from the original image.
 *     - Must be >= 0. Default value if not provided is compiled as g_kmax=2.
 *
 *   --bmax B
 *     - Maximum number of bit flips across all pixels.
 *     - For each pixel, the Hamming distance between old and new uint8_t
 *       value contributes to this total.
 *     - Must be >= 0. Default value if not provided is g_bmax=16.
 *
 *   --widen W
 *     - Bound widening factor applied to the optimistic residual bound in the
 *       branch-and-bound pruning criterion.
 *     - W >= 1.0. Values > 1.0 relax the bound to avoid marginal pruning
 *       under numerical uncertainty. Default is g_widen=2.0.
 *
 *   --depth D
 *     - Depth limit for the recursive BaB search.
 *     - A value of D > 0 sets a hard cap on recursion. If omitted or set to
 *       <= 0, the depth limit defaults to 8 * nSym, where nSym is the number
 *       of symbolic pixels.
 *
 * Pruning condition:
 *   At each node, the verifier computes:
 *     - cur_margin = best_rival_logit - true_class_logit,
 *     - rem        = optimistic bound on additional margin increase within
 *                    remaining budgets Kleft and Bleft,
 *     - upper      = cur_margin + rem.
 *
 *   If upper <= 0, the node is pruned as incapable of causing a class change.
 *
 * --------------------------------------------------------------------------
 * STOPPING CRITERIA: TIME, IDLE, NODE, DEPTH
 * --------------------------------------------------------------------------
 * OPTION: --timelimit T
 *   - Wall clock time limit in seconds for the entire run.
 *   - If T > 0 and elapsed_time > T, the BaB search stops and returns
 *     BABS_TIME, mapped to OUTCOME_TIMEOUT with note "stopped_by_time_limit".
 *
 * OPTION: --idlelimit L
 *   - Idle limit in seconds for "no bound improvement".
 *   - If L > 0 and (current_time - g_last_progress_t) > L, the BaB search
 *     stops and returns BABS_IDLE, mapped to OUTCOME_IDLE with note
 *     "stopped_by_idle_limit".
 *
 * OPTION: --idle-eps E
 *   - Minimum improvement required in best_upper_margin to count as "progress".
 *   - Default is g_idle_eps=1e-6. If the new upper bound stays within E of
 *     the previous best, it does not reset the idle timer.
 *
 * OPTION: --nodelimit N
 *   - Limit on the number of BaB nodes visited.
 *   - If N > 0 and g_nodes_seen exceeds N, search stops with BABS_NODE,
 *     mapped to OUTCOME_NODE with note "stopped_by_node_limit".
 *
 * DEPTH LIMIT (from --depth or default)
 *   - If the recursion depth exceeds the limit, search stops with BABS_DEPTH,
 *     mapped to OUTCOME_DEPTH with note "stopped_by_depth_limit".
 *
 * --------------------------------------------------------------------------
 * GREEDY WARM START OPTIONS
 * --------------------------------------------------------------------------
 * Greedy modes are disabled by default. They can be used to quickly find a
 * misclassification before exploring the full BaB tree. Both respect kmax and
 * bmax budgets.
 *
 * OPTION: --greedy
 *   - Enables both greedy-byte and greedy-bit modes.
 *
 * OPTION: --greedy-byte
 *   - Enables the greedy-byte warm start only.
 *   - This mode searches for a misclassification by choosing full-byte
 *     changes for each pixel, with a budget on K and B.
 *
 * OPTION: --greedy-bit
 *   - Enables the greedy-bit warm start only.
 *   - This mode performs bit-level greedy flips, guided by bit-level bounds.
 *
 * OPTION: --no-greedy
 *   - Disables all greedy modes (overrides --greedy).
 *
 * OPTION: --no-greedy-byte
 *   - Disables greedy-byte specifically.
 *
 * OPTION: --no-greedy-bit
 *   - Disables greedy-bit specifically.
 *
 * Greedy semantics:
 *   - If a greedy mode finds a witness:
 *       - It logs detailed pre/post logits and per-pixel changes,
 *       - Runs a replay forward pass to confirm the witness,
 *       - Sets g_last_K_used and g_last_B_used to the witness budgets,
 *       - Emits a final STATUS line with outcome "Counterexample" and note
 *         "witness_found_greedy_byte" or "witness_found_greedy_bit",
 *       - Exits immediately with exit code 0.
 *
 *   - If no greedy witness is found within budgets, the code prints a message
 *     and proceeds to full BaB search.
 *
 * --------------------------------------------------------------------------
 * LOGGING VERBOSITY AND PROGRESS
 * --------------------------------------------------------------------------
 * OPTION: --verbose V
 *   - V is an integer in [1,3].
 *   - V=1 (LOG_INFO): basic status, clean prediction, final STATUS line.
 *   - V=2 (LOG_DEBUG): BaB events, patch selection, witness logs.
 *   - V=3 (LOG_TRACE): bound updates, detailed pruning, more internal info.
 *
 * OPTION: --progress G
 *   - This option is parsed into g_progress_every but currently not used
 *     in the logging macros. It is reserved for periodic progress logging
 *     or heartbeat extensions.
 *
 * --------------------------------------------------------------------------
 * RANK-FAST OPTION
 * --------------------------------------------------------------------------
 * OPTION: --rank-fast
 *   - Enables pre_rank_pixels_fast_u8 for influence estimation.
 *   - For each candidate pixel:
 *       1) The pixel is set to 0 and the margin is evaluated.
 *       2) The pixel is set to 255 and the margin is evaluated.
 *       3) The best margin degradation over these two extremes is taken as
 *          the influence score.
 *   - This is usually cheaper than running a full quick_gate_bounds for
 *     ranking, especially for large candidate sets.
 *   - If not set, ranking is based on TMP_k_gain from quick_gate_bounds.
 *
 * --------------------------------------------------------------------------
 * ROOT BOUND CHECK
 * --------------------------------------------------------------------------
 * Before launching the full BaB search, the verifier performs a clean UNSAT
 * root check:
 *
 *   - Computes clean logits for the original image,
 *   - Computes the current margin,
 *   - Computes an optimistic residual bound for the root node,
 *   - If margin + rem <= 0, then no adversarial example can exist within
 *     budgets and the run concludes UNSAT immediately:
 *       - A log line "====== UNSAT by root bound ======" is printed,
 *       - STATUS reports "No counterexample" with note "unsat_by_root_bound".
 *
 * This is a cheap global robustness check under the same budgets, and in
 * many cases it stops the search early.
 *
 * --------------------------------------------------------------------------
 * BINARY SEARCH STRUCTURE
 * --------------------------------------------------------------------------
 * The BaB search maintains nodes of type BaBNode, which include:
 *
 *   - img_cur[784]: current perturbed image (uint8_t).
 *   - pixel_changed_flags[t]: per-symbolic-pixel flags that track whether
 *     symPix[t] has changed from its original value.
 *   - changed_bytes: K used so far.
 *   - changed_bits:  B used so far.
 *   - considered_bit[t][b]: marks whether bit b of symPix[t] has been
 *     explored in the current path (to avoid revisiting the same flip).
 *
 * The search:
 *   - Checks time, idle, and node limits at each call,
 *   - Evaluates the current image and checks for misclassification,
 *   - Computes the margin and residual bound,
 *   - Prunes the branch if margin + rem <= 0,
 *   - Selects the best (t,b) bit according to TMP_b_gain and spawns:
 *       - Left child: apply the flip (if budgets allow),
 *       - Right child: skip the flip and continue.
 *
 * SAT at any node produces a detailed witness log and replay, then a STATUS
 * line with "Counterexample" and an appropriate note.
 *
 * --------------------------------------------------------------------------
 * NUMERICAL SUPPORT TABLES
 * --------------------------------------------------------------------------
 * Internal tables are precomputed at startup for speed:
 *
 *   - P8_DBL[256]: maps posit8_t byte values to double using
 *       convertP8ToDouble; used in logging and bound calculations.
 *
 *   - P8_NORM[256]: maps uint8_t image bytes to normalized posit-8 values
 *       using (v - 127.5) / 127.5 followed by convertDoubleToP8; used in
 *       the forward pass.
 *
 * These are built by:
 *
 *   build_p8_double_table();
 *   build_norm_table_zero_centered();
 *
 * --------------------------------------------------------------------------
 * SUMMARY
 * --------------------------------------------------------------------------
 * This verifier provides a controlled and configurable branch-and-bound
 * framework for analyzing posit-8 MNIST robustness under pixel and bit flip
 * budgets. The most important user-facing controls are:
 *
 *   - --idx:     which MNIST test index to verify.
 *   - --xrc:     region of pixels to consider (or full image 0-27,0-27).
 *   - --roi-heur HxW: automatically select a high-influence ROI block.
 *   - --topx X: number of most-influential pixels to keep symbolic.
 *   - --kmax K: maximum pixels allowed to change.
 *   - --bmax B: maximum bit flips allowed across all pixels.
 *   - --widen W: widening factor on residual bounds.
 *   - --depth D: recursion depth limit.
 *   - --timelimit T, --idlelimit L, --nodelimit N: stopping criteria.
 *   - --greedy*, --no-greedy*: greedy warm-start options.
 *   - --verbose V: logging verbosity.
 *   - --rank-fast: faster influence ranking.
 *
 * For large sweeps, it is recommended to:
 *
 *   - Compile once with the specified gcc command and SoftPosit linkage.
 *   - Always run with "ulimit -s unlimited".
 *   - Use moderate budgets (for example kmax=36, bmax=288) and non-zero
 *     idle/timelimits to balance completeness and runtime.
 *   - Use --xrc 0-27,0-27 with --topx K to perform heuristic global checks
 *     where all pixels are considered for ranking but only top-K are symbolic.
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

/* NEW: tracking the active patch/ROI used for symPix selection */
static int g_patch_has = 0;
static int g_patch_r0  = 0;
static int g_patch_r1  = 27;
static int g_patch_c0  = 0;
static int g_patch_c1  = 27;

static inline double now_s(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC,&ts);
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
    double avg_ham = (g_last_K_used>0) ? ((double)g_last_B_used / (double)g_last_K_used) : 0.0;
    fprintf(stdout,
        "STATUS: %s | best_upper_margin=%.6f | elapsed=%.3fs | "
        "pixels_changed=%d | total_bit_flips=%d | avg_hamming_per_pixel=%.6f",
        s, g_best_upper_seen, elapsed,
        g_last_K_used, g_last_B_used, avg_ham);

    /* NEW: report the patch/ROI if we have one */
    if (g_patch_has){
        fprintf(stdout,
                " | patch_rows=%d-%d | patch_cols=%d-%d",
                g_patch_r0, g_patch_r1, g_patch_c0, g_patch_c1);
    }

    /* existing note handling */
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
#define LOGI(fmt,...) do{ if(g_verbosity>=LOG_INFO ){ \
    fprintf(stdout,"[%7.3fs] " fmt "\n", now_s()-g_run_t0, ##__VA_ARGS__); \
    fflush(stdout);} }while(0)
#define LOGD(fmt,...) do{ if(g_verbosity>=LOG_DEBUG){ \
    fprintf(stdout,"[%7.3fs] " fmt "\n", now_s()-g_run_t0, ##__VA_ARGS__); \
    fflush(stdout);} }while(0)
#define LOGT(fmt,...) do{ if(g_verbosity>=LOG_TRACE){ \
    fprintf(stdout,"[%7.3fs] " fmt "\n", now_s()-g_run_t0, ##__VA_ARGS__); \
    fflush(stdout);} }while(0)

/* ---------- Pretty-print helpers ---------- */
static inline void bits8_str(uint8_t v, char out[9]){
    for (int i=7;i>=0;i--) out[7-i] = ((v>>i)&1)?'1':'0';
    out[8]='\0';
}

/* ==== Optimization #1: cached posit->double table ==== */
static double P8_DBL[256];
static void build_p8_double_table(void){
    for (int v=0; v<256; ++v)
        P8_DBL[v] = convertP8ToDouble((posit8_t){ .v = (uint8_t)v });
}
static inline double p8_to_double_fast(posit8_t x){ return P8_DBL[x.v]; }
/* ==================================================== */

static void print_logits_all(const char* title, const posit8_t L[10]){
    printf("%s (hex):", title);
    for (int d=0; d<10; d++) printf("  0x%02x", L[d].v);
    puts("");
    printf("%s (bits):", title);
    for (int d=0; d<10; d++){
        char b[9]; bits8_str(L[d].v, b); printf(" %s", b);
    }
    puts("");
    printf("%s (double):", title);
    for (int d=0; d<10; d++) printf(" %+.6f", p8_to_double_fast(L[d]));
    puts("");
}

/* ---------- Forward declarations for replay ---------- */
static void forward_posit8_quire_logits(const uint8_t img[784],
                                        const uint8_t *W1,const uint8_t *b1,
                                        const uint8_t *W2,const uint8_t *b2,
                                        posit8_t outL[10]);
static int argmax10_pure(const posit8_t L[10]);

/* ---------- Stage-2 replay forward-pass logger ---------- */
static void replay_forward_and_log(const uint8_t img_u8[784],
                                   const uint8_t *W1,const uint8_t *b1,
                                   const uint8_t *W2,const uint8_t *b2,
                                   int ytrue, const char* note)
{
    puts("----- REPLAY FORWARD PASS (stage-2) -----");
    posit8_t Lr[10];
    forward_posit8_quire_logits(img_u8, W1,b1,W2,b2, Lr);
    print_logits_all("replay logits", Lr);
    int pred_r = argmax10_pure(Lr);
    LOGI("replay prediction: %d (true label %d)%s%s",
         pred_r, ytrue,
         (note&&*note)?" | note=":"", (note&&*note)?note:"");
    puts("----- END REPLAY -----");
}

/* ---- Pixel change pretty printer ---- */
static inline double byte_to_norm_double(uint8_t v){
    /* matches build_norm_table_zero_centered(): (v - 127.5)/127.5 */
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

/* ---------- Files ---------- */
#define F_IMGS "mnist_images_u8.bin"
#define F_LBLS "mnist_labels_u8.bin"
#define F_W1   "W1_p8.bin"
#define F_b1   "b1_p8.bin"
#define F_W2   "W2_p8.bin"
#define F_b2   "b2_p8.bin"

/* ---------- CLI / Globals ---------- */
static int g_idx=0, g_kmax=2, g_bmax=16, g_topx=0;
static double g_widen = 2.0;           /* bound widening factor (>=1.0) */
static int g_depth_limit_cli = -1;     /* -1 = auto (8 * nSym) */
static double g_time_limit_s = 0.0;    /* wall-clock cap (inconclusive) */
static long   g_node_limit   = 0;      /* node-count cap (inconclusive) */
static long   g_nodes_seen   = 0;
static double g_idle_limit_s = 0.0;    /* idle cap (no bound improvement) */
static double g_idle_eps     = 1e-6;   /* min improvement to count as progress */
static double g_last_progress_t = 0.0; /* wall time of last progress */
static int    g_greedy_en    = 0;      /* DEFAULT OFF unless --greedy */
static int    g_greedy_byte  = 0;
static int    g_greedy_bit   = 0;
static int    g_progress_every=0;
static int    g_rank_fast    = 0;      /* NEW: fast pre-ranking (off by default) */

/* NEW: heuristic ROI block selection globals */
static int    g_roi_auto = 0;
static int    g_roi_h = 0;
static int    g_roi_w = 0;

/* NEW: flag to disable root bound pre-check */
static int    g_no_root_bound = 0;


/* ---------- Region selection ---------- */
typedef struct { int r0,r1,c0,c1; } RRect;
static RRect Xspecs[4096]; static int Xn=0;
static void add_Xspec_rc(const char*s){
    int r0,r1,c0,c1;
    if (sscanf(s,"%d-%d,%d-%d",&r0,&r1,&c0,&c1)==4){
        if(r0<0) r0=0; if(r1>27) r1=27;
        if(c0<0) c0=0; if(c1>27) c1=27;
        if (r0>r1 || c0>c1) return;
        for(int r=r0;r<=r1;r++){
            int a=r*28+c0, b=r*28+c1;
            if (Xn<4096) Xspecs[Xn++]=(RRect){.r0=a,.r1=b,.c0=0,.c1=0};
        }
    } else {
        int r,c;
        if (sscanf(s,"%d,%d",&r,&c)==2){
            if (r>=0 && r<28 && c>=0 && c<28){
                int idx=r*28+c;
                if (Xn<4096) Xspecs[Xn++]=(RRect){.r0=idx,.r1=idx,.c0=0,.c1=0};
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
        if(p>=0&&p<784&&Xn<4096) Xspecs[Xn++]=(RRect){.r0=p,.r1=p,.c0=0,.c1=0};
    }
}

/* ---------- IO helpers ---------- */
static inline void must_read(const char*fn, void*buf, size_t sz){
    FILE*f=fopen(fn,"rb");
    if(!f) die_io(fn, "open failed");
    size_t n=fread(buf,1,sz,f);
    fclose(f);
    if(n!=sz){
        char msg[96];
        snprintf(msg,sizeof msg,"short read (%zu/%zu)", (size_t)n, sz);
        die_io(fn,msg);
    }
}

/* ---------- Posit8 + Quire8 forward ---------- */
static posit8_t P8_NORM[256];
static void build_norm_table_zero_centered(void){
    for(int v=0; v<256; v++){
        double xd=((double)v - 127.5)/127.5;
        P8_NORM[v]=convertDoubleToP8(xd);
    }
}
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
static inline double p8_to_double(posit8_t x){ return convertP8ToDouble(x); }

static void forward_posit8_quire_logits(const uint8_t img[784],
                                        const uint8_t *W1,const uint8_t *b1,
                                        const uint8_t *W2,const uint8_t *b2,
                                        posit8_t outL[10])
{
    posit8_t x[784];
    for(int i=0;i<784;i++) x[i]=P8_NORM[ img[i] ];

    posit8_t h[8];
    for(int j=0;j<8;j++){
        quire8_t q=q8_clr((quire8_t){0});
        for(int i=0;i<784;i++){
            posit8_t w={.v=W1[i*8+j]};
            q=q8_fdp_add(q,w,x[i]);
        }
        posit8_t sum=q8_to_p8(q);
        posit8_t acc=p8_add(sum,(posit8_t){.v=b1[j]});
        h[j]=p8_relu(acc);
    }
    for(int d=0; d<10; d++){
        quire8_t q=q8_clr((quire8_t){0});
        for(int j=0;j<8;j++){
            posit8_t w={.v=W2[j*10+d]};
            q=q8_fdp_add(q,w,h[j]);
        }
        posit8_t sum=q8_to_p8(q);
        outL[d]=p8_add(sum,(posit8_t){.v=b2[d]});
    }
}
static int argmax10_pure(const posit8_t L[10]){
    int best=0;
    for(int d=1; d<10; d++) if (p8_gt_pure(L[d],L[best])) best=d;
    return best;
}

/* ---------- Bounds (quick-gate style) ---------- */
typedef struct { double inc[10]; double dec[10]; } SwingPerClass;

static int cmp_double_desc(const void*a,const void*b){
    double x=*(double*)a, y=*(double*)b;
    return (x>y)?-1:(x<y)?1:0;
}
static double sum_top_k(double *arr, int n, int K){
    if (K<=0 || n<=0) return 0.0;
    qsort(arr,n,sizeof(double),cmp_double_desc);
    double s=0;
    int m=(K<n?K:n);
    for(int i=0;i<m;i++) if (arr[i]>0) s+=arr[i];
    return s;
}

/* ==== Optimization #2: reusable buffers for bounds ==== */
static SwingPerClass  TMP_pxSwing[2048];
static double         TMP_k_gain[2048];
static double         TMP_b_gain[2048][8];
static double         TMP_Lc_opt[10];
/* ===================================================== */

/* Compute node-local pixel swings and per-bit gains */
static void quick_gate_bounds(const uint8_t *img,
                              const uint8_t *W1,const uint8_t *b1,
                              const uint8_t *W2,const uint8_t *b2,
                              int ytrue,
                              const int *symPix,int nSym,
                              SwingPerClass *pxSwing,       /* out */
                              double *k_gain,               /* out */
                              double b_gain[][8],           /* out */
                              double *Lc_opt /* out: length 10 */)
{
    posit8_t Lc[10];
    forward_posit8_quire_logits(img,W1,b1,W2,b2,Lc);
    for(int d=0; d<10; d++) Lc_opt[d]=p8_to_double_fast(Lc[d]);

    for(int t=0;t<nSym;t++){
        int p_idx=symPix[t];
        double minv[10], maxv[10];
        for(int d=0; d<10; d++){ minv[d]=+1e300; maxv[d]=-1e300; }
        posit8_t Ltmp[10];

        /* reuse one buffer */
        uint8_t tmp[784];
        memcpy(tmp, img, 784);
        uint8_t orig = tmp[p_idx];

        for(int v=0; v<256; v++){
            tmp[p_idx]=(uint8_t)v;
            forward_posit8_quire_logits(tmp,W1,b1,W2,b2,Ltmp);
            for(int d=0; d<10; d++){
                double lv=p8_to_double_fast(Ltmp[d]);
                if(lv<minv[d]) minv[d]=lv;
                if(lv>maxv[d]) maxv[d]=lv;
            }
        }
        tmp[p_idx]=orig;

        for(int d=0; d<10; d++){
            double inc=maxv[d]-Lc_opt[d];
            double dec=Lc_opt[d]-minv[d];
            pxSwing[t].inc[d]=inc<0?0:inc;
            pxSwing[t].dec[d]=dec<0?0:dec;
        }

        /* single-bit gains (reuse tmp buffer) */
        double best_gain;
        for(int bit=0; bit<8; bit++){
            tmp[p_idx] = ((const uint8_t*)img)[p_idx] ^ (1u<<bit);
            forward_posit8_quire_logits(tmp,W1,b1,W2,b2,Ltmp);
            best_gain=0.0;
            for(int d=0; d<10; d++){
                if(d==ytrue) continue;
                double gain = (p8_to_double_fast(Ltmp[d]) - Lc_opt[d])
                            + (Lc_opt[ytrue] - p8_to_double_fast(Ltmp[ytrue]));
                if (gain>best_gain) best_gain=gain;
            }
            b_gain[t][bit] = (best_gain>0?best_gain:0.0);
        }
        tmp[p_idx]=orig;

        double kbest=0.0;
        for(int d=0; d<10; d++){
            if (d==ytrue) continue;
            double s=pxSwing[t].inc[d] + pxSwing[t].dec[ytrue];
            if (s>kbest) kbest=s;
        }
        k_gain[t]=kbest;
    }
}

/* ---------- FAST RANK helper (optional pre-ranking) ----------
 * Cheap influence probe: for each candidate pixel, try extreme values {0,255}
 * and measure how much the margin (best_rival - Ly) can be worsened.
 * Used only when --rank-fast is enabled.
 */
static void pre_rank_pixels_fast_u8(
    const uint8_t *img_u8,
    const uint8_t *W1,const uint8_t *b1,
    const uint8_t *W2,const uint8_t *b2,
    int ytrue,
    const int *cand, int nCand,
    double *out_infl)               /* length nCand */
{
    posit8_t Lc[10];
    forward_posit8_quire_logits(img_u8, W1,b1,W2,b2, Lc);

    double Ly = p8_to_double_fast(Lc[ytrue]);
    double br = -1e300;
    for (int d=0; d<10; ++d){
        if (d == ytrue) continue;
        double vd = p8_to_double_fast(Lc[d]);
        if (vd > br) br = vd;
    }

    for (int t=0; t<nCand; ++t){
        int p = cand[t];
        double best_gain = 0.0;

        uint8_t tmp[784];
        memcpy(tmp, img_u8, 784);
        uint8_t orig = tmp[p];

        /* probe extreme 0 */
        tmp[p] = 0;
        posit8_t L2[10];
        forward_posit8_quire_logits(tmp, W1,b1,W2,b2, L2);
        double Ly2 = p8_to_double_fast(L2[ytrue]);
        double br2 = -1e300;
        for (int d=0; d<10; ++d){
            if (d == ytrue) continue;
            double vd = p8_to_double_fast(L2[d]);
            if (vd > br2) br2 = vd;
        }
        double g0 = (br2 - br) + (Ly - Ly2);
        if (g0 > best_gain) best_gain = g0;

        /* probe extreme 255 */
        memcpy(tmp, img_u8, 784);
        tmp[p] = 255;
        forward_posit8_quire_logits(tmp, W1,b1,W2,b2, L2);
        Ly2 = p8_to_double_fast(L2[ytrue]);
        br2 = -1e300;
        for (int d=0; d<10; ++d){
            if (d == ytrue) continue;
            double vd = p8_to_double_fast(L2[d]);
            if (vd > br2) br2 = vd;
        }
        double g1 = (br2 - br) + (Ly - Ly2);
        if (g1 > best_gain) best_gain = g1;

        out_infl[t] = (best_gain > 0.0 ? best_gain : 0.0);
    }
}

/* NEW: heuristic block selection using influence scores */
static void choose_roi_block_heuristic(
    int h, int w,
    const uint8_t img[784],
    const uint8_t *W1,const uint8_t *b1,
    const uint8_t *W2,const uint8_t *b2,
    int ytrue,
    int *out_r0, int *out_c0, int *out_r1, int *out_c1)
{
    int cand[784];
    double infl[784];
    for (int i=0;i<784;i++) cand[i]=i;

    pre_rank_pixels_fast_u8(img, W1,b1,W2,b2, ytrue, cand, 784, infl);

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
    Kg *kg = malloc(sizeof(Kg)*nSym);
    int kgN=0;
    for (int t=0; t<nSym; t++){
        if (!pixel_changed_flags[t]) {
            double g = k_gain[t]; if (g<0) g=0;
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
    int *picked = calloc(nSym, sizeof(int));
    for (int i=0; i<takeK; i++){
        k_part += kg[i].g;
        picked[ kg[i].t ] = 1;
    }
    free(kg);

    int total_bits = nSym*8;
    double *bg = malloc(sizeof(double)*total_bits);
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

    free(bg); free(picked);

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

static int node_apply_bitflip(BaBNode *nd, int pix_global_idx, int sym_pos, int bit,
                              int kmax, int bmax, uint8_t orig_byte)
{
    uint8_t old = nd->img_cur[pix_global_idx];
    uint8_t newv = old ^ (1u<<bit);

    int additional_bit = ((old ^ newv)&(1u<<bit))? 1:0;
    int additional_byte = (!nd->pixel_changed_flags[sym_pos] && (newv!=orig_byte)) ? 1 : 0;

    if (nd->changed_bits + additional_bit > bmax) return 0;
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
static int greedy_byte_warm_start(const uint8_t *img_orig,
                                  const uint8_t *W1,const uint8_t *b1,
                                  const uint8_t *W2,const uint8_t *b2,
                                  int ytrue,
                                  const int* symPix, int nSym,
                                  int kmax, int bmax,
                                  int* outK, int* outB, int* outPred)
{
    uint8_t cur[784]; memcpy(cur, img_orig, 784);
    uint8_t byte_changed[2048]={0};
    int K=0, B=0;

    posit8_t L_clean0[10];
    forward_posit8_quire_logits(img_orig, W1,b1,W2,b2, L_clean0);

    for (;;){
        posit8_t L[10];
        forward_posit8_quire_logits(cur, W1,b1,W2,b2, L);
        int pred = argmax10_pure(L);
        if (pred != ytrue){
            LOGI("Greedy BYTE warm start: SAT (K=%d,B=%d).", K,B);
            print_logits_all("clean logits", L_clean0);
            print_logits_all("post-perturb logits", L);
            printf("changes: %d bytes, %d bit flips\n", K,B);
            if (K){
                print_change_header();
                for(int t=0;t<nSym;t++){
                    int p=symPix[t]; uint8_t old=img_orig[p], nw=cur[p];
                    if (old!=nw) print_change_row(t, p, old, nw);
                }
            }
            printf("new prediction: %d (true %d)\n", pred, ytrue);

            require_or_die(K<=kmax, "K_used exceeds kmax (greedy byte)");
            require_or_die(B<=bmax, "B_used exceeds bmax (greedy byte)");
            require_or_die(pred!=ytrue, "greedy pred_after == true label");

            if (outK) *outK=K; if (outB) *outB=B; if (outPred) *outPred=pred;

            replay_forward_and_log(cur, W1,b1,W2,b2, ytrue, "witness_found_greedy_byte");
            return 1;
        }
        if (K>=kmax && B>=bmax) return 0;

        double best_gain = 0.0; int best_t=-1; uint8_t best_v=0;
        for (int t=0; t<nSym; t++){
            int p = symPix[t];
            uint8_t old = cur[p];
            for (int v=0; v<256; v++){
                uint8_t nv = (uint8_t)v;
                if (nv == old) continue;
                int ham = __builtin_popcount((unsigned)(old ^ nv));
                int addK = byte_changed[t] ? 0 : 1;
                if (B + ham > bmax || K + addK > kmax) continue;

                uint8_t tmp[784]; memcpy(tmp, cur, 784); tmp[p]=nv;
                posit8_t L2[10]; forward_posit8_quire_logits(tmp, W1,b1,W2,b2, L2);

                double Ly = p8_to_double_fast(L[ytrue]), Ly2 = p8_to_double_fast(L2[ytrue]);
                double br = -1e300, br2=-1e300;
                for(int d=0; d<10; d++){
                    if(d==ytrue) continue;
                    double vd=p8_to_double_fast(L[d]);  if (vd>br)  br=vd;
                    double v2=p8_to_double_fast(L2[d]); if (v2>br2) br2=v2;
                }
                double gain = (br2 - br) + (Ly - Ly2);
                if (gain > best_gain){ best_gain=gain; best_t=t; best_v=nv; }
            }
        }

        if (best_t<0 || best_gain <= 0.0) return 0;

        int p = symPix[best_t]; uint8_t old = cur[p];
        int ham = __builtin_popcount((unsigned)(old ^ best_v));
        int addK = byte_changed[best_t] ? 0 : 1;
        cur[p]=best_v; B += ham; if (!byte_changed[best_t]){ byte_changed[best_t]=1; K++; }
    }
}

static int greedy_bit_warm_start(const uint8_t *img_orig,
                                 const uint8_t *W1,const uint8_t *b1,
                                 const uint8_t *W2,const uint8_t *b2,
                                 int ytrue,
                                 const int* symPix, int nSym,
                                 int kmax, int bmax,
                                 int* outK, int* outB, int* outPred)
{
    BaBNode nd; memcpy(nd.img_cur, img_orig, 784);
    memset(nd.pixel_changed_flags, 0, sizeof(nd.pixel_changed_flags));
    memset(nd.considered_bit, 0, sizeof(nd.considered_bit));
    nd.changed_bits = 0; nd.changed_bytes = 0;

    posit8_t L_clean0[10];
    forward_posit8_quire_logits(img_orig, W1,b1,W2,b2, L_clean0);

    for (;;){
        posit8_t L[10];
        forward_posit8_quire_logits(nd.img_cur, W1,b1,W2,b2, L);
        int pred = argmax10_pure(L);
        if (pred != ytrue){
            LOGI("Greedy BIT warm start: SAT (K=%d, B=%d).", nd.changed_bytes, nd.changed_bits);
            print_logits_all("clean logits", L_clean0);
            print_logits_all("post-perturb logits", L);
            printf("changes: %d bytes, %d bit flips\n", nd.changed_bytes, nd.changed_bits);
            if (nd.changed_bytes){
                print_change_header();
                for(int t=0;t<nSym;t++){
                    int p=symPix[t]; uint8_t old=img_orig[p], nw=nd.img_cur[p];
                    if (old!=nw) print_change_row(t, p, old, nw);
                }
            }
            printf("new prediction: %d (true %d)\n", pred, ytrue);

            require_or_die(nd.changed_bytes<=kmax, "K_used exceeds kmax (greedy bit)");
            require_or_die(nd.changed_bits <=bmax, "B_used exceeds bmax (greedy bit)");
            require_or_die(pred!=ytrue, "greedy pred_after == true label");

            if (outK) *outK=nd.changed_bytes;
            if (outB) *outB=nd.changed_bits;
            if (outPred) *outPred=pred;

            replay_forward_and_log(nd.img_cur, W1,b1,W2,b2, ytrue, "witness_found_greedy_bit");
            return 1;
        }
        if (nd.changed_bytes >= kmax && nd.changed_bits >= bmax) return 0;

        quick_gate_bounds(nd.img_cur, W1,b1,W2,b2, ytrue, symPix,nSym,
                          TMP_pxSwing, TMP_k_gain, TMP_b_gain, TMP_Lc_opt);

        double best=-1.0; int best_t=-1, best_b=-1;
        for(int t=0;t<nSym;t++) for(int b=0;b<8;b++){
            double g = TMP_b_gain[t][b];
            if (g>best){ best=g; best_t=t; best_b=b; }
        }

        if (best_t<0 || best <= 0.0) return 0;

        int p = symPix[best_t];
        if (!node_apply_bitflip(&nd, p, best_t, best_b, kmax, bmax, img_orig[p])) return 0;
    }
}

/* ---------- BaB search ---------- */
typedef enum {
    BABS_UNSAT = 0,  /* proof UNSAT within budgets */
    BABS_SAT   = 1,  /* witness found */
    BABS_TIME  = 2,  /* time limit hit */
    BABS_IDLE  = 3,  /* idle-limit stop */
    BABS_NODE  = 4,  /* node-limit stop */
    BABS_DEPTH = 5   /* depth-limit stop */
} BabResult;

static BabResult bab_search(BaBNode *nd,
                            const uint8_t *img_orig,
                            const uint8_t *W1,const uint8_t *b1,
                            const uint8_t *W2,const uint8_t *b2,
                            int ytrue,
                            const int* symPix, int nSym,
                            int kmax, int bmax,
                            int depth_limit, double widen)
{
    extern double g_last_progress_t;
    extern long   g_nodes_seen;
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

    posit8_t L[10];
    forward_posit8_quire_logits(nd->img_cur, W1,b1,W2,b2, L);
    int pred = argmax10_pure(L);
    if (pred != ytrue){
        LOGI("Counterexample found: BaB witness (K=%d, B=%d).",
             nd->changed_bytes, nd->changed_bits);

        require_or_die(nd->changed_bytes<=kmax, "K_used exceeds kmax (BaB)");
        require_or_die(nd->changed_bits <=bmax, "B_used exceeds bmax (BaB)");
        require_or_die(pred!=ytrue, "pred_after == true label (BaB)");

        g_last_K_used = nd->changed_bytes;
        g_last_B_used = nd->changed_bits;

        posit8_t Lc0[10];
        forward_posit8_quire_logits(img_orig,W1,b1,W2,b2,Lc0);
        print_logits_all("clean logits", Lc0);
        print_logits_all("post-perturb logits", L);

        printf("changes: %d bytes, %d bit flips\n",
               nd->changed_bytes, nd->changed_bits);
        if (nd->changed_bytes){
            print_change_header();
            for(int t=0;t<nSym;t++){
                int p=symPix[t]; uint8_t old=img_orig[p], nw=nd->img_cur[p];
                if (old!=nw) print_change_row(t, p, old, nw);
            }
        }
        printf("new prediction: %d (true %d)\n", pred, ytrue);

        replay_forward_and_log(nd->img_cur, W1,b1,W2,b2, ytrue, "witness_found_bab");
        return BABS_SAT;
    }

    double Ly = p8_to_double_fast(L[ytrue]);
    double best_rival = -1e300;
    for(int d=0; d<10; d++){
        if(d==ytrue) continue;
        double vd=p8_to_double_fast(L[d]);
        if (vd>best_rival) best_rival=vd;
    }
    double cur_margin = best_rival - Ly;

    quick_gate_bounds(nd->img_cur, W1,b1,W2,b2, ytrue, symPix,nSym,
                      TMP_pxSwing, TMP_k_gain, TMP_b_gain, TMP_Lc_opt);

    double rem = optimistic_residual_bound_disjoint(
        nd->pixel_changed_flags, nd->changed_bytes, nd->changed_bits,
        symPix, nSym, TMP_k_gain, TMP_b_gain, kmax, bmax, widen);

    double upper = cur_margin + rem;
    if (upper > g_best_upper_seen + g_idle_eps){
        g_best_upper_seen = upper;
        g_last_progress_t = now_s();
        LOGT("progress: best_upper=%.6f (margin=%.6f, rem=%.6f)",
             g_best_upper_seen, cur_margin, rem);
    }

    if (cur_margin + rem <= 0){
        LOGD("PRUNE: margin=%.6f, optimistic_rem(widened)=%.6f → cannot exceed 0.",
             cur_margin, rem);
        return BABS_UNSAT;
    }

    if (depth_limit<=0){
        LOGD("STOP: depth limit.");
        return BABS_DEPTH;
    }

    double best=-1.0; int best_t=-1, best_b=-1;
    for(int tt=0; tt<nSym; tt++){
        for(int b=0; b<8; b++){
            if (nd->considered_bit[tt][b]) continue;
            double g = TMP_b_gain[tt][b];
            if (g>best){ best=g; best_t=tt; best_b=b; }
        }
    }
    if (best_t<0){
        return BABS_UNSAT;
    }
    nd->considered_bit[best_t][best_b]=1;

    BaBNode left = *nd;
    int p = symPix[best_t];
    if (node_apply_bitflip(&left, p, best_t, best_b, kmax, bmax, img_orig[p])){
        BabResult r = bab_search(&left, img_orig, W1,b1,W2,b2,
                                 ytrue, symPix,nSym, kmax,bmax, depth_limit-1, widen);
        if (r!=BABS_UNSAT){ return r; }
    }

    BaBNode right = *nd;
    BabResult rr = bab_search(&right, img_orig, W1,b1,W2,b2,
                              ytrue, symPix,nSym, kmax,bmax, depth_limit-1, widen);
    return rr;
}

/* ---------- Influence sort for topX ---------- */
typedef struct { int pix; double infl; } PixInfl;
static int cmp_infl_desc(const void*a,const void*b){
    double x=((const PixInfl*)a)->infl, y=((const PixInfl*)b)->infl;
    return (x>y)?-1:(x<y)?1:0;
}

/* ---------- Main ---------- */
int main(int argc, char**argv){
    g_run_t0 = now_s();
    setvbuf(stdout, NULL, _IONBF, 0); /* unbuffered stdout */
    g_last_progress_t = g_run_t0;
    g_best_upper_seen = -1e300;
    g_last_K_used = 0;
    g_last_B_used = 0;

        static struct option Lopt[] = {
        {"idx",required_argument,0,'i'},
        {"kmax",required_argument,0,'k'},
        {"bmax",required_argument,0,'b'},
        {"xrc",required_argument,0,'y'},
        {"topx",required_argument,0,'p'},
        {"widen",required_argument,0,'w'},
        {"depth",required_argument,0,'d'},
        {"timelimit",required_argument,0,'T'},
        {"nodelimit",required_argument,0,'N'},
        {"idlelimit",required_argument,0,2000},
        {"idle-eps",required_argument,0,2001},
        {"greedy",no_argument,0,1000},
        {"greedy-byte",no_argument,0,1001},
        {"greedy-bit", no_argument,0,1002},
        {"no-greedy",no_argument,0,1100},
        {"no-greedy-byte",no_argument,0,1101},
        {"no-greedy-bit", no_argument,0,1102},
        {"verbose",required_argument,0,'v'},
        {"progress",required_argument,0,'g'},
        {"rank-fast",no_argument,0,2100},   /* NEW: fast pre-ranking toggle */
        {"roi-heur",required_argument,0,2200}, /* NEW: heuristic ROI block */
        {"no-root-bound",no_argument,0,2300},  /* NEW: disable root UNSAT */
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
            /* Accept "HxW", "H W", or single "H" (square) */
            if (sscanf(optarg,"%dx%d",&h,&w)==2 ||
                sscanf(optarg,"%dX%d",&h,&w)==2){
                /* parsed as HxW */
            } else if (sscanf(optarg,"%d %d",&h,&w)==2){
                /* parsed as "H W" */
            } else if (sscanf(optarg,"%d",&h)==1){
                w = h;
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

    static uint8_t W1[784*8], b1[8], W2[8*10], b2[10];
    must_read(F_W1, W1, sizeof W1);
    must_read(F_b1, b1, sizeof b1);
    must_read(F_W2, W2, sizeof W2);
    must_read(F_b2, b2, sizeof b2);

    build_norm_table_zero_centered();
    build_p8_double_table();

    /* Clean logits (pretty) */
    posit8_t Lc_p[10];
    forward_posit8_quire_logits(imgs[g_idx], W1,b1,W2,b2, Lc_p);
    print_logits_all("clean logits", Lc_p);

    int true_label = lbl[g_idx];
    int pred_clean = argmax10_pure(Lc_p);
    int yref = pred_clean;
    LOGI("clean prediction: %d (true label %d)", pred_clean, true_label);

    /* If heuristic ROI requested, choose best HxW block and override Xspecs */
    if (g_roi_auto){
        int r0,c0,r1,c1;
        choose_roi_block_heuristic(g_roi_h, g_roi_w,
                                   imgs[g_idx], W1,b1,W2,b2,
                                   yref,
                                   &r0,&c0,&r1,&c1);
        Xn = 0; /* override any previous xrc */
        char buf[64];
        snprintf(buf,sizeof buf,"%d-%d,%d-%d", r0,r1,c0,c1);
        add_Xspec_rc(buf);
        LOGI("heuristic ROI: %dx%d block at rows %d-%d, cols %d-%d",
             g_roi_h,g_roi_w,r0,r1,c0,c1);
    }

    /* Build selected pixel list from Xspec (fallback to all if none) */
    int symPix_all[2048]; int nSym_all=0;
    for(int i=0;i<784;i++) if (in_X(i)) symPix_all[nSym_all++]=i;
    if (nSym_all==0){
        for(int i=0;i<784;i++) symPix_all[nSym_all++]=i;
    }

    /* Rank by influence and optionally trim to --topx
     * - Default: old behavior via quick_gate_bounds (k_gain ranking)
     * - With --rank-fast: use cheap 0/255 probes instead
     */
    typedef struct { int pix; double infl; } PixInfl2;
    PixInfl2 infl[2048]; int inflN=0;

    if (g_rank_fast){
        double *score = (double*)malloc(sizeof(double)*nSym_all);
        pre_rank_pixels_fast_u8(imgs[g_idx],
                                W1,b1,W2,b2,
                                yref,
                                symPix_all, nSym_all,
                                score);
        for (int t=0; t<nSym_all; ++t){
            infl[inflN++] = (PixInfl2){ .pix = symPix_all[t], .infl = score[t] };
        }
        free(score);
    } else {
        quick_gate_bounds(imgs[g_idx], W1,b1,W2,b2, yref,
                          symPix_all,nSym_all,
                          TMP_pxSwing, TMP_k_gain, TMP_b_gain, TMP_Lc_opt);
        for(int t=0;t<nSym_all;t++)
            infl[inflN++] = (PixInfl2){ .pix=symPix_all[t], .infl=TMP_k_gain[t] };
    }

    for (int i=0;i<inflN;i++)
        for(int j=i+1;j<inflN;j++)
            if (infl[j].infl>infl[i].infl){
                PixInfl2 tmp=infl[i]; infl[i]=infl[j]; infl[j]=tmp;
            }
    int take = (g_topx>0 && g_topx<inflN)? g_topx : inflN;
    int *symPix = (int*)malloc(sizeof(int)*take);
    for(int i=0;i<take;i++) symPix[i]=infl[i].pix;

    LOGI("selection: nSym=%d (from %d), topx=%d, rank_fast=%s",
         take, nSym_all, g_topx,
         g_rank_fast ? "on" : "off");

    /* NEW: derive a patch/ROI bounding box from the selected pixels */
    if (take > 0){
        int min_r = 27, max_r = 0;
        int min_c = 27, max_c = 0;
        for (int i=0; i<take; i++){
            int p   = symPix[i];
            int r   = p / 28;
            int c   = p % 28;
            if (r < min_r) min_r = r;
            if (r > max_r) max_r = r;
            if (c < min_c) min_c = c;
            if (c > max_c) max_c = c;
        }
        g_patch_has = 1;
        g_patch_r0  = min_r;
        g_patch_r1  = max_r;
        g_patch_c0  = min_c;
        g_patch_c1  = max_c;

        LOGI("active patch: rows %d-%d, cols %d-%d (derived from symPix)",
             g_patch_r0, g_patch_r1, g_patch_c0, g_patch_c1);
    }

    reset_Xspec_to_list(symPix, take);


    /* Warm starts (explicitly enabled only) */
    int gK=0,gB=0,gPred=-1;
    if (g_greedy_en || g_greedy_byte){
        if (greedy_byte_warm_start(imgs[g_idx], W1,b1,W2,b2,
                                   yref, symPix, take, g_kmax, g_bmax,
                                   &gK,&gB,&gPred)){
            g_last_K_used = gK; g_last_B_used = gB;
            status_and_exit(OUTCOME_SAT, "witness_found_greedy_byte");
        }
    }
    if (g_greedy_en || g_greedy_bit){
        if (greedy_bit_warm_start(imgs[g_idx], W1,b1,W2,b2,
                                  yref, symPix, take, g_kmax, g_bmax,
                                  &gK,&gB,&gPred)){
            g_last_K_used = gK; g_last_B_used = gB;
            status_and_exit(OUTCOME_SAT, "witness_found_greedy_bit");
        }
    }

    LOGI("Proceeding to BaB (greedy %s).",
         (g_greedy_en||g_greedy_byte||g_greedy_bit)? "enabled" : "disabled");

    /* ----- CLEAN UNSAT: root pre-check ----- */
        /* ----- CLEAN UNSAT: root pre-check (optional) ----- */
    BaBNode root;
    memcpy(root.img_cur, imgs[g_idx], 784);
    memset(root.pixel_changed_flags, 0, sizeof(root.pixel_changed_flags));
    memset(root.considered_bit, 0, sizeof(root.considered_bit));
    root.changed_bits = 0;
    root.changed_bytes = 0;

    if (!g_no_root_bound){
        posit8_t Lroot[10];
        forward_posit8_quire_logits(root.img_cur, W1,b1,W2,b2, Lroot);
        double Ly  = p8_to_double_fast(Lroot[yref]);
        double br  = -1e300;
        for(int d=0; d<10; d++) if (d!=yref) {
            double vd = p8_to_double_fast(Lroot[d]);
            if (vd>br) br=vd;
        }
        double cur_margin = br - Ly;

        quick_gate_bounds(root.img_cur, W1,b1,W2,b2, yref,
                          symPix,take,
                          TMP_pxSwing, TMP_k_gain, TMP_b_gain, TMP_Lc_opt);

        double rem = optimistic_residual_bound_disjoint(
            root.pixel_changed_flags, root.changed_bytes, root.changed_bits,
            symPix, take, TMP_k_gain, TMP_b_gain, g_kmax, g_bmax, g_widen);

        double upper0 = cur_margin + rem;
        if (upper0 > g_best_upper_seen + g_idle_eps){
            g_best_upper_seen = upper0;
            g_last_progress_t = now_s();
        }

        if (cur_margin + rem <= 0.0){
            LOGI("====== UNSAT by root bound ======");
            LOGI("margin=%.6f, optimistic_rem(widened)=%.6f (sum ≤ 0).",
                 cur_margin, rem);
            status_and_exit(OUTCOME_UNSAT, "unsat_by_root_bound");
        }
    } else {
        LOGI("Skipping root bound check (--no-root-bound).");
    }


    int depth_limit = (g_depth_limit_cli>0)? g_depth_limit_cli : (8*take);
    LOGI("BaB search starting (K≤%d, B≤%d, nSym=%d, depth_limit=%d, widen=%.2f)...",
         g_kmax, g_bmax, take, depth_limit, g_widen);

    BabResult br = bab_search(&root, imgs[g_idx],
                              W1,b1,W2,b2,
                              yref,
                              symPix,take,
                              g_kmax,g_bmax,
                              depth_limit, g_widen);

    switch (br){
        case BABS_SAT:
            status_and_exit(OUTCOME_SAT, "witness_found_bab");
            break;

        case BABS_UNSAT:
            LOGI("No counterexample within budgets (BaB): no misclassification found.");
            status_and_exit(OUTCOME_UNSAT, "no_counterexample_within_budgets");
            break;

        case BABS_TIME:
            LOGI("Stopped without proof (time limit).");
            status_and_exit(OUTCOME_TIMEOUT, "stopped_by_time_limit");
            break;

        case BABS_IDLE:
            LOGI("Stopped without proof (idle limit).");
            status_and_exit(OUTCOME_IDLE, "stopped_by_idle_limit");
            break;

        case BABS_NODE:
            LOGI("Stopped without proof (node limit).");
            status_and_exit(OUTCOME_NODE, "stopped_by_node_limit");
            break;

        case BABS_DEPTH:
            LOGI("Stopped without proof (depth limit).");
            status_and_exit(OUTCOME_DEPTH, "stopped_by_depth_limit");
            break;

        default:
            LOGI("Stopped for unknown reason.");
            status_and_exit(OUTCOME_NODE, "unknown_bab_result");
            break;
    }
}
