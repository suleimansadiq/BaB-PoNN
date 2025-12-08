#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "softposit.h"

/* ---------- robust file read ---------- */
static void must_read(const char *fname, void *buf, size_t sz) {
    FILE *f = fopen(fname, "rb");
    if (!f) { perror(fname); exit(1); }
    size_t n = fread(buf, 1, sz, f);
    if (n != sz) {
        fprintf(stderr, "Short read on %s (got %zu, need %zu)\n", fname, n, sz);
        exit(1);
    }
    fclose(f);
}

/* ---------- posit helpers ---------- */
static inline int p8_is_NaR(posit8_t x) { return x.v == 0x80; }

/* pure-posit strict greater: a > b  (no doubles) */
static inline int p8_gt_pure(posit8_t a, posit8_t b) {
    posit8_t d = p8_sub(a, b);
    if (p8_is_NaR(d)) return 0;
    /* positive and non-zero => greater */
    return (d.v != 0x00) && ((d.v & 0x80u) == 0);
}

/* ReLU in posit domain */
static inline posit8_t p8_relu(posit8_t x) {
    const posit8_t z = (posit8_t){ .v = 0x00 };
    return p8_gt_pure(x, z) ? x : z;
}

/* ---------- input normalization LUT: x = (byte - 127.5)/127.5 ---------- */
static posit8_t P8_NORM[256];
static void build_norm_table_zero_centered(void) {
    for (int v = 0; v < 256; v++) {
        double xd = ((double)v - 127.5) / 127.5;
        P8_NORM[v] = convertDoubleToP8(xd);  /* constants only */
    }
}

/* ---------- quire-based fused dot product with bias ---------- */
/* acc = bias + sum_i (w[i] * x[i]) */
static inline posit8_t dot_p8_q8_bias(const uint8_t *w_codes,
                                      const posit8_t *x_vec,
                                      int len,
                                      posit8_t bias)
{
    quire8_t q;
    memset(&q, 0, sizeof q);                 /* clear quire struct */
    for (int i = 0; i < len; i++) {
        posit8_t w = (posit8_t){ .v = w_codes[i] };
        q = q8_fdp_add(q, w, x_vec[i]);      /* fused accumulate */
    }
    posit8_t sum = q8_to_p8(q);
    return p8_add(sum, bias);
}

int main(int argc, char **argv) {
    /* dataset */
    static uint8_t imgs[10000][784];
    static uint8_t lbl[10000];

    /* posit8 weights/biases (raw .bin) */
    static uint8_t W1[784*8], b1[8], W2[8*10], b2[10];

    must_read("mnist_images_u8.bin", imgs, 784u*10000u);
    must_read("mnist_labels_u8.bin", lbl, 10000u);
    must_read("W1_p8.bin", W1, sizeof W1);
    must_read("b1_p8.bin", b1, sizeof b1);
    must_read("W2_p8.bin", W2, sizeof W2);
    must_read("b2_p8.bin", b2, sizeof b2);

    build_norm_table_zero_centered();

    int total = (argc > 1) ? atoi(argv[1]) : 10000;
    if (total < 1) total = 1;
    if (total > 10000) total = 10000;

    /* set to 1 only if you want to *print* doubles for inspection */
    const int LOG_DOUBLES = 0;

    FILE *fout = fopen("check_pred.txt", "w");
    if (!fout) { perror("check_pred.txt"); exit(1); }

    int correct = 0;
    posit8_t x[784];

    for (int idx = 0; idx < total; idx++) {
        /* normalize bytes -> posit8 once */
        for (int i = 0; i < 784; i++) x[i] = P8_NORM[ imgs[idx][i] ];

        /* hidden layer (8) — note W1 layout: element (i,j) at W1[i*8 + j] */
        posit8_t h[8];
        for (int j = 0; j < 8; j++) {
            /* build a view of weights column j into a contiguous temp using stride 8 */
            /* but we can accumulate with stride directly: */
            quire8_t q;
            memset(&q, 0, sizeof q);
            for (int i = 0; i < 784; i++) {
                posit8_t w = (posit8_t){ .v = W1[i*8 + j] };
                q = q8_fdp_add(q, w, x[i]);
            }
            posit8_t sum = q8_to_p8(q);
            posit8_t acc = p8_add(sum, (posit8_t){ .v = b1[j] });
            h[j] = p8_relu(acc);
        }

        /* output logits (10) — W2 layout: element (j,d) at W2[j*10 + d] */
        posit8_t logits[10];
        for (int d = 0; d < 10; d++) {
            quire8_t q;
            memset(&q, 0, sizeof q);
            for (int j = 0; j < 8; j++) {
                posit8_t w = (posit8_t){ .v = W2[j*10 + d] };
                q = q8_fdp_add(q, w, h[j]);
            }
            posit8_t sum = q8_to_p8(q);
            logits[d] = p8_add(sum, (posit8_t){ .v = b2[d] });
        }

        /* argmax in pure posit */
        int best = 0;
        for (int d = 1; d < 10; d++) if (p8_gt_pure(logits[d], logits[best])) best = d;

        int ok = (best == lbl[idx]); if (ok) correct++;

        if (!LOG_DOUBLES) {
            printf("idx=%4d  label=%d  pred=%d  [%s]  logits_hex:",
                   idx, lbl[idx], best, ok ? "CORRECT" : "WRONG");
            for (int d = 0; d < 10; d++) printf(" 0x%02x", logits[d].v);
            printf("\n");
        } else {
            printf("idx=%4d  label=%d  pred=%d  [%s]\n   logits_hex:",
                   idx, lbl[idx], best, ok ? "CORRECT" : "WRONG");
            for (int d = 0; d < 10; d++) printf(" 0x%02x", logits[d].v);
            printf("\n   logits_dec:");
            for (int d = 0; d < 10; d++) printf(" %+8.3f", convertP8ToDouble(logits[d]));
            printf("\n");
        }

        fprintf(fout, "idx=%d,label=%d,pred=%d,%s\n",
                idx, lbl[idx], best, ok ? "CORRECT" : "WRONG");
    }

    double acc = (double)correct / total;
    printf("\nTested %d samples, Accuracy = %.4f\n", total, acc);
    fprintf(fout, "\nTested %d samples, Accuracy = %.4f\n", total, acc);
    fclose(fout);
    return 0;
}
