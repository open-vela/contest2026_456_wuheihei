/*
 * keyword_detect.c — MFCC + DTW keyword recognition (embedded C)
 *
 * Pure C implementation with no external dependencies beyond math.h.
 * Matches the Python keyword_detect.py algorithm:
 *   1. Pre-emphasis filter
 *   2. Framing + Hamming window
 *   3. FFT power spectrum
 *   4. Mel filter bank → log → DCT = MFCC
 *   5. DTW matching against templates
 *
 * Memory usage:
 *   - Mel filter bank: 26 × 257 floats = 26.7 KB (static)
 *   - DCT matrix:      13 × 26 floats = 1.4 KB (static)
 *   - Hamming window:  400 floats = 1.6 KB (static)
 *   - FFT workspace:   512 × 2 floats = 4 KB (stack)
 *   - DTW cost matrix: 200 × 200 floats = 160 KB (static)
 *   - Templates:       ~53 KB (from keyword_templates.h)
 *   Total: ~247 KB (fits in OpenVela RAM)
 */

#include "keyword_detect.h"
#include "keyword_templates.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- Static tables (pre-computed) ---- */

/* Mel filter bank: NUM_FILTERS × (NFFT/2+1) */
static float s_mel_fb[KW_NUM_FILTERS][KW_NFFT / 2 + 1];
static int s_mel_fb_init = 0;

/* DCT-II matrix: NUM_MFCC × NUM_FILTERS */
static float s_dct_mat[KW_NUM_MFCC][KW_NUM_FILTERS];

/* Hamming window */
static float s_hamming[KW_FRAME_LEN];

/* DTW cost matrix (static to avoid stack overflow) */
static float s_dtw_cost[KW_MAX_FRAMES + 1][KW_MAX_FRAMES + 1];

/* Shared MFCC workspace for keyword matching and event feature pooling. */
static float s_mfcc_workspace[KW_MAX_FRAMES][KW_NUM_MFCC];

/* Keyword names */
static const char *const s_kw_names[KW_NUM_KEYWORDS] = {
    "\xE6\x95\x91\xE5\x91\xBD",  /* 救命 (UTF-8) */
    "\xE5\xB8\xAE\xE5\x8A\xA9",  /* 帮助 (UTF-8) */
    "\xE6\x8A\xA5\xE8\xAD\xA6",  /* 报警 (UTF-8) */
};

static const char *const s_kw_pinyin[KW_NUM_KEYWORDS] = {
    "jiu_ming",
    "bang_zhu",
    "bao_jing",
};

/* ---- Helper: Hz <-> Mel conversion ---- */

static float hz_to_mel(float hz)
{
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel)
{
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

/* ---- Initialize static tables (called once) ---- */

static void init_tables(void)
{
    if (s_mel_fb_init) return;

    /* Build Mel filter bank */
    float mel_min = hz_to_mel(0.0f);
    float mel_max = hz_to_mel((float)KW_SAMPLE_RATE / 2.0f);

    float mel_points[KW_NUM_FILTERS + 2];
    float hz_points[KW_NUM_FILTERS + 2];
    int bin_points[KW_NUM_FILTERS + 2];

    for (int i = 0; i < KW_NUM_FILTERS + 2; i++) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (KW_NUM_FILTERS + 1);
        hz_points[i] = mel_to_hz(mel_points[i]);
        bin_points[i] = (int)floorf((KW_NFFT + 1) * hz_points[i] / KW_SAMPLE_RATE);
    }

    memset(s_mel_fb, 0, sizeof(s_mel_fb));
    for (int m = 1; m <= KW_NUM_FILTERS; m++) {
        int left = bin_points[m - 1];
        int center = bin_points[m];
        int right = bin_points[m + 1];
        for (int k = left; k < center; k++) {
            if (center > left)
                s_mel_fb[m - 1][k] = (float)(k - left) / (center - left);
        }
        for (int k = center; k < right; k++) {
            if (right > center)
                s_mel_fb[m - 1][k] = (float)(right - k) / (right - center);
        }
    }

    /* Build DCT-II matrix */
    for (int k = 0; k < KW_NUM_MFCC; k++) {
        for (int n = 0; n < KW_NUM_FILTERS; n++) {
            s_dct_mat[k][n] = cosf(M_PI * k * (2 * n + 1) / (2.0f * KW_NUM_FILTERS));
            s_dct_mat[k][n] *= sqrtf(2.0f / KW_NUM_FILTERS);
        }
    }
    /* Scale k=0 by sqrt(0.5) */
    for (int n = 0; n < KW_NUM_FILTERS; n++) {
        s_dct_mat[0][n] *= sqrtf(0.5f);
    }

    /* Build Hamming window */
    for (int i = 0; i < KW_FRAME_LEN; i++) {
        s_hamming[i] = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (KW_FRAME_LEN - 1));
    }

    s_mel_fb_init = 1;
}

/* ---- In-place radix-2 FFT (N must be power of 2) ---- */

static void fft_radix2(float *real, float *imag, int n)
{
    /* Bit reversal permutation */
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            float tr = real[i]; real[i] = real[j]; real[j] = tr;
            float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
        }
    }

    /* Cooley-Tukey */
    for (int len = 2; len <= n; len <<= 1) {
        float angle = -2.0f * M_PI / len;
        float wr = cosf(angle), wi = sinf(angle);
        for (int i = 0; i < n; i += len) {
            float wmr = 1.0f, wmi = 0.0f;
            for (int k = 0; k < len / 2; k++) {
                float tr = wmr * real[i + k + len/2] - wmi * imag[i + k + len/2];
                float ti = wmr * imag[i + k + len/2] + wmi * real[i + k + len/2];
                real[i + k + len/2] = real[i + k] - tr;
                imag[i + k + len/2] = imag[i + k] - ti;
                real[i + k] += tr;
                imag[i + k] += ti;
                float nwr = wmr * wr - wmi * wi;
                wmi = wmr * wi + wmi * wr;
                wmr = nwr;
            }
        }
    }
}

/* ---- MFCC extraction ---- */

static int extract_mfcc(const int16_t *pcm, int num_samples,
                        float mfcc_out[KW_MAX_FRAMES][KW_NUM_MFCC])
{
    init_tables();

    /* Pre-emphasis: y[n] = x[n] - 0.97 * x[n-1] */
    static float x[KW_SAMPLE_RATE * 2];  /* max 2 seconds */
    int n = (num_samples < KW_SAMPLE_RATE * 2) ? num_samples : KW_SAMPLE_RATE * 2;

    x[0] = (float)pcm[0] / 32768.0f;
    for (int i = 1; i < n; i++) {
        x[i] = (float)pcm[i] / 32768.0f - 0.97f * (float)pcm[i - 1] / 32768.0f;
    }

    /* Frame the signal */
    int num_frames = 1 + (n - KW_FRAME_LEN) / KW_HOP_LEN;
    if (num_frames > KW_MAX_FRAMES) num_frames = KW_MAX_FRAMES;
    if (num_frames < 1) num_frames = 1;

    /* FFT workspace */
    static float fft_real[KW_NFFT];
    static float fft_imag[KW_NFFT];
    static float power[KW_NFFT / 2 + 1];
    static float mel_energy[KW_NUM_FILTERS];
    static float log_mel[KW_NUM_FILTERS];

    for (int f = 0; f < num_frames; f++) {
        int start = f * KW_HOP_LEN;

        /* Copy frame + Hamming window */
        memset(fft_real, 0, sizeof(fft_real));
        memset(fft_imag, 0, sizeof(fft_imag));
        for (int i = 0; i < KW_FRAME_LEN; i++) {
            fft_real[i] = x[start + i] * s_hamming[i];
        }

        /* FFT */
        fft_radix2(fft_real, fft_imag, KW_NFFT);

        /* Power spectrum */
        for (int i = 0; i <= KW_NFFT / 2; i++) {
            power[i] = fft_real[i] * fft_real[i] + fft_imag[i] * fft_imag[i];
        }

        /* Mel filter bank */
        for (int m = 0; m < KW_NUM_FILTERS; m++) {
            float sum = 0.0f;
            for (int i = 0; i <= KW_NFFT / 2; i++) {
                sum += power[i] * s_mel_fb[m][i];
            }
            mel_energy[m] = (sum < 1e-10f) ? 1e-10f : sum;
            log_mel[m] = logf(mel_energy[m]);
        }

        /* DCT → MFCC */
        for (int k = 0; k < KW_NUM_MFCC; k++) {
            float sum = 0.0f;
            for (int m = 0; m < KW_NUM_FILTERS; m++) {
                sum += log_mel[m] * s_dct_mat[k][m];
            }
            mfcc_out[f][k] = sum;
        }
    }

    return num_frames;
}

/* ---- DTW distance ---- */

static float dtw_distance(const float *seq1, int t1,
                          const float *seq2, int t2, int dim)
{
    if (t1 <= 0 || t2 <= 0) return 1e9f;
    if (t1 > KW_MAX_FRAMES || t2 > KW_MAX_FRAMES) return 1e9f;

    int band = abs(t1 - t2);
    int max_t = (t1 > t2) ? t1 : t2;
    int bw = (int)(0.3f * max_t);
    if (bw > band) band = bw;

    /* Initialize cost matrix */
    for (int i = 0; i <= t1; i++) {
        for (int j = 0; j <= t2; j++) {
            s_dtw_cost[i][j] = 1e9f;
        }
    }
    s_dtw_cost[0][0] = 0.0f;

    for (int i = 1; i <= t1; i++) {
        int j_start = i - band; if (j_start < 1) j_start = 1;
        int j_end = i + band;   if (j_end > t2) j_end = t2;

        for (int j = j_start; j <= j_end; j++) {
            /* Euclidean distance between frames */
            float dist = 0.0f;
            for (int d = 0; d < dim; d++) {
                float diff = seq1[(i - 1) * dim + d] - seq2[(j - 1) * dim + d];
                dist += diff * diff;
            }
            dist = sqrtf(dist);

            /* DTW recurrence */
            float c1 = s_dtw_cost[i - 1][j];
            float c2 = s_dtw_cost[i][j - 1];
            float c3 = s_dtw_cost[i - 1][j - 1];
            float min_c = (c1 < c2) ? c1 : c2;
            if (c3 < min_c) min_c = c3;

            s_dtw_cost[i][j] = dist + min_c;
        }
    }

    return s_dtw_cost[t1][t2] / (float)((t1 > t2) ? t1 : t2);
}

/* ---- Public API ---- */

int keyword_extract_mfcc_stats(const int16_t *pcm, int num_samples,
                               float stats[KW_MFCC_STATS_DIM])
{
    int frames;

    if (pcm == NULL || stats == NULL || num_samples <= 0) {
        return -1;
    }
    memset(stats, 0, KW_MFCC_STATS_DIM * sizeof(float));
    frames = extract_mfcc(pcm, num_samples, s_mfcc_workspace);
    if (frames <= 0) {
        return -1;
    }

    for (int coefficient = 0; coefficient < KW_NUM_MFCC; coefficient++) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        float delta_sum = 0.0f;
        float delta_sum_sq = 0.0f;

        for (int frame = 0; frame < frames; frame++) {
            float value = s_mfcc_workspace[frame][coefficient];
            sum += value;
            sum_sq += value * value;
            if (frame > 0) {
                float delta = value - s_mfcc_workspace[frame - 1][coefficient];
                delta_sum += delta;
                delta_sum_sq += delta * delta;
            }
        }

        float mean = sum / (float)frames;
        stats[coefficient] = mean;
        stats[KW_NUM_MFCC + coefficient] =
            sqrtf(fmaxf(0.0f, sum_sq / (float)frames - mean * mean));

        if (frames > 1) {
            int delta_frames = frames - 1;
            float delta_mean = delta_sum / (float)delta_frames;
            stats[2 * KW_NUM_MFCC + coefficient] = delta_mean;
            stats[3 * KW_NUM_MFCC + coefficient] =
                sqrtf(fmaxf(0.0f, delta_sum_sq / (float)delta_frames -
                                      delta_mean * delta_mean));
        }
    }
    return 0;
}

const char *keyword_name(int index)
{
    if (index < 0 || index >= KW_NUM_KEYWORDS) return "none";
    return s_kw_names[index];
}

const char *keyword_pinyin(int index)
{
    if (index < 0 || index >= KW_NUM_KEYWORDS) return "none";
    return s_kw_pinyin[index];
}

int keyword_detect(const int16_t *pcm, int num_samples, float *confidence)
{
    init_tables();

    /* Energy check: reject silence/noise */
    float sum_sq = 0.0f;
    int n = (num_samples < KW_SAMPLE_RATE * 2) ? num_samples : KW_SAMPLE_RATE * 2;
    for (int i = 0; i < n; i++) {
        float v = (float)pcm[i] / 32768.0f;
        sum_sq += v * v;
    }
    float rms = sqrtf(sum_sq / n);
    if (rms < KW_MIN_ENERGY) {
        if (confidence) *confidence = 0.0f;
        return -1;
    }

    /* Extract MFCC features from input */
    int input_frames = extract_mfcc(pcm, num_samples, s_mfcc_workspace);

    /* Match against each keyword's templates */
    float best_dist = 1e9f;
    int best_kw = -1;

    for (int kw = 0; kw < KW_NUM_KEYWORDS; kw++) {
        float kw_min_dist = 1e9f;
        int num_templates = KW_TMPL_COUNTS[kw];

        for (int t = 0; t < num_templates; t++) {
            const float *tmpl = KW_ALL_TMPL[kw][t];
            int tmpl_frames = KW_ALL_FRAMES[kw][t];

            float dist = dtw_distance(
                (const float *)s_mfcc_workspace, input_frames,
                tmpl, tmpl_frames, KW_NUM_MFCC
            );

            if (dist < kw_min_dist) kw_min_dist = dist;
        }

        if (kw_min_dist < best_dist) {
            best_dist = kw_min_dist;
            best_kw = kw;
        }
    }

    /* Check threshold */
    if (best_dist < KW_DTW_THRESHOLD) {
        if (confidence) {
            *confidence = 1.0f - best_dist / KW_DTW_THRESHOLD;
            if (*confidence < 0.0f) *confidence = 0.0f;
        }
        return best_kw;
    }

    if (confidence) *confidence = 0.0f;
    return -1;
}
