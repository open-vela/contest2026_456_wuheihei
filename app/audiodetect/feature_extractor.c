#include "feature_extractor.h"
#include "keyword_detect.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define FRAME_LEN 400
#define HOP_LEN 160
#define FRAME_METRICS 5
#define STATS_PER_METRIC 8
#define MAX_FRAMES 128

static int cmp_float(const void *a, const void *b)
{
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

static float percentile(const float *values, int n, float pct)
{
    float tmp[MAX_FRAMES];
    float pos;
    int lo;
    int hi;
    float frac;

    if (n <= 0) {
        return 0.0f;
    }
    memcpy(tmp, values, (size_t)n * sizeof(float));
    qsort(tmp, (size_t)n, sizeof(float), cmp_float);
    pos = (pct / 100.0f) * (float)(n - 1);
    lo = (int)floorf(pos);
    hi = (int)ceilf(pos);
    frac = pos - (float)lo;
    return tmp[lo] * (1.0f - frac) + tmp[hi] * frac;
}

static void write_stats(const float *values, int n, float *out)
{
    float sum = 0.0f;
    float sum_sq = 0.0f;
    float max_v = 0.0f;
    float min_v = 0.0f;
    float first_sum = 0.0f;
    float second_sum = 0.0f;
    int half;
    int i;

    if (n <= 0) {
        for (i = 0; i < STATS_PER_METRIC; ++i) {
            out[i] = 0.0f;
        }
        return;
    }

    max_v = values[0];
    min_v = values[0];
    for (i = 0; i < n; ++i) {
        float v = values[i];
        sum += v;
        sum_sq += v * v;
        if (v > max_v) {
            max_v = v;
        }
        if (v < min_v) {
            min_v = v;
        }
    }

    half = n / 2;
    if (half == 0) {
        first_sum = sum;
        second_sum = sum;
        half = n;
    } else {
        for (i = 0; i < half; ++i) {
            first_sum += values[i];
        }
        for (i = half; i < n; ++i) {
            second_sum += values[i];
        }
    }

    out[0] = sum / (float)n;
    out[1] = sqrtf(fmaxf(0.0f, sum_sq / (float)n - out[0] * out[0]));
    out[2] = max_v;
    out[3] = min_v;
    out[4] = percentile(values, n, 25.0f);
    out[5] = percentile(values, n, 50.0f);
    out[6] = percentile(values, n, 75.0f);
    out[7] = second_sum / (float)(n - n / 2) - first_sum / (float)(n / 2 == 0 ? n : n / 2);
}

void extract_audio_features(const int16_t *pcm, int num_samples, float features[AUDIO_FEATURE_DIM])
{
    float metrics[FRAME_METRICS][MAX_FRAMES];
    float prev_energy = 0.0f;
    int frame_count = 0;
    int start;
    int metric;

    memset(features, 0, AUDIO_FEATURE_DIM * sizeof(float));
    if (pcm == NULL || num_samples <= 0) {
        return;
    }

    for (start = 0; start + FRAME_LEN <= num_samples && frame_count < MAX_FRAMES; start += HOP_LEN) {
        float energy = 0.0f;
        float mean_abs = 0.0f;
        float max_abs = 0.0f;
        int zc = 0;
        int i;

        for (i = 0; i < FRAME_LEN; ++i) {
            float sample = (float)pcm[start + i] / 32768.0f;
            float abs_sample = fabsf(sample);
            energy += sample * sample;
            mean_abs += abs_sample;
            if (abs_sample > max_abs) {
                max_abs = abs_sample;
            }
            if (i > 0) {
                int prev_neg = pcm[start + i - 1] < 0;
                int curr_neg = pcm[start + i] < 0;
                if (prev_neg != curr_neg) {
                    zc++;
                }
            }
        }

        energy /= (float)FRAME_LEN;
        mean_abs /= (float)FRAME_LEN;
        metrics[0][frame_count] = energy;
        metrics[1][frame_count] = (float)zc / (float)(FRAME_LEN - 1);
        metrics[2][frame_count] = mean_abs;
        metrics[3][frame_count] = max_abs;
        metrics[4][frame_count] = frame_count == 0 ? 0.0f : fabsf(energy - prev_energy);
        prev_energy = energy;
        frame_count++;
    }

    for (metric = 0; metric < FRAME_METRICS; ++metric) {
        write_stats(metrics[metric], frame_count, &features[metric * STATS_PER_METRIC]);
    }

    /* Reuse the app's existing pure-C MFCC front-end. The appended layout
     * matches extract_features_mfcc.py exactly. */
    (void)keyword_extract_mfcc_stats(pcm, num_samples,
                                     &features[AUDIO_TIME_FEATURE_DIM]);
}
