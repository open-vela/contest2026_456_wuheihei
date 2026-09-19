#ifndef KEYWORD_DETECT_H
#define KEYWORD_DETECT_H

#include <stdint.h>

/*
 * Keyword Detection — MFCC + DTW (C implementation)
 *
 * Recognizes custom voice keywords locally on embedded device.
 * No external dependencies — pure C with math.h only.
 *
 * Keywords: "救命", "帮助", "报警"
 *
 * Algorithm:
 *   1. Extract MFCC features (13 coeffs per frame)
 *   2. Match against pre-computed templates using DTW
 *   3. Return keyword with lowest DTW distance
 *
 * Usage:
 *   int kw_id = keyword_detect(pcm, num_samples, &confidence);
 *   if (kw_id >= 0) {
 *       printf("Keyword: %s (%.1f%%)\n", keyword_name(kw_id), confidence*100);
 *   }
 */

#define KW_SAMPLE_RATE   16000
#define KW_FRAME_LEN     400
#define KW_HOP_LEN       160
#define KW_NFFT          512
#define KW_NUM_MFCC      13
#define KW_NUM_FILTERS   26
#define KW_NUM_KEYWORDS  3
#define KW_MFCC_STATS_DIM (KW_NUM_MFCC * 4)

/* Maximum number of frames in a 2-second audio */
#define KW_MAX_FRAMES    200

/* DTW threshold: distances below this are accepted as a match */
#define KW_DTW_THRESHOLD 50.0f

/* Minimum RMS energy to consider as speech */
#define KW_MIN_ENERGY    0.01f

/* Detect keyword from PCM audio.
 *
 * Args:
 *   pcm:        int16 mono audio samples (16kHz)
 *   num_samples: number of samples
 *   confidence: output confidence [0.0, 1.0]
 *
 * Returns:
 *   Keyword index (0..KW_NUM_KEYWORDS-1), or -1 if no match.
 */
int keyword_detect(const int16_t *pcm, int num_samples, float *confidence);

/* Extract pooled MFCC statistics for the audio-event classifier.
 * Layout: mean[13], std[13], delta_mean[13], delta_std[13]. */
int keyword_extract_mfcc_stats(const int16_t *pcm, int num_samples,
                               float stats[KW_MFCC_STATS_DIM]);

/* Get keyword name by index.
 * Returns "none" if index is invalid. */
const char *keyword_name(int index);

/* Get keyword pinyin by index. */
const char *keyword_pinyin(int index);

#endif /* KEYWORD_DETECT_H */
