#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <stdint.h>

#define AUDIO_SAMPLE_RATE 16000
#define AUDIO_NUM_SAMPLES 16000
#define AUDIO_TIME_FEATURE_DIM 40
#define AUDIO_FEATURE_DIM 92

void extract_audio_features(const int16_t *pcm, int num_samples, float features[AUDIO_FEATURE_DIM]);

#endif
