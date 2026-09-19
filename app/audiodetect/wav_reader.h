#ifndef WAV_READER_H
#define WAV_READER_H

#include <stdint.h>

int read_wav_16k_mono(const char *path, int16_t *pcm, int max_samples);

#endif
