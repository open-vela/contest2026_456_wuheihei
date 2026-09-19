#include "wav_reader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint16_t read_u16_le(const unsigned char *p)
{
    return (uint16_t)(p[0] | (p[1] << 8));
}

static uint32_t read_u32_le(const unsigned char *p)
{
    return (uint32_t)(p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24));
}

int read_wav_16k_mono(const char *path, int16_t *pcm, int max_samples)
{
    FILE *fp;
    unsigned char header[12];
    int fmt_ok = 0;
    int data_found = 0;
    uint16_t audio_format = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    uint32_t data_size = 0;
    int samples_to_read;
    int i;

    if (pcm == NULL || max_samples <= 0) {
        fprintf(stderr, "invalid PCM buffer\n");
        return -1;
    }
    for (i = 0; i < max_samples; ++i) {
        pcm[i] = 0;
    }

    fp = fopen(path, "rb");
    if (fp == NULL) {
        fprintf(stderr, "failed to open WAV: %s\n", path);
        return -1;
    }
    if (fread(header, 1, sizeof(header), fp) != sizeof(header) ||
        memcmp(header, "RIFF", 4) != 0 || memcmp(header + 8, "WAVE", 4) != 0) {
        fprintf(stderr, "not a RIFF/WAVE file: %s\n", path);
        fclose(fp);
        return -1;
    }

    while (!data_found) {
        unsigned char chunk[8];
        uint32_t chunk_size;
        if (fread(chunk, 1, sizeof(chunk), fp) != sizeof(chunk)) {
            break;
        }
        chunk_size = read_u32_le(chunk + 4);
        if (memcmp(chunk, "fmt ", 4) == 0) {
            unsigned char fmt[32];
            if (chunk_size > sizeof(fmt)) {
                fprintf(stderr, "unsupported large fmt chunk\n");
                fclose(fp);
                return -1;
            }
            if (fread(fmt, 1, chunk_size, fp) != chunk_size) {
                fclose(fp);
                return -1;
            }
            audio_format = read_u16_le(fmt);
            channels = read_u16_le(fmt + 2);
            sample_rate = read_u32_le(fmt + 4);
            bits_per_sample = read_u16_le(fmt + 14);
            fmt_ok = 1;
        } else if (memcmp(chunk, "data", 4) == 0) {
            data_size = chunk_size;
            data_found = 1;
            break;
        } else {
            if (fseek(fp, (long)(chunk_size + (chunk_size & 1U)), SEEK_CUR) != 0) {
                fclose(fp);
                return -1;
            }
        }
    }

    if (!fmt_ok || !data_found) {
        fprintf(stderr, "missing fmt or data chunk in WAV: %s\n", path);
        fclose(fp);
        return -1;
    }
    if (audio_format != 1 || channels != 1 || sample_rate != 16000 || bits_per_sample != 16) {
        fprintf(stderr, "WAV must be 16-bit PCM mono 16 kHz; got format=%u channels=%u rate=%u bits=%u\n",
                audio_format, channels, sample_rate, bits_per_sample);
        fclose(fp);
        return -1;
    }

    samples_to_read = (int)(data_size / 2U);
    if (samples_to_read > max_samples) {
        samples_to_read = max_samples;
    }
    if ((int)fread(pcm, sizeof(int16_t), (size_t)samples_to_read, fp) != samples_to_read) {
        fprintf(stderr, "failed to read WAV samples\n");
        fclose(fp);
        return -1;
    }
    fclose(fp);
    return samples_to_read;
}
