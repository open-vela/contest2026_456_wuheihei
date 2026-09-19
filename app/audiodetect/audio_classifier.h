#ifndef AUDIO_CLASSIFIER_H
#define AUDIO_CLASSIFIER_H

#include "model_infer.h"

typedef struct
{
  int class_index;
  float confidence;
  float energy;
  float probabilities[NUM_CLASSES];
  int analyzed_windows;
} aed_result_t;

int aed_classify_three_second_wav(const char *path, aed_result_t *result);

#endif
