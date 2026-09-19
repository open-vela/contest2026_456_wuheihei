#include "audio_classifier.h"

#include "feature_extractor.h"
#include "wav_reader.h"

#include <stdint.h>
#include <string.h>

#define AED_SEGMENT_SECONDS 3
#define AED_TOTAL_SAMPLES (AUDIO_NUM_SAMPLES * AED_SEGMENT_SECONDS)
#define AED_QUIET_ENERGY_GATE 2.0e-5f
#define AED_EVENT_THRESHOLD 0.80f

int model_predict_int8(const float features[FEATURE_DIM],
                       float probabilities[NUM_CLASSES]);

int aed_classify_three_second_wav(const char *path, aed_result_t *result)
{
  static int16_t pcm[AED_TOTAL_SAMPLES];
  static float features[AUDIO_FEATURE_DIM];
  float average[NUM_CLASSES] = {0};
  float best_event_probs[NUM_CLASSES] = {0};
  float probabilities[NUM_CLASSES];
  float best_event_confidence = 0.0f;
  int best_event_class = 0;
  int sample_count;
  int windows;
  int window;
  int i;

  if (path == NULL || result == NULL)
    {
      return -1;
    }

  memset(result, 0, sizeof(*result));
  sample_count = read_wav_16k_mono(path, pcm, AED_TOTAL_SAMPLES);
  if (sample_count <= 0)
    {
      return -1;
    }

  windows = (sample_count + AUDIO_NUM_SAMPLES - 1) / AUDIO_NUM_SAMPLES;
  if (windows > AED_SEGMENT_SECONDS)
    {
      windows = AED_SEGMENT_SECONDS;
    }

  for (window = 0; window < windows; window++)
    {
      int prediction;

      extract_audio_features(pcm + window * AUDIO_NUM_SAMPLES,
                             AUDIO_NUM_SAMPLES, features);
      if (features[0] > result->energy)
        {
          result->energy = features[0];
        }

      memset(probabilities, 0, sizeof(probabilities));
      if (features[0] < AED_QUIET_ENERGY_GATE)
        {
          probabilities[0] = 1.0f;
          prediction = 0;
        }
      else
        {
          prediction = model_predict_int8(features, probabilities);
        }

      for (i = 0; i < NUM_CLASSES; i++)
        {
          average[i] += probabilities[i];
        }

      if (prediction != 0 &&
          probabilities[prediction] > best_event_confidence)
        {
          best_event_class = prediction;
          best_event_confidence = probabilities[prediction];
          memcpy(best_event_probs, probabilities, sizeof(best_event_probs));
        }
    }

  result->analyzed_windows = windows;
  if (best_event_confidence >= AED_EVENT_THRESHOLD)
    {
      result->class_index = best_event_class;
      result->confidence = best_event_confidence;
      memcpy(result->probabilities, best_event_probs,
             sizeof(result->probabilities));
      return 0;
    }

  result->class_index = 0;
  for (i = 0; i < NUM_CLASSES; i++)
    {
      result->probabilities[i] = average[i] / windows;
      if (result->probabilities[i] >
          result->probabilities[result->class_index])
        {
          result->class_index = i;
        }
    }

  result->confidence = result->probabilities[result->class_index];
  return 0;
}
