#include "model_infer_int8.h"

#include <math.h>

int model_predict_int8(const float features[FEATURE_DIM],
                       float probabilities[NUM_CLASSES])
{
  float hidden[HIDDEN_DIM];
  float logits[NUM_CLASSES];
  float max_logit;
  float sum_exp = 0.0f;
  int i;
  int j;

  for (j = 0; j < HIDDEN_DIM; ++j)
    {
      float acc = (float)B1_INT8[j] * B1_SCALE;
      for (i = 0; i < FEATURE_DIM; ++i)
        {
          const float std = FEATURE_STD[i] == 0.0f ? 1.0f : FEATURE_STD[i];
          const float normalized = (features[i] - FEATURE_MEAN[i]) / std;
          acc += normalized * (float)W1_INT8[i][j] * W1_SCALE;
        }
      hidden[j] = acc > 0.0f ? acc : 0.0f;
    }

  for (j = 0; j < NUM_CLASSES; ++j)
    {
      float acc = (float)B2_INT8[j] * B2_SCALE;
      for (i = 0; i < HIDDEN_DIM; ++i)
        {
          acc += hidden[i] * (float)W2_INT8[i][j] * W2_SCALE;
        }
      logits[j] = acc;
    }

  max_logit = logits[0];
  for (j = 1; j < NUM_CLASSES; ++j)
    {
      if (logits[j] > max_logit)
        {
          max_logit = logits[j];
        }
    }
  for (j = 0; j < NUM_CLASSES; ++j)
    {
      probabilities[j] = expf(logits[j] - max_logit);
      sum_exp += probabilities[j];
    }
  for (j = 0; j < NUM_CLASSES; ++j)
    {
      probabilities[j] /= sum_exp;
    }

  j = 0;
  for (i = 1; i < NUM_CLASSES; ++i)
    {
      if (probabilities[i] > probabilities[j])
        {
          j = i;
        }
    }
  return j;
}

const char *model_label_name_int8(int index)
{
  if (index < 0 || index >= NUM_CLASSES)
    {
      return "unknown";
    }
  return LABEL_NAMES[index];
}
