#include "model_infer.h"
#include "model_weights.h"

#include <math.h>

int model_predict(const float features[FEATURE_DIM], float probabilities[NUM_CLASSES])
{
    float hidden[HIDDEN_DIM];
    float logits[NUM_CLASSES];
    float max_logit;
    float sum_exp = 0.0f;
    int i;
    int j;

    for (j = 0; j < HIDDEN_DIM; ++j) {
        float acc = B1[j];
        for (i = 0; i < FEATURE_DIM; ++i) {
            float std = FEATURE_STD[i] == 0.0f ? 1.0f : FEATURE_STD[i];
            float x = (features[i] - FEATURE_MEAN[i]) / std;
            acc += x * W1[i][j];
        }
        hidden[j] = acc > 0.0f ? acc : 0.0f;
    }

    for (j = 0; j < NUM_CLASSES; ++j) {
        float acc = B2[j];
        for (i = 0; i < HIDDEN_DIM; ++i) {
            acc += hidden[i] * W2[i][j];
        }
        logits[j] = acc;
    }

    max_logit = logits[0];
    for (j = 1; j < NUM_CLASSES; ++j) {
        if (logits[j] > max_logit) {
            max_logit = logits[j];
        }
    }
    for (j = 0; j < NUM_CLASSES; ++j) {
        probabilities[j] = expf(logits[j] - max_logit);
        sum_exp += probabilities[j];
    }
    for (j = 0; j < NUM_CLASSES; ++j) {
        probabilities[j] /= sum_exp;
    }

    j = 0;
    for (i = 1; i < NUM_CLASSES; ++i) {
        if (probabilities[i] > probabilities[j]) {
            j = i;
        }
    }
    return j;
}

const char *model_label_name(int index)
{
    if (index < 0 || index >= NUM_CLASSES) {
        return "unknown";
    }
    return LABEL_NAMES[index];
}
