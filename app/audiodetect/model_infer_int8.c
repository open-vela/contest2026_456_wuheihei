/*
 * INT8 quantized model inference.
 *
 * Weights are stored as int8 with per-tensor scale factors.
 * During inference, each weight is dequantized on-the-fly:
 *   float_weight = (float)int8_weight * scale
 *
 * This reduces model storage by ~75% (4x compression) with
 * negligible accuracy loss (<0.5% typically).
 *
 * Drop-in replacement for model_infer.c — just change the include
 * and call model_predict_int8() instead of model_predict().
 */
#include "model_infer_int8.h"
#include <math.h>

int model_predict_int8(const float features[FEATURE_DIM], float probabilities[NUM_CLASSES])
{
    float hidden[HIDDEN_DIM];
    float logits[NUM_CLASSES];
    float max_logit;
    float sum_exp = 0.0f;
    int i;
    int j;

    /* Layer 1: FEATURE_DIM -> HIDDEN_DIM with ReLU */
    for (j = 0; j < HIDDEN_DIM; ++j) {
        float acc = (float)B1_INT8[j] * B1_SCALE;  /* dequantize bias */
        for (i = 0; i < FEATURE_DIM; ++i) {
            float std = FEATURE_STD[i] == 0.0f ? 1.0f : FEATURE_STD[i];
            float x = (features[i] - FEATURE_MEAN[i]) / std;
            /* dequantize weight on-the-fly */
            float w = (float)W1_INT8[i][j] * W1_SCALE;
            acc += x * w;
        }
        hidden[j] = acc > 0.0f ? acc : 0.0f;  /* ReLU */
    }

    /* Layer 2: HIDDEN_DIM -> NUM_CLASSES */
    for (j = 0; j < NUM_CLASSES; ++j) {
        float acc = (float)B2_INT8[j] * B2_SCALE;  /* dequantize bias */
        for (i = 0; i < HIDDEN_DIM; ++i) {
            float w = (float)W2_INT8[i][j] * W2_SCALE;  /* dequantize weight */
            acc += hidden[i] * w;
        }
        logits[j] = acc;
    }

    /* Softmax */
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

    /* Return argmax */
    j = 0;
    for (i = 1; i < NUM_CLASSES; ++i) {
        if (probabilities[i] > probabilities[j]) {
            j = i;
        }
    }
    return j;
}

const char *model_label_name_int8(int index)
{
    if (index < 0 || index >= NUM_CLASSES) {
        return "unknown";
    }
    return LABEL_NAMES[index];
}
