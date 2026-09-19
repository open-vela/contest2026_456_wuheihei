#ifndef MODEL_INFER_H
#define MODEL_INFER_H

#include "model_weights.h"

int model_predict(const float features[FEATURE_DIM], float probabilities[NUM_CLASSES]);
const char *model_label_name(int index);

#endif
