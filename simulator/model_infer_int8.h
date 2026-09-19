#ifndef SIMULATOR_MODEL_INFER_INT8_H
#define SIMULATOR_MODEL_INFER_INT8_H

#include "generated/model_weights_int8.h"

int model_predict_int8(const float features[FEATURE_DIM],
                       float probabilities[NUM_CLASSES]);
const char *model_label_name_int8(int index);

#endif
