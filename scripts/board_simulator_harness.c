#include "audio_classifier.h"

#include <stdio.h>

/* Keep the float and INT8 generated weight headers in separate translation
 * units, exactly as the OpenVela build does. */
const char *model_label_name_int8(int index);

int main(int argc, char **argv)
{
  int index;

  if (argc < 2)
    {
      fprintf(stderr, "usage: %s WAV...\n", argv[0]);
      return 2;
    }

  puts("file\tpredicted\tconfidence\tenergy\twindows\talert");
  for (index = 1; index < argc; index++)
    {
      aed_result_t result;
      if (aed_classify_three_second_wav(argv[index], &result) < 0)
        {
          fprintf(stderr, "classification failed: %s\n", argv[index]);
          return 1;
        }

      printf("%s\t%s\t%.9f\t%.9f\t%d\t%s\n",
             argv[index],
             model_label_name_int8(result.class_index),
             result.confidence,
             result.energy,
             result.analyzed_windows,
             result.class_index != 0 && result.confidence >= 0.80f
               ? "yes" : "no");
    }

  return 0;
}
