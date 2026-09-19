include $(APPDIR)/Make.defs

PROGNAME = audiodetect
PRIORITY = $(CONFIG_AE_AUDIODETECT_PRIORITY)
STACKSIZE = $(CONFIG_AE_AUDIODETECT_STACKSIZE)
MODULE = $(CONFIG_AE_AUDIODETECT)

CSRCS = wav_reader.c feature_extractor.c model_infer.c model_infer_int8.c alarm_report.c keyword_detect.c

ifeq ($(CONFIG_AE_AUDIODETECT_UI),y)
CSRCS += audiodetect_ui.c audio_classifier.c
endif

MAINSRC = audiodetect_main.c

include $(APPDIR)/Application.mk
