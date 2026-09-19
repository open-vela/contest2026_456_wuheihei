# CMakeLists.txt for audiodetect app (OpenVela / NuttX CMake build)
# Lightweight offline audio event detection demo (5 classes + alarm + keyword + int8)

if(CONFIG_AE_AUDIODETECT)
  set(AUDIODETECT_SRCS
      audiodetect_main.c
      wav_reader.c
      feature_extractor.c
      model_infer.c
      model_infer_int8.c
      alarm_report.c
      keyword_detect.c)

  if(CONFIG_AE_AUDIODETECT_UI)
    list(APPEND AUDIODETECT_SRCS audiodetect_ui.c audio_classifier.c)
    set(AUDIODETECT_DEPENDS lvgl)
  endif()

  nuttx_add_application(
    NAME
    ${CONFIG_AE_AUDIODETECT_PROGNAME}
    MODULE
    ${CONFIG_AE_AUDIODETECT}
    SRCS
    ${AUDIODETECT_SRCS}
    DEPENDS
    ${AUDIODETECT_DEPENDS}
    STACKSIZE
    ${CONFIG_AE_AUDIODETECT_STACKSIZE}
    PRIORITY
    ${CONFIG_AE_AUDIODETECT_PRIORITY})
endif()
