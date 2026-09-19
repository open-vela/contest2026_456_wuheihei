#include <nuttx/config.h>

#include "audiodetect_ui.h"
#include "audio_classifier.h"

#include <lvgl/lvgl.h>

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define AED_CAPTURE_PATH "/data/aed_live.wav"
#define AED_AUTOSTART_PATH "/data/aed_autostart"
#define AED_CAPTURE_COMMAND \
  "arecord -D default -r 16000 -f 16 -c 1 -d 3 " AED_CAPTURE_PATH
#define AED_HISTORY_SIZE 3
#define AED_DISPLAY_INIT_ATTEMPTS 8
#define AED_DISPLAY_INIT_RETRY_SECONDS 1

enum aed_phase_e
{
  AED_PHASE_IDLE = 0,
  AED_PHASE_RECORDING,
  AED_PHASE_ANALYZING,
  AED_PHASE_ERROR
};

struct aed_shared_state_s
{
  pthread_mutex_t lock;
  bool enabled;
  bool auto_start;
  enum aed_phase_e phase;
  uint32_t segment_count;
  uint32_t event_count;
  uint32_t revision;
  aed_result_t result;
  char error[64];
  char history[AED_HISTORY_SIZE][64];
};

struct aed_widgets_s
{
  lv_obj_t *status_dot;
  lv_obj_t *status_label;
  lv_obj_t *result_label;
  lv_obj_t *confidence_label;
  lv_obj_t *confidence_bar;
  lv_obj_t *energy_label;
  lv_obj_t *probability_bars[NUM_CLASSES];
  lv_obj_t *probability_values[NUM_CLASSES];
  lv_obj_t *start_button;
  lv_obj_t *stop_button;
  lv_obj_t *auto_switch;
  lv_obj_t *segment_label;
  lv_obj_t *event_label;
  lv_obj_t *history_labels[AED_HISTORY_SIZE];
};

struct aed_snapshot_s
{
  bool enabled;
  bool auto_start;
  enum aed_phase_e phase;
  uint32_t segment_count;
  uint32_t event_count;
  aed_result_t result;
  char error[64];
  char history[AED_HISTORY_SIZE][64];
};

static struct aed_shared_state_s g_state =
{
  .lock = PTHREAD_MUTEX_INITIALIZER,
};
static struct aed_widgets_s g_widgets;

static const char *g_class_names[NUM_CLASSES] =
{
  "BACKGROUND", "COUGH", "GLASS BREAK", "BABY CRY", "DOG BARK"
};

static const uint32_t g_class_colors[NUM_CLASSES] =
{
  0x637381, 0x2a6fbb, 0xd54848, 0xe29b27, 0x7b5cc7
};

static void set_auto_start_preference(bool enabled)
{
  FILE *file = fopen(AED_AUTOSTART_PATH, "w");
  if (file != NULL)
    {
      fputs(enabled ? "1\n" : "0\n", file);
      fclose(file);
    }
}

static bool get_auto_start_preference(void)
{
  char value = '1';
  FILE *file = fopen(AED_AUTOSTART_PATH, "r");
  if (file != NULL)
    {
      if (fread(&value, 1, 1, file) != 1)
        {
          value = '1';
        }

      fclose(file);
    }

  return value != '0';
}

static void set_phase(enum aed_phase_e phase, const char *error)
{
  pthread_mutex_lock(&g_state.lock);
  g_state.phase = phase;
  if (error != NULL)
    {
      snprintf(g_state.error, sizeof(g_state.error), "%s", error);
    }
  else
    {
      g_state.error[0] = '\0';
    }

  g_state.revision++;
  pthread_mutex_unlock(&g_state.lock);
}

static void save_result(const aed_result_t *result)
{
  pthread_mutex_lock(&g_state.lock);
  g_state.result = *result;
  g_state.segment_count++;
  if (result->class_index != 0 && result->confidence >= 0.80f)
    {
      int i;
      for (i = AED_HISTORY_SIZE - 1; i > 0; i--)
        {
          memcpy(g_state.history[i], g_state.history[i - 1],
                 sizeof(g_state.history[i]));
        }

      g_state.event_count++;
      snprintf(g_state.history[0], sizeof(g_state.history[0]),
               "#%lu  %-12s  %3.0f%%",
               (unsigned long)g_state.segment_count,
               g_class_names[result->class_index],
               result->confidence * 100.0f);
    }

  g_state.phase = g_state.enabled ? AED_PHASE_RECORDING : AED_PHASE_IDLE;
  g_state.revision++;
  pthread_mutex_unlock(&g_state.lock);
}

static void *capture_worker(void *arg)
{
  (void)arg;

  system("amixer set 25 1");
  system("amixer set 17 1");

  for (;;)
    {
      bool enabled;
      aed_result_t result;

      pthread_mutex_lock(&g_state.lock);
      enabled = g_state.enabled;
      pthread_mutex_unlock(&g_state.lock);

      if (!enabled)
        {
          set_phase(AED_PHASE_IDLE, NULL);
          usleep(100000);
          continue;
        }

      set_phase(AED_PHASE_RECORDING, NULL);
      if (system(AED_CAPTURE_COMMAND) != 0)
        {
          set_phase(AED_PHASE_ERROR, "Microphone capture failed");
          sleep(1);
          continue;
        }

      set_phase(AED_PHASE_ANALYZING, NULL);
      if (aed_classify_three_second_wav(AED_CAPTURE_PATH, &result) < 0)
        {
          set_phase(AED_PHASE_ERROR, "Audio analysis failed");
          sleep(1);
          continue;
        }

      save_result(&result);
    }

  return NULL;
}

static lv_obj_t *make_label(lv_obj_t *parent, const char *text,
                            const lv_font_t *font, uint32_t color)
{
  lv_obj_t *label = lv_label_create(parent);
  lv_label_set_text(label, text);
  lv_obj_set_style_text_font(label, font, 0);
  lv_obj_set_style_text_color(label, lv_color_hex(color), 0);
  return label;
}

static lv_obj_t *make_panel(lv_obj_t *parent, int32_t x, int32_t y,
                            int32_t width, int32_t height)
{
  lv_obj_t *panel = lv_obj_create(parent);
  lv_obj_set_pos(panel, x, y);
  lv_obj_set_size(panel, width, height);
  lv_obj_set_style_radius(panel, 6, 0);
  lv_obj_set_style_bg_color(panel, lv_color_hex(0xffffff), 0);
  lv_obj_set_style_border_color(panel, lv_color_hex(0xdce3e8), 0);
  lv_obj_set_style_border_width(panel, 1, 0);
  lv_obj_set_style_pad_all(panel, 0, 0);
  lv_obj_clear_flag(panel, LV_OBJ_FLAG_SCROLLABLE);
  return panel;
}

static lv_obj_t *make_button(lv_obj_t *parent, const char *text,
                             int32_t x, int32_t y, int32_t width,
                             uint32_t color,
                             lv_event_cb_t callback)
{
  lv_obj_t *button = lv_button_create(parent);
  lv_obj_set_pos(button, x, y);
  lv_obj_set_size(button, width, 44);
  lv_obj_set_style_radius(button, 4, 0);
  lv_obj_set_style_bg_color(button, lv_color_hex(color), 0);
  lv_obj_add_event_cb(button, callback, LV_EVENT_CLICKED, NULL);

  lv_obj_t *label = make_label(button, text, &lv_font_montserrat_16,
                               0xffffff);
  lv_obj_center(label);
  return button;
}

static void start_event_cb(lv_event_t *event)
{
  (void)event;
  pthread_mutex_lock(&g_state.lock);
  g_state.enabled = true;
  g_state.revision++;
  pthread_mutex_unlock(&g_state.lock);
}

static void stop_event_cb(lv_event_t *event)
{
  (void)event;
  pthread_mutex_lock(&g_state.lock);
  g_state.enabled = false;
  g_state.revision++;
  pthread_mutex_unlock(&g_state.lock);
}

static void auto_event_cb(lv_event_t *event)
{
  lv_obj_t *toggle = lv_event_get_target_obj(event);
  bool checked = lv_obj_has_state(toggle, LV_STATE_CHECKED);

  pthread_mutex_lock(&g_state.lock);
  g_state.auto_start = checked;
  g_state.enabled = checked;

  g_state.revision++;
  pthread_mutex_unlock(&g_state.lock);
  set_auto_start_preference(checked);
}

static void build_ui(void)
{
  lv_obj_t *screen = lv_screen_active();
  lv_obj_t *header;
  lv_obj_t *result_panel;
  lv_obj_t *control_panel;
  lv_obj_t *footer;
  lv_obj_t *label;

  lv_obj_clean(screen);
  lv_obj_set_style_bg_color(screen, lv_color_hex(0xe9eef2), 0);
  lv_obj_clear_flag(screen, LV_OBJ_FLAG_SCROLLABLE);

  header = lv_obj_create(screen);
  lv_obj_set_pos(header, 0, 0);
  lv_obj_set_size(header, 320, 30);
  lv_obj_set_style_radius(header, 0, 0);
  lv_obj_set_style_border_width(header, 0, 0);
  lv_obj_set_style_bg_color(header, lv_color_hex(0x17232d), 0);
  lv_obj_set_style_pad_all(header, 0, 0);
  lv_obj_clear_flag(header, LV_OBJ_FLAG_SCROLLABLE);

  label = make_label(header, "AUDIO SENTINEL",
                     &lv_font_montserrat_16, 0xffffff);
  lv_obj_set_pos(label, 10, 6);
  label = make_label(header, "OFFLINE",
                     &lv_font_montserrat_10, 0x9fd7cc);
  lv_obj_align(label, LV_ALIGN_RIGHT_MID, -10, 0);

  result_panel = make_panel(screen, 8, 36, 304, 94);
  g_widgets.status_dot = lv_obj_create(result_panel);
  lv_obj_set_pos(g_widgets.status_dot, 12, 11);
  lv_obj_set_size(g_widgets.status_dot, 8, 8);
  lv_obj_set_style_radius(g_widgets.status_dot, LV_RADIUS_CIRCLE, 0);
  lv_obj_set_style_border_width(g_widgets.status_dot, 0, 0);
  lv_obj_set_style_bg_color(g_widgets.status_dot, lv_color_hex(0x637381), 0);

  g_widgets.status_label = make_label(result_panel, "READY",
                                      &lv_font_montserrat_12, 0x637381);
  lv_obj_set_pos(g_widgets.status_label, 27, 7);
  lv_obj_set_width(g_widgets.status_label, 260);

  g_widgets.result_label = make_label(result_panel, "BACKGROUND",
                                      &lv_font_montserrat_20, 0x17232d);
  lv_obj_set_pos(g_widgets.result_label, 12, 27);
  lv_obj_set_width(g_widgets.result_label, 280);

  g_widgets.confidence_label = make_label(result_panel, "Confidence --",
                                          &lv_font_montserrat_12, 0x384854);
  lv_obj_set_pos(g_widgets.confidence_label, 12, 53);
  g_widgets.energy_label = make_label(result_panel, "Energy --",
                                      &lv_font_montserrat_10, 0x637381);
  lv_obj_align(g_widgets.energy_label, LV_ALIGN_TOP_RIGHT, -12, 54);

  g_widgets.confidence_bar = lv_bar_create(result_panel);
  lv_obj_set_pos(g_widgets.confidence_bar, 12, 74);
  lv_obj_set_size(g_widgets.confidence_bar, 280, 10);
  lv_bar_set_range(g_widgets.confidence_bar, 0, 100);
  lv_obj_set_style_radius(g_widgets.confidence_bar, 3, LV_PART_MAIN);
  lv_obj_set_style_radius(g_widgets.confidence_bar, 3, LV_PART_INDICATOR);
  lv_obj_set_style_bg_color(g_widgets.confidence_bar,
                            lv_color_hex(0xe8edf1), LV_PART_MAIN);
  lv_obj_set_style_bg_color(g_widgets.confidence_bar,
                            lv_color_hex(0x0b8f78), LV_PART_INDICATOR);

  control_panel = make_panel(screen, 8, 136, 304, 62);
  g_widgets.start_button = make_button(control_panel, "START", 10, 9, 80,
                                       0x0b8f78, start_event_cb);
  g_widgets.stop_button = make_button(control_panel, "STOP", 98, 9, 80,
                                      0xd54848, stop_event_cb);

  label = make_label(control_panel, "AUTO",
                     &lv_font_montserrat_12, 0x384854);
  lv_obj_set_pos(label, 190, 8);
  g_widgets.auto_switch = lv_switch_create(control_panel);
  lv_obj_set_pos(g_widgets.auto_switch, 246, 6);
  lv_obj_set_size(g_widgets.auto_switch, 48, 28);
  lv_obj_set_style_bg_color(g_widgets.auto_switch, lv_color_hex(0x0b8f78),
                            LV_PART_INDICATOR | LV_STATE_CHECKED);
  lv_obj_add_event_cb(g_widgets.auto_switch, auto_event_cb,
                      LV_EVENT_VALUE_CHANGED, NULL);

  label = make_label(control_panel, "3-second windows",
                     &lv_font_montserrat_10, 0x637381);
  lv_obj_set_pos(label, 190, 40);

  footer = make_panel(screen, 8, 204, 304, 28);
  g_widgets.segment_label = make_label(footer, "Windows 0",
                                        &lv_font_montserrat_10, 0x384854);
  lv_obj_set_pos(g_widgets.segment_label, 10, 7);
  g_widgets.event_label = make_label(footer, "Alerts 0",
                                      &lv_font_montserrat_10, 0x384854);
  lv_obj_set_pos(g_widgets.event_label, 112, 7);
  label = make_label(footer, "LOCAL",
                     &lv_font_montserrat_10, 0x0b8f78);
  lv_obj_align(label, LV_ALIGN_RIGHT_MID, -10, 0);
}

static void refresh_ui(lv_timer_t *timer)
{
  struct aed_snapshot_s snapshot;
  uint32_t status_color = 0x637381;
  uint32_t result_color;
  const char *status_text = "READY";
  int confidence;

  (void)timer;
  pthread_mutex_lock(&g_state.lock);
  snapshot.enabled = g_state.enabled;
  snapshot.auto_start = g_state.auto_start;
  snapshot.phase = g_state.phase;
  snapshot.segment_count = g_state.segment_count;
  snapshot.event_count = g_state.event_count;
  snapshot.result = g_state.result;
  memcpy(snapshot.error, g_state.error, sizeof(snapshot.error));
  memcpy(snapshot.history, g_state.history, sizeof(snapshot.history));
  pthread_mutex_unlock(&g_state.lock);

  if (snapshot.phase == AED_PHASE_RECORDING)
    {
      status_text = snapshot.enabled ? "LISTENING - CAPTURING 3S"
                                     : "STOPPING CURRENT WINDOW";
      status_color = 0x0b8f78;
    }
  else if (snapshot.phase == AED_PHASE_ANALYZING)
    {
      status_text = "ANALYZING";
      status_color = 0x2a6fbb;
    }
  else if (snapshot.phase == AED_PHASE_ERROR)
    {
      status_text = snapshot.error[0] ? snapshot.error : "AUDIO ERROR";
      status_color = 0xd54848;
    }
  else if (!snapshot.enabled)
    {
      status_text = "PAUSED";
    }

  lv_label_set_text(g_widgets.status_label, status_text);
  lv_obj_set_style_text_color(g_widgets.status_label,
                              lv_color_hex(status_color), 0);
  lv_obj_set_style_bg_color(g_widgets.status_dot,
                            lv_color_hex(status_color), 0);

  result_color = g_class_colors[snapshot.result.class_index];
  lv_label_set_text(g_widgets.result_label,
                    g_class_names[snapshot.result.class_index]);
  lv_obj_set_style_text_color(g_widgets.result_label,
                              lv_color_hex(result_color), 0);

  confidence = (int)(snapshot.result.confidence * 100.0f + 0.5f);
  lv_label_set_text_fmt(g_widgets.confidence_label, "Confidence  %d%%",
                        confidence);
  lv_bar_set_value(g_widgets.confidence_bar, confidence, LV_ANIM_ON);
  lv_obj_set_style_bg_color(g_widgets.confidence_bar,
                            lv_color_hex(result_color), LV_PART_INDICATOR);
  lv_label_set_text_fmt(g_widgets.energy_label, "Energy  %.6f",
                        snapshot.result.energy);

  if (snapshot.enabled)
    {
      lv_obj_add_state(g_widgets.start_button, LV_STATE_DISABLED);
      lv_obj_clear_state(g_widgets.stop_button, LV_STATE_DISABLED);
    }
  else
    {
      lv_obj_clear_state(g_widgets.start_button, LV_STATE_DISABLED);
      lv_obj_add_state(g_widgets.stop_button, LV_STATE_DISABLED);
    }

  if (snapshot.auto_start)
    {
      lv_obj_add_state(g_widgets.auto_switch, LV_STATE_CHECKED);
    }
  else
    {
      lv_obj_clear_state(g_widgets.auto_switch, LV_STATE_CHECKED);
    }

  lv_label_set_text_fmt(g_widgets.segment_label, "Windows %lu",
                        (unsigned long)snapshot.segment_count);
  lv_label_set_text_fmt(g_widgets.event_label, "Alerts %lu",
                        (unsigned long)snapshot.event_count);
}

int audiodetect_ui_main(int argc, char *argv[])
{
  lv_nuttx_dsc_t info;
  lv_nuttx_result_t result;
  pthread_attr_t attributes;
  pthread_t worker;
  int attempt;

  (void)argc;
  (void)argv;

  if (lv_is_initialized())
    {
      fprintf(stderr, "Audio UI is already running\n");
      return 1;
    }

  g_state.auto_start = get_auto_start_preference();
  g_state.enabled = g_state.auto_start;
  g_state.phase = g_state.enabled ? AED_PHASE_RECORDING : AED_PHASE_IDLE;
  snprintf(g_state.history[0], sizeof(g_state.history[0]), "--");
  snprintf(g_state.history[1], sizeof(g_state.history[1]), "--");
  snprintf(g_state.history[2], sizeof(g_state.history[2]), "--");

  lv_init();
  for (attempt = 1; attempt <= AED_DISPLAY_INIT_ATTEMPTS; attempt++)
    {
      lv_nuttx_dsc_init(&info);
#ifdef CONFIG_LV_USE_NUTTX_LCD
      info.fb_path = "/dev/lcd0";
#else
      info.fb_path = "/dev/fb0";
#endif
      info.input_path = CONFIG_AE_AUDIODETECT_UI_INPUT_DEVPATH;
      lv_nuttx_init(&info, &result);
      if (result.disp != NULL && result.indev != NULL)
        {
          break;
        }

      fprintf(stderr,
              "Audio UI hardware init failed (%d/%d): display=%p input=%p\n",
              attempt, AED_DISPLAY_INIT_ATTEMPTS, result.disp, result.indev);
      lv_nuttx_deinit(&result);

      if (attempt < AED_DISPLAY_INIT_ATTEMPTS)
        {
          lv_deinit();
          sleep(AED_DISPLAY_INIT_RETRY_SECONDS);
          lv_init();
        }
    }

  if (result.disp == NULL || result.indev == NULL)
    {
      fprintf(stderr, "Audio UI hardware did not become ready\n");
      lv_deinit();
      return 1;
    }

#ifndef CONFIG_LV_USE_NUTTX_LCD
  lv_display_set_rotation(result.disp, LV_DISPLAY_ROTATION_180);
#endif

  build_ui();
  lv_timer_create(refresh_ui, 200, NULL);
  refresh_ui(NULL);

  pthread_attr_init(&attributes);
  pthread_attr_setstacksize(&attributes, 65536);
  if (pthread_create(&worker, &attributes, capture_worker, NULL) != 0)
    {
      fprintf(stderr, "Failed to start the audio worker\n");
      pthread_attr_destroy(&attributes);
      return 1;
    }

  pthread_attr_destroy(&attributes);
  pthread_detach(worker);

  for (;;)
    {
      uint32_t idle = lv_timer_handler();
      usleep((idle ? idle : 1) * 1000);
    }

  return 0;
}
