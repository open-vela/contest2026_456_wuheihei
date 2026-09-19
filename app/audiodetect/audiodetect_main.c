/* audiodetect_main.c — A/B/C/E/F 全功能主程序（Linux C 与 OpenVela 通用）
 *   A 类别扩展：5 类检测（默认）
 *   B 鲁棒性  ：robust <wav>，clean/20/10/5 dB 噪声下检测对比
 *   C 模型优化：--int8 走 INT8 量化推理
 *   E 告警上报：--report 通过 HTTP/MQTT 上报
 *   F 关键词  ：keyword <wav>，MFCC+DTW 识别 救命/帮助/报警
 */
#include "feature_extractor.h"
#include "model_infer.h"
#include "wav_reader.h"
#include "alarm_report.h"
#include "keyword_detect.h"
#ifdef CONFIG_AE_AUDIODETECT_UI
#include "audiodetect_ui.h"
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define ALARM_SERVER_HOST   "127.0.0.1"
#define ALARM_HTTP_PORT     8080
#define ALARM_MQTT_PORT     1883
#define ALARM_DEVICE_ID     "openvela_aed_01"

/* The board microphone's idle floor is far below the ESC-50 training
 * clips. Avoid turning electrically quiet rooms into a high-confidence
 * event merely because the classifier was never trained on this ADC floor. */
#define QUIET_ENERGY_GATE   2.0e-5f

/* INT8 推理入口：单独前向声明，避免同时包含 float/int8 两个权重头（同名 static 数组冲突） */
int model_predict_int8(const float features[FEATURE_DIM], float probabilities[NUM_CLASSES]);

static void usage(const char *prog)
{
    printf("usage:\n");
    printf("  %s input.wav [--debug|--report|--int8]   classify an audio file\n", prog);
    printf("  %s robust input.wav                      compare several SNR levels\n", prog);
    printf("  %s keyword input.wav                     detect emergency keywords\n", prog);
#ifdef CONFIG_AE_AUDIODETECT_UI
    printf("  %s ui                                    open the touch interface\n", prog);
#endif
}

static int event_severity(const char *label)
{
    if (strcmp(label, "glass_break") == 0) return ALARM_SEVERITY_CRITICAL;
    if (strcmp(label, "baby_cry") == 0)    return ALARM_SEVERITY_WARNING;
    if (strcmp(label, "dog_bark") == 0)    return ALARM_SEVERITY_WARNING;
    if (strcmp(label, "cough") == 0)       return ALARM_SEVERITY_INFO;
    return ALARM_SEVERITY_INFO;
}

/* Box-Muller 高斯噪声（确定性，seed=42） */
static double gauss_rand(void)
{
    double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    double u2 = rand() / (RAND_MAX + 1.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static void add_noise(int16_t *pcm, int n, float snr_db)
{
    double sum = 0.0;
    int i;
    for (i = 0; i < n; i++) sum += (double)pcm[i] * pcm[i];
    double rms = sqrt(sum / n);
    double noise_rms = rms / pow(10.0, snr_db / 20.0);
    srand(42);
    for (i = 0; i < n; i++) {
        double v = pcm[i] + gauss_rand() * noise_rms;
        if (v > 32767.0) v = 32767.0;
        if (v < -32768.0) v = -32768.0;
        pcm[i] = (int16_t)v;
    }
}

/* B：多 SNR 鲁棒性对比 */
static int robust_demo_main(int argc, char *argv[])
{
    static int16_t pcm[AUDIO_NUM_SAMPLES];
    static int16_t noisy[AUDIO_NUM_SAMPLES];
    static float features[AUDIO_FEATURE_DIM];
    float probs[NUM_CLASSES];
    int snrs[] = {20, 10, 5};
    int i, k;

    if (argc != 2) { usage(argv[-1]); return 1; }
    printf("Robustness Demo (multi-SNR)\n");
    printf("Input: %s\n", argv[1]);
    if (read_wav_16k_mono(argv[1], pcm, AUDIO_NUM_SAMPLES) < 0) return 1;

    extract_audio_features(pcm, AUDIO_NUM_SAMPLES, features);
    int pred = model_predict(features, probs);
    printf("  SNR=clean -> %-12s (%.3f)\n", model_label_name(pred), probs[pred]);

    for (k = 0; k < 3; k++) {
        memcpy(noisy, pcm, sizeof(pcm));
        add_noise(noisy, AUDIO_NUM_SAMPLES, (float)snrs[k]);
        extract_audio_features(noisy, AUDIO_NUM_SAMPLES, features);
        pred = model_predict(features, probs);
        printf("  SNR=%3ddB  -> %-12s (%.3f)\n", snrs[k], model_label_name(pred), probs[pred]);
    }
    (void)i;
    return 0;
}

/* F：关键词识别 */
static int keyword_demo_main(int argc, char *argv[])
{
    static int16_t pcm[AUDIO_NUM_SAMPLES];
    float confidence;

    if (argc != 2) { usage(argv[-1]); return 1; }
    printf("Keyword Detection Demo\n");
    printf("Input: %s\n", argv[1]);
    if (read_wav_16k_mono(argv[1], pcm, AUDIO_NUM_SAMPLES) < 0) return 1;

    int kw = keyword_detect(pcm, AUDIO_NUM_SAMPLES, &confidence);
    if (kw >= 0)
        printf("Keyword: %s (%.1f%%)\n", keyword_name(kw), confidence * 100.0f);
    else
        printf("Keyword: none\n");
    return 0;
}

int audiodetect_main(int argc, char *argv[])
{
    static int16_t pcm[AUDIO_NUM_SAMPLES];
    static float features[AUDIO_FEATURE_DIM];
    float probabilities[NUM_CLASSES];
    int pred, i;
    int debug = 0;
    int remote_report = 0;
    int use_int8 = 0;

#ifdef CONFIG_AE_AUDIODETECT_UI
    if (argc >= 2 && strcmp(argv[1], "ui") == 0)
        return audiodetect_ui_main(argc - 1, argv + 1);
#endif

    if (argc >= 2 && strcmp(argv[1], "keyword") == 0)
        return keyword_demo_main(argc - 1, argv + 1);
    if (argc >= 2 && strcmp(argv[1], "robust") == 0)
        return robust_demo_main(argc - 1, argv + 1);

    if (argc < 2 || argc > 4) { usage(argv[0]); return 1; }
    for (i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--debug") == 0) debug = 1;
        else if (strcmp(argv[i], "--report") == 0) remote_report = 1;
        else if (strcmp(argv[i], "--int8") == 0) use_int8 = 1;
        else { usage(argv[0]); return 1; }
    }

    printf("Audio Event Detection Demo\n");
    printf("Input: %s%s\n", argv[1], use_int8 ? " (INT8)" : "");
    if (read_wav_16k_mono(argv[1], pcm, AUDIO_NUM_SAMPLES) < 0) return 1;
    extract_audio_features(pcm, AUDIO_NUM_SAMPLES, features);

    if (features[0] < QUIET_ENERGY_GATE) {
        /* features[0] is the mean normalized frame energy. */
        memset(probabilities, 0, sizeof(probabilities));
        probabilities[0] = 1.0f;
        pred = 0;
        printf("Input energy %.8f is below quiet-room gate\n", features[0]);
    } else if (use_int8) {
        pred = model_predict_int8(features, probabilities);
    } else {
        pred = model_predict(features, probabilities);
    }
    const char *label = model_label_name(pred);
    float confidence = probabilities[pred];

    printf("Result: %s\n", label);
    printf("Confidence: %.3f\n", confidence);

    if (debug) {
        printf("\nProbabilities:\n");
        for (i = 0; i < NUM_CLASSES; i++)
            printf("  %s: %.6f\n", model_label_name(i), probabilities[i]);
    }

    if (pred != 0 && confidence > 0.80f) {
        printf("[ALERT] Local alarm triggered: %s (%.1f%%)\n",
               label, confidence * 100.0f);
        if (remote_report) {
            if (alarm_report_init(ALARM_SERVER_HOST, ALARM_HTTP_PORT,
                                  ALARM_MQTT_PORT, ALARM_DEVICE_ID) == 0) {
                int severity = event_severity(label);
                int64_t ts = (int64_t)time(NULL);
                printf("[ALARM] Payload: %s\n",
                       alarm_build_json(label, confidence, severity, ts));
                int http_ok = alarm_send_http(label, confidence, severity, ts);
                printf("[ALARM] HTTP report: %s\n",
                       http_ok == 0 ? "SUCCESS" : "FAILED (server not running?)");
                int mqtt_ok = alarm_send_mqtt(label, confidence, severity, ts);
                printf("[ALARM] MQTT report: %s\n",
                       mqtt_ok == 0 ? "SUCCESS" : "FAILED (broker not running?)");
            }
        } else {
            printf("[ALERT] Remote reporting disabled (use --report to enable)\n");
        }
    }
    return 0;
}

#ifndef __NuttX__
int main(int argc, char *argv[])
{
    return audiodetect_main(argc, argv);
}
#endif
