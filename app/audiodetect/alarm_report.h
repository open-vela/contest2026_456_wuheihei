#ifndef ALARM_REPORT_H
#define ALARM_REPORT_H

#include <stdint.h>

/*
 * Alarm Report Module
 *
 * Provides remote alarm reporting via HTTP POST and MQTT publish.
 * On OpenVela, network sockets are available via the POSIX layer.
 * On PC simulation, the same code compiles with standard sockets.
 *
 * Usage:
 *   alarm_report_init("192.168.1.100", 8080, "audiodetect/device_01");
 *   alarm_send_http(event_label, confidence, timestamp);
 *   alarm_send_mqtt(event_label, confidence, timestamp);
 */

/* Maximum lengths for configuration strings */
#define ALARM_MAX_HOST     64
#define ALARM_MAX_TOPIC    128
#define ALARM_MAX_JSON     512

/* Alarm severity levels */
typedef enum {
    ALARM_SEVERITY_INFO = 0,
    ALARM_SEVERITY_WARNING = 1,
    ALARM_SEVERITY_CRITICAL = 2
} alarm_severity_t;

/* Initialize the alarm reporter.
 * host:     IP address or hostname of the upper-computer server
 * http_port: HTTP port for REST API (e.g. 8080)
 * mqtt_port: MQTT broker port (e.g. 1883)
 * device_id: unique device identifier string
 * Returns 0 on success, -1 on error. */
int alarm_report_init(const char *host, int http_port, int mqtt_port,
                      const char *device_id);

/* Send alarm via HTTP POST.
 * event_label: detected event name (e.g. "glass_break")
 * confidence:  detection confidence [0.0, 1.0]
 * timestamp:   Unix timestamp in seconds
 * Returns 0 on success, -1 on error. */
int alarm_send_http(const char *event_label, float confidence,
                    int severity, int64_t timestamp);

/* Send alarm via MQTT publish.
 * event_label: detected event name
 * confidence:  detection confidence [0.0, 1.0]
 * timestamp:   Unix timestamp in seconds
 * Returns 0 on success, -1 on error. */
int alarm_send_mqtt(const char *event_label, float confidence,
                    int severity, int64_t timestamp);

/* Build the alarm JSON payload without sending.
 * Useful for testing or custom transport.
 * Returns the JSON string (static buffer, overwritten on next call). */
const char *alarm_build_json(const char *event_label, float confidence,
                             int severity, int64_t timestamp);

/* Get the configured device ID. */
const char *alarm_get_device_id(void);

/* Get the configured MQTT topic. */
const char *alarm_get_mqtt_topic(void);

#endif /* ALARM_REPORT_H */
