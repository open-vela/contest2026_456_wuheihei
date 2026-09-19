/*
 * alarm_report.c - Remote alarm reporting via HTTP POST and MQTT
 *
 * Implements:
 *   - JSON payload construction (lightweight, no external JSON library)
 *   - HTTP POST to a REST endpoint (upper-computer)
 *   - MQTT CONNECT + PUBLISH (minimal MQTT 3.1.1 client, no external lib)
 *   - Works on both OpenVela (POSIX sockets) and PC (Linux/Win32)
 *
 * Design choices:
 *   - No dynamic allocation (all static buffers) → safe for embedded
 *   - No external dependencies (no cJSON, no paho-mqtt)
 *   - MQTT implements just enough of 3.1.1 to publish one message
 *   - HTTP uses raw socket with Content-Length header
 */

#include "alarm_report.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* Platform-specific socket headers */
#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
    typedef int socklen_t_win;
    #define ALARM_CLOSE_SOCKET closesocket
    #define ALARM_SOCKET_ERR SOCKET_ERROR
#else
    #include <unistd.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <netdb.h>
    #define ALARM_CLOSE_SOCKET close
    #define ALARM_SOCKET_ERR (-1)
#endif

/* ---- Static configuration (set by alarm_report_init) ---- */
static char s_host[ALARM_MAX_HOST]       = "127.0.0.1";
static int  s_http_port                  = 8080;
static int  s_mqtt_port                  = 1883;
static char s_device_id[64]              = "device_01";
static char s_mqtt_topic[ALARM_MAX_TOPIC] = "audiodetect/alarm";
static int  s_initialized                 = 0;

/* ---- Severity names ---- */
static const char *severity_names[] = {
    "info", "warning", "critical"
};

/* ---- JSON builder ---- */
const char *alarm_build_json(const char *event_label, float confidence,
                             int severity, int64_t timestamp)
{
    static char json_buf[ALARM_MAX_JSON];
    const char *sev = (severity >= 0 && severity <= 2)
                      ? severity_names[severity] : "unknown";

    int n = snprintf(json_buf, sizeof(json_buf),
        "{\"device\":\"%s\","
        "\"event\":\"%s\","
        "\"confidence\":%.4f,"
        "\"severity\":\"%s\","
        "\"timestamp\":%lld}",
        s_device_id,
        event_label ? event_label : "unknown",
        confidence,
        sev,
        (long long)timestamp);

    if (n < 0 || (size_t)n >= sizeof(json_buf)) {
        /* Truncated — shouldn't happen with our buffer sizes */
        json_buf[sizeof(json_buf) - 1] = '\0';
    }
    return json_buf;
}

int alarm_report_init(const char *host, int http_port, int mqtt_port,
                      const char *device_id)
{
    if (!host || !device_id) return -1;

    strncpy(s_host, host, sizeof(s_host) - 1);
    s_host[sizeof(s_host) - 1] = '\0';
    s_http_port = http_port;
    s_mqtt_port = mqtt_port;

    strncpy(s_device_id, device_id, sizeof(s_device_id) - 1);
    s_device_id[sizeof(s_device_id) - 1] = '\0';

    /* Build MQTT topic from device_id */
    snprintf(s_mqtt_topic, sizeof(s_mqtt_topic),
             "audiodetect/%s/alarm", s_device_id);

#ifdef _WIN32
    /* Initialize Winsock once */
    static int wsa_done = 0;
    if (!wsa_done) {
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
        wsa_done = 1;
    }
#endif

    s_initialized = 1;
    printf("[alarm] initialized: host=%s http=%d mqtt=%d device=%s topic=%s\n",
           s_host, s_http_port, s_mqtt_port, s_device_id, s_mqtt_topic);
    return 0;
}

const char *alarm_get_device_id(void)  { return s_device_id; }
const char *alarm_get_mqtt_topic(void) { return s_mqtt_topic; }

/* ---- Helper: create and connect a TCP socket ---- */
static int connect_tcp(const char *host, int port)
{
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);

    /* Try parsing as IP first, then DNS lookup */
    if (inet_pton(AF_INET, host, &addr.sin_addr) <= 0) {
        struct hostent *he = gethostbyname(host);
        if (!he) return -1;
        memcpy(&addr.sin_addr, he->h_addr_list[0], he->h_length);
    }

    int sock = (int)socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return -1;

    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        ALARM_CLOSE_SOCKET(sock);
        return -1;
    }
    return sock;
}

/* ---- HTTP POST implementation ---- */
int alarm_send_http(const char *event_label, float confidence,
                    int severity, int64_t timestamp)
{
    if (!s_initialized) return -1;

    const char *json = alarm_build_json(event_label, confidence,
                                        severity, timestamp);

    int sock = connect_tcp(s_host, s_http_port);
    if (sock < 0) {
        printf("[alarm] HTTP: failed to connect %s:%d\n", s_host, s_http_port);
        return -1;
    }

    /* Build HTTP POST request */
    char request[1024];
    int req_len = snprintf(request, sizeof(request),
        "POST /api/alarm HTTP/1.1\r\n"
        "Host: %s:%d\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "\r\n"
        "%s",
        s_host, s_http_port, strlen(json), json);

    if (req_len < 0 || (size_t)req_len >= sizeof(request)) {
        ALARM_CLOSE_SOCKET(sock);
        return -1;
    }

    /* Send request */
    int sent = 0;
    while (sent < req_len) {
        int n = (int)send(sock, request + sent, req_len - sent, 0);
        if (n <= 0) {
            ALARM_CLOSE_SOCKET(sock);
            return -1;
        }
        sent += n;
    }

    /* Read response (we just check for 200 OK) */
    char response[256];
    int resp_len = (int)recv(sock, response, sizeof(response) - 1, 0);
    ALARM_CLOSE_SOCKET(sock);

    if (resp_len <= 0) return -1;
    response[resp_len] = '\0';

    /* Check for HTTP 200 */
    if (strstr(response, "200") || strstr(response, "201")) {
        printf("[alarm] HTTP POST OK → %s:%d/api/alarm\n", s_host, s_http_port);
        return 0;
    }
    printf("[alarm] HTTP response: %.80s\n", response);
    return -1;
}

/* ---- Minimal MQTT 3.1.1 client (CONNECT + PUBLISH only) ---- */

/* MQTT Remaining Length encoding (variable-length integer) */
static int mqtt_encode_remaining_length(uint8_t *buf, int value)
{
    int bytes = 0;
    do {
        uint8_t encoded = (uint8_t)(value % 128);
        value /= 128;
        if (value > 0) encoded |= 0x80;
        buf[bytes++] = encoded;
    } while (value > 0 && bytes < 4);
    return bytes;
}

int alarm_send_mqtt(const char *event_label, float confidence,
                    int severity, int64_t timestamp)
{
    if (!s_initialized) return -1;

    const char *json = alarm_build_json(event_label, confidence,
                                        severity, timestamp);

    int sock = connect_tcp(s_host, s_mqtt_port);
    if (sock < 0) {
        printf("[alarm] MQTT: failed to connect %s:%d\n", s_host, s_mqtt_port);
        return -1;
    }

    /* --- Step 1: MQTT CONNECT packet --- */
    /*
     * CONNECT packet structure (MQTT 3.1.1):
     *   Fixed header: 0x10, remaining_length
     *   Variable header: Protocol Name "MQTT", Level 4, Connect Flags, Keep Alive
     *   Payload: Client ID
     */
    const char *client_id = s_device_id;
    int client_id_len = (int)strlen(client_id);

    /* Variable header: 10 bytes (protocol name + level + flags + keepalive) */
    /* Payload: 2 + client_id_len */
    int connect_remaining = 10 + 2 + client_id_len;

    uint8_t connect_pkt[128];
    int idx = 0;
    connect_pkt[idx++] = 0x10;  /* CONNECT packet type */
    idx += mqtt_encode_remaining_length(connect_pkt + idx, connect_remaining);

    /* Protocol Name: "MQTT" */
    connect_pkt[idx++] = 0x00;
    connect_pkt[idx++] = 0x04;
    connect_pkt[idx++] = 'M';
    connect_pkt[idx++] = 'Q';
    connect_pkt[idx++] = 'T';
    connect_pkt[idx++] = 'T';
    /* Protocol Level: 4 = MQTT 3.1.1 */
    connect_pkt[idx++] = 0x04;
    /* Connect Flags: Clean Session = 1 */
    connect_pkt[idx++] = 0x02;
    /* Keep Alive: 60 seconds */
    connect_pkt[idx++] = 0x00;
    connect_pkt[idx++] = 0x3C;
    /* Payload: Client ID length + Client ID */
    connect_pkt[idx++] = (uint8_t)(client_id_len >> 8);
    connect_pkt[idx++] = (uint8_t)(client_id_len & 0xFF);
    memcpy(connect_pkt + idx, client_id, client_id_len);
    idx += client_id_len;

    int connect_len = idx;
    int sent = 0;
    while (sent < connect_len) {
        int n = (int)send(sock, (char *)(connect_pkt + sent),
                          connect_len - sent, 0);
        if (n <= 0) { ALARM_CLOSE_SOCKET(sock); return -1; }
        sent += n;
    }

    /* Read CONNACK (4 bytes: 0x20 0x02 0x00 0x00 = accepted); loop until full */
    uint8_t connack[4];
    int ack_len = 0;
    while (ack_len < 4) {
        int n = (int)recv(sock, (char *)connack + ack_len, 4 - ack_len, 0);
        if (n <= 0) break;
        ack_len += n;
    }
    if (ack_len < 4 || connack[0] != 0x20 || connack[3] != 0x00) {
        printf("[alarm] MQTT CONNACK failed\n");
        ALARM_CLOSE_SOCKET(sock);
        return -1;
    }

    /* --- Step 2: MQTT PUBLISH packet --- */
    /*
     * PUBLISH packet structure:
     *   Fixed header: 0x30 (QoS 0), remaining_length
     *   Variable header: Topic Name length + Topic Name
     *   Payload: JSON message
     */
    int topic_len = (int)strlen(s_mqtt_topic);
    int payload_len = (int)strlen(json);

    /* remaining_length = 2 (topic len) + topic_len + payload_len */
    int pub_remaining = 2 + topic_len + payload_len;

    uint8_t pub_pkt[ALARM_MAX_JSON + 256];
    idx = 0;
    pub_pkt[idx++] = 0x30;  /* PUBLISH, QoS 0 */
    idx += mqtt_encode_remaining_length(pub_pkt + idx, pub_remaining);

    /* Topic Name */
    pub_pkt[idx++] = (uint8_t)(topic_len >> 8);
    pub_pkt[idx++] = (uint8_t)(topic_len & 0xFF);
    memcpy(pub_pkt + idx, s_mqtt_topic, topic_len);
    idx += topic_len;

    /* Payload */
    memcpy(pub_pkt + idx, json, payload_len);
    idx += payload_len;

    int pub_len = idx;
    sent = 0;
    while (sent < pub_len) {
        int n = (int)send(sock, (char *)(pub_pkt + sent),
                          pub_len - sent, 0);
        if (n <= 0) { ALARM_CLOSE_SOCKET(sock); return -1; }
        sent += n;
    }

    ALARM_CLOSE_SOCKET(sock);
    printf("[alarm] MQTT PUBLISH OK → %s:%d topic=%s\n",
           s_host, s_mqtt_port, s_mqtt_topic);
    return 0;
}
