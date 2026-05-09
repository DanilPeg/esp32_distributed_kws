#pragma once

#include "protocol.h"
#include "config.h"
#include <Arduino.h>

// Wrapper that augments a raw packet with coordinator timestamps.
struct QueueItem {
    InferencePacket pkt;
    uint32_t received_at;    // millis() when coordinator received it
    uint32_t processed_at;   // millis() when coordinator processed it (0 = pending)
    uint8_t  sender_mac[6];
};

// ---- Thread-safe FIFO queue (ISR -> loop) ---------------------------------
// Backed by a FreeRTOS queue so the ESP-NOW callback can push safely.

class RequestQueue {
public:
    void begin() {
        _handle = xQueueCreate(QUEUE_SIZE, sizeof(QueueItem));
    }

    bool pushFromISR(const QueueItem& item) {
        if (!_handle) return false;
        BaseType_t woken = pdFALSE;
        BaseType_t ok = xQueueSendToBackFromISR(_handle, &item, &woken);
        if (woken) portYIELD_FROM_ISR();
        return ok == pdTRUE;
    }

    bool pop(QueueItem& out) {
        if (!_handle) return false;
        return xQueueReceive(_handle, &out, 0) == pdTRUE;
    }

    int count() const {
        if (!_handle) return 0;
        return uxQueueMessagesWaiting(_handle);
    }

private:
    QueueHandle_t _handle = nullptr;
};

// ---- Circular history buffer (non-thread-safe, only used in loop) ---------

template <int CAPACITY>
class CircularHistory {
public:
    void push(const QueueItem& item) {
        _buf[_head] = item;
        _head = (_head + 1) % CAPACITY;
        if (_count < CAPACITY) ++_count;
    }

    int count() const { return _count; }

    const QueueItem& at(int i) const {
        int start = (_count < CAPACITY) ? 0 : _head;
        return _buf[(start + i) % CAPACITY];
    }

    const QueueItem& newest() const { return at(_count - 1); }

    template <typename Fn>
    void forEachNewest(int limit, Fn fn) const {
        int n = min(limit, _count);
        for (int i = 0; i < n; ++i) {
            fn(at(_count - 1 - i));
        }
    }

private:
    QueueItem _buf[CAPACITY];
    int _head  = 0;
    int _count = 0;
};

// ---- System log ring ------------------------------------------------------

class SystemLog {
public:
    void add(const char* msg) {
        Entry& e = _buf[_head];
        e.time_ms = millis();
        strncpy(e.text, msg, sizeof(e.text) - 1);
        e.text[sizeof(e.text) - 1] = '\0';
        _head = (_head + 1) % MAX_LOG_ENTRIES;
        if (_count < MAX_LOG_ENTRIES) ++_count;

        Serial.printf("[LOG %lu] %s\n", e.time_ms, msg);
    }

    void addf(const char* fmt, ...) {
        char tmp[128];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(tmp, sizeof(tmp), fmt, ap);
        va_end(ap);
        add(tmp);
    }

    struct Entry {
        uint32_t time_ms;
        char text[128];
    };

    int count() const { return _count; }

    const Entry& at(int i) const {
        int start = (_count < MAX_LOG_ENTRIES) ? 0 : _head;
        return _buf[(start + i) % MAX_LOG_ENTRIES];
    }

    template <typename Fn>
    void forEachNewest(int limit, Fn fn) const {
        int n = min(limit, _count);
        for (int i = 0; i < n; ++i) {
            fn(at(_count - 1 - i));
        }
    }

private:
    Entry _buf[MAX_LOG_ENTRIES];
    int _head  = 0;
    int _count = 0;
};
