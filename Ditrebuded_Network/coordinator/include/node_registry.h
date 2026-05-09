#pragma once

#include "protocol.h"
#include "config.h"
#include <Arduino.h>

struct NodeInfo {
    char     node_id[MAX_NODE_ID_LEN];
    NodeType node_type;
    uint8_t  mac[6];
    uint32_t last_seen_ms;
    uint32_t first_seen_ms;
    uint32_t total_packets;
    char     last_top_label[MAX_LABEL_LEN];
    uint8_t  last_top_score;
    bool     is_online;
    bool     active;         // slot in use
};

class NodeRegistry {
public:
    NodeInfo* update(const uint8_t* mac, const InferencePacket& pkt) {
        NodeInfo* n = findByMac(mac);
        if (!n) n = allocate();
        if (!n) return nullptr;

        bool isNew = !n->active;
        n->active = true;
        memcpy(n->mac, mac, 6);
        strncpy(n->node_id, pkt.node_id, MAX_NODE_ID_LEN - 1);
        n->node_id[MAX_NODE_ID_LEN - 1] = '\0';
        n->node_type = pkt.node_type;
        n->last_seen_ms = millis();
        if (isNew) n->first_seen_ms = n->last_seen_ms;
        n->total_packets++;

        if (pkt.top[0].score > 0) {
            strncpy(n->last_top_label, pkt.top[0].label, MAX_LABEL_LEN - 1);
            n->last_top_label[MAX_LABEL_LEN - 1] = '\0';
            n->last_top_score = pkt.top[0].score;
        }
        n->is_online = true;
        return n;
    }

    bool refreshOnlineStatus() {
        bool changed = false;
        uint32_t now = millis();
        for (int i = 0; i < MAX_NODES; ++i) {
            if (!_nodes[i].active) continue;
            bool shouldBeOnline = (now - _nodes[i].last_seen_ms) < NODE_TIMEOUT_MS;
            if (_nodes[i].is_online != shouldBeOnline) {
                _nodes[i].is_online = shouldBeOnline;
                changed = true;
            }
        }
        return changed;
    }

    int activeCount() const {
        int c = 0;
        for (int i = 0; i < MAX_NODES; ++i)
            if (_nodes[i].active) ++c;
        return c;
    }

    int onlineCount() const {
        int c = 0;
        for (int i = 0; i < MAX_NODES; ++i)
            if (_nodes[i].active && _nodes[i].is_online) ++c;
        return c;
    }

    const NodeInfo* node(int idx) const {
        if (idx < 0 || idx >= MAX_NODES) return nullptr;
        return _nodes[idx].active ? &_nodes[idx] : nullptr;
    }

    template <typename Fn>
    void forEach(Fn fn) const {
        for (int i = 0; i < MAX_NODES; ++i)
            if (_nodes[i].active) fn(_nodes[i]);
    }

private:
    NodeInfo* findByMac(const uint8_t* mac) {
        for (int i = 0; i < MAX_NODES; ++i)
            if (_nodes[i].active && memcmp(_nodes[i].mac, mac, 6) == 0)
                return &_nodes[i];
        return nullptr;
    }

    NodeInfo* allocate() {
        for (int i = 0; i < MAX_NODES; ++i)
            if (!_nodes[i].active) return &_nodes[i];
        return nullptr;
    }

    NodeInfo _nodes[MAX_NODES] = {};
};
