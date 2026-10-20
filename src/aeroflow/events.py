"""Event publishing. Backends: jsonl (default), kafka, mqtt (works with Solace PubSub+ MQTT), none."""

from __future__ import annotations

import json
import logging
import threading

log = logging.getLogger("aeroflow.events")


class EventBus:
    def publish(self, topic: str, payload: dict) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        pass


class NullBus(EventBus):
    def publish(self, topic, payload):
        pass


class JsonlBus(EventBus):
    def __init__(self, path: str):
        self._f = open(path, "a", buffering=1, encoding="utf-8")  # noqa: SIM115 - kept open for the process lifetime
        self._lock = threading.Lock()

    def publish(self, topic, payload):
        with self._lock:
            self._f.write(json.dumps({"topic": topic, **payload}, default=str) + "\n")

    def close(self):
        self._f.close()


class KafkaBus(EventBus):
    def __init__(self, bootstrap: str):
        from kafka import KafkaProducer  # pip install aeroflow[kafka]

        self._p = KafkaProducer(
            bootstrap_servers=bootstrap, value_serializer=lambda v: json.dumps(v, default=str).encode()
        )

    def publish(self, topic, payload):
        self._p.send(topic.replace("/", "."), payload)

    def close(self):
        self._p.flush()


class MqttBus(EventBus):
    """MQTT publisher. Solace PubSub+ (free Docker 'Standard' edition) exposes MQTT on 1883."""

    def __init__(self, host: str, port: int):
        import paho.mqtt.client as mqtt  # pip install aeroflow[mqtt]

        self._c = mqtt.Client()
        self._c.connect(host, port)
        self._c.loop_start()

    def publish(self, topic, payload):
        self._c.publish(topic, json.dumps(payload, default=str), qos=1)

    def close(self):
        self._c.loop_stop()
        self._c.disconnect()


def make_bus(ev_cfg: dict) -> EventBus:
    backend = ev_cfg.get("backend", "none")
    try:
        if backend == "jsonl":
            return JsonlBus(ev_cfg["jsonl_path"])
        if backend == "kafka":
            return KafkaBus(ev_cfg["kafka_bootstrap"])
        if backend == "mqtt":
            return MqttBus(ev_cfg["mqtt_host"], ev_cfg["mqtt_port"])
    except Exception as e:  # noqa: BLE001 - never let a broker outage kill the vision pipeline
        log.warning("event backend %s unavailable (%s); events disabled", backend, e)
    return NullBus()
