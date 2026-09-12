import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import paho.mqtt.client as mqtt

from ..config import Settings
from ..models import DeadLetterEvent, SourceName
from ..topics import INGESTION_DLQ, WIS2_NOTIFICATION


class Wis2MqttSubscriber:
    def __init__(
        self,
        settings: Settings,
        on_message: Callable[[dict[str, Any]], None],
        on_dead_letter: Callable[[DeadLetterEvent], None],
    ) -> None:
        if not settings.wis2_mqtt_host:
            raise RuntimeError("WEATHERGPT_WIS2_MQTT_HOST is required for WIS2 MQTT ingestion")
        self.settings = settings
        self.on_message = on_message
        self.on_dead_letter = on_dead_letter
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"weathergpt-wis2-{uuid4()}")
        if settings.wis2_mqtt_username:
            self.client.username_pw_set(settings.wis2_mqtt_username, settings.wis2_mqtt_password)
        if settings.wis2_mqtt_port == 8883:
            self.client.tls_set()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def run_forever(self) -> None:
        self.client.connect(self.settings.wis2_mqtt_host, self.settings.wis2_mqtt_port, keepalive=60)
        self.client.loop_forever()

    def _on_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        if int(reason_code) == 0:
            client.subscribe(self.settings.wis2_mqtt_topic, qos=1)
            return
        self.on_dead_letter(
            DeadLetterEvent(
                source=SourceName.WIS2,
                topic=INGESTION_DLQ,
                error_type="Wis2ConnectError",
                error_message=f"failed to connect to WIS2 broker: {reason_code}",
                payload={"host": self.settings.wis2_mqtt_host, "topic": self.settings.wis2_mqtt_topic},
            )
        )

    def _on_message(self, client, userdata, message) -> None:
        try:
            payload = json.loads(message.payload.decode("utf-8"))
            self.on_message(
                {
                    "id": str(uuid4()),
                    "source": SourceName.WIS2.value,
                    "topic": WIS2_NOTIFICATION,
                    "mqtt_topic": message.topic,
                    "ingested_at": datetime.now(UTC).isoformat(),
                    "payload": payload,
                }
            )
        except Exception as exc:
            self.on_dead_letter(
                DeadLetterEvent(
                    source=SourceName.WIS2,
                    topic=INGESTION_DLQ,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    payload={"mqtt_topic": message.topic, "payload": message.payload.decode("utf-8", errors="replace")},
                )
            )
