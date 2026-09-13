import json
import logging
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class SessionStore:
    def __init__(self, redis: Redis | None, ttl_seconds: int = 86400) -> None:
        self.redis = redis
        self.ttl = ttl_seconds
        self._local_sessions: dict[str, list[dict]] = {}

    async def add_message(self, session_id: str, role: str, content: str, extra: dict | None = None) -> None:
        msg = {
            "role": role,
            "content": content,
            "extra": extra or {},
        }
        if self.redis:
            try:
                key = f"chat:session:{session_id}"
                await self.redis.rpush(key, json.dumps(msg))
                await self.redis.expire(key, self.ttl)
                return
            except Exception as exc:
                logger.warning("Failed to store message in Redis: %s", exc)

        if session_id not in self._local_sessions:
            self._local_sessions[session_id] = []
        self._local_sessions[session_id].append(msg)

    async def get_history(self, session_id: str, limit: int = 20) -> list[dict]:
        if self.redis:
            try:
                key = f"chat:session:{session_id}"
                raw = await self.redis.lrange(key, -limit, -1)
                return [json.loads(x) for x in raw]
            except Exception as exc:
                logger.warning("Failed to read messages from Redis: %s", exc)

        return self._local_sessions.get(session_id, [])[-limit:]
