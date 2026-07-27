from __future__ import annotations

from redis import Redis
from rq import Queue

from backend.config import QUEUE_NAME, REDIS_URL


DEFAULT_JOB_TIMEOUT_SECONDS = 900

redis_conn = Redis.from_url(
    REDIS_URL,
    socket_connect_timeout=2,
    socket_timeout=2,
)
relight_queue = Queue(
    QUEUE_NAME,
    connection=redis_conn,
    default_timeout=DEFAULT_JOB_TIMEOUT_SECONDS,
)
