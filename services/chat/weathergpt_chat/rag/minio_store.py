import io
import logging
from minio import Minio

logger = logging.getLogger(__name__)


class MinioDocumentStore:
    def __init__(
        self,
        endpoint: str = "localhost:9000",
        access_key: str = "weathergpt",
        secret_key: str = "weathergpt-secret",
        secure: bool = False,
        bucket_name: str = "weathergpt-documents",
    ) -> None:
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure
        self.bucket_name = bucket_name
        self._client: Minio | None = None
        self._memory_store: dict[str, bytes] = {}

    def _get_client(self) -> Minio | None:
        if self._client is None:
            try:
                self._client = Minio(
                    self.endpoint,
                    access_key=self.access_key,
                    secret_key=self.secret_key,
                    secure=self.secure,
                )
                if not self._client.bucket_exists(self.bucket_name):
                    self._client.make_bucket(self.bucket_name)
            except Exception as e:
                logger.warning("MinIO server unavailable (%s), operating with memory fallback", e)
                self._client = None
        return self._client

    def upload_document(self, object_name: str, data: bytes, content_type: str = "text/plain") -> str:
        client = self._get_client()
        if client:
            try:
                client.put_object(
                    self.bucket_name,
                    object_name,
                    io.BytesIO(data),
                    length=len(data),
                    content_type=content_type,
                )
                return f"minio://{self.bucket_name}/{object_name}"
            except Exception as e:
                logger.warning("MinIO put failed (%s), storing in local fallback", e)
        self._memory_store[object_name] = data
        return f"memory://{self.bucket_name}/{object_name}"

    def get_document(self, object_name: str) -> bytes | None:
        client = self._get_client()
        if client:
            try:
                resp = client.get_object(self.bucket_name, object_name)
                return resp.read()
            except Exception:
                pass
        return self._memory_store.get(object_name)
