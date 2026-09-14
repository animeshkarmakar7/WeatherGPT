import io

from minio import Minio


class MinioDocumentStore:
    def __init__(self, endpoint: str = "localhost:9000", access_key: str = "weathergpt", secret_key: str = "weathergpt-secret", secure: bool = False, bucket_name: str = "weathergpt-documents") -> None:
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure
        self.bucket_name = bucket_name
        self._client: Minio | None = None

    def _get_client(self) -> Minio:
        if self._client is None:
            self._client = Minio(self.endpoint, access_key=self.access_key, secret_key=self.secret_key, secure=self.secure)
            if not self._client.bucket_exists(self.bucket_name):
                self._client.make_bucket(self.bucket_name)
        return self._client

    def health(self) -> bool:
        client = self._get_client()
        return client.bucket_exists(self.bucket_name)

    def upload_document(self, object_name: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        client = self._get_client()
        client.put_object(self.bucket_name, object_name, io.BytesIO(data), length=len(data), content_type=content_type)
        return f"minio://{self.bucket_name}/{object_name}"

    def get_document(self, object_name: str) -> bytes | None:
        client = self._get_client()
        response = client.get_object(self.bucket_name, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()
