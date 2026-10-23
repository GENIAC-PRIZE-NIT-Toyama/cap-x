"""S3 互換のストレージ（MinIO）を、`requests` だけで呼ぶ最小のクライアント。

backend に boto3 を入れずに済ませる（依存を増やすと、CUDA の無い手元では uv.lock を
作り直せない）。使うのは 5 つだけ: PutObject / GetObject / DeleteObject / ListObjectsV2 /
HeadBucket。署名は AWS Signature Version 4（MinIO も同じ）。パスは path-style
（`http://host:9000/<bucket>/<key>`）。

署名の正しさは、botocore の署名と同じ値になることで確かめている（tests/test_snapshot_s3.py）。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlsplit

_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class S3Error(RuntimeError):
    def __init__(self, message: str, status: int | None = None, code: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass
class S3Object:
    key: str
    last_modified: dt.datetime
    size: int


class S3Client:
    def __init__(
        self,
        endpoint: str,
        access_key_id: str,
        secret_access_key: str,
        region: str = "us-east-1",
        *,
        timeout_s: float = 120.0,
        retries: int = 3,
    ) -> None:
        parts = urlsplit(endpoint)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError(f"S3 の endpoint は http(s)://host:port の形にする: {endpoint!r}")
        self._base = f"{parts.scheme}://{parts.netloc}"
        self._host = parts.netloc
        self._access_key = access_key_id
        self._secret_key = secret_access_key
        self._region = region
        self._timeout_s = timeout_s
        self._retries = retries
        import requests

        self._session = requests.Session()

    # -- 操作 ----------------------------------------------------------------

    def head_bucket(self, bucket: str) -> None:
        self._request("HEAD", bucket)

    def put_object(self, bucket: str, key: str, body: bytes, content_type: str = "application/octet-stream") -> None:
        self._request("PUT", bucket, key, body=body, headers={"content-type": content_type})

    def get_object(self, bucket: str, key: str) -> bytes | None:
        """無ければ None。"""
        try:
            return self._request("GET", bucket, key).content
        except S3Error as exc:
            if exc.status == 404:
                return None
            raise

    def delete_object(self, bucket: str, key: str) -> None:
        self._request("DELETE", bucket, key)

    def list_objects(self, bucket: str, prefix: str) -> list[S3Object]:
        out: list[S3Object] = []
        token: str | None = None
        while True:
            query = {"list-type": "2", "prefix": prefix}
            if token:
                query["continuation-token"] = token
            root = ET.fromstring(self._request("GET", bucket, query=query).content)
            for item in root.iter(f"{_NS}Contents"):
                out.append(
                    S3Object(
                        key=item.findtext(f"{_NS}Key") or "",
                        last_modified=_parse_time(item.findtext(f"{_NS}LastModified") or ""),
                        size=int(item.findtext(f"{_NS}Size") or 0),
                    )
                )
            if (root.findtext(f"{_NS}IsTruncated") or "false").lower() != "true":
                return out
            token = root.findtext(f"{_NS}NextContinuationToken")
            if not token:
                return out

    # -- 署名と送信 ------------------------------------------------------------

    def _request(
        self,
        method: str,
        bucket: str,
        key: str = "",
        *,
        body: bytes = b"",
        query: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        import requests

        path = "/" + quote(bucket, safe="") + ("/" + quote(key, safe="/~") if key else "")
        # クエリは署名と同じ順・同じ符号化のまま送る（requests に組み立て直させない）
        url = self._base + path + (f"?{_canonical_query(query)}" if query else "")
        last: Exception | None = None
        for attempt in range(self._retries):
            signed = self.sign(method, path, query or {}, body, headers or {})
            try:
                response = self._session.request(
                    method, url, data=body or None, headers=signed, timeout=self._timeout_s
                )
            except requests.RequestException as exc:
                last = S3Error(f"S3 に繋げない（{self._base}）: {exc}")
                continue
            if response.status_code < 300:
                return response
            error = _error(response)
            if response.status_code >= 500 and attempt + 1 < self._retries:
                last = error
                continue
            raise error
        assert last is not None
        raise last

    def sign(
        self,
        method: str,
        path: str,
        query: dict[str, str],
        body: bytes,
        headers: dict[str, str],
        now: dt.datetime | None = None,
    ) -> dict[str, str]:
        """SigV4 の署名を付けたヘッダを返す。`path` は URI 符号化済み。"""
        now = now or dt.datetime.now(dt.timezone.utc)  # noqa: UP017 — worker と backend は Python 3.10
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date = now.strftime("%Y%m%d")
        payload_hash = hashlib.sha256(body).hexdigest() if body else _EMPTY_SHA256
        all_headers = {
            **{k.lower(): v.strip() for k, v in headers.items()},
            "host": self._host,
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": amz_date,
        }
        signed_names = sorted(all_headers)
        canonical_request = "\n".join(
            [
                method,
                path,
                _canonical_query(query),
                "".join(f"{name}:{all_headers[name]}\n" for name in signed_names),
                ";".join(signed_names),
                payload_hash,
            ]
        )
        scope = f"{date}/{self._region}/s3/aws4_request"
        string_to_sign = "\n".join(
            ["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical_request.encode()).hexdigest()]
        )
        key = _hmac(f"AWS4{self._secret_key}".encode(), date)
        for part in (self._region, "s3", "aws4_request"):
            key = _hmac(key, part)
        signature = hmac.new(key, string_to_sign.encode(), hashlib.sha256).hexdigest()
        all_headers["authorization"] = (
            f"AWS4-HMAC-SHA256 Credential={self._access_key}/{scope}, "
            f"SignedHeaders={';'.join(signed_names)}, Signature={signature}"
        )
        return all_headers


def _hmac(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode(), hashlib.sha256).digest()


def _canonical_query(query: dict[str, str]) -> str:
    return "&".join(
        f"{quote(k, safe='-_.~')}={quote(v, safe='-_.~')}" for k, v in sorted(query.items())
    )


def _parse_time(text: str) -> dt.datetime:
    try:
        return dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return dt.datetime.fromtimestamp(0, dt.timezone.utc)  # noqa: UP017


def _error(response: Any) -> S3Error:
    code = message = None
    try:
        root = ET.fromstring(response.content)
        code, message = root.findtext("Code"), root.findtext("Message")
    except ET.ParseError:
        pass
    return S3Error(
        f"S3 が {response.status_code} を返した: {code or ''} {message or response.text[:200]}".strip(),
        status=response.status_code,
        code=code,
    )
