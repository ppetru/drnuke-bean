"""Tests for IBKR Flex statement download handling."""

from types import SimpleNamespace

import pytest
from ibflex import client

from drnukebean.importer import ibkr


class FakeResponse:
    def __init__(self, content, status_code=200):
        self.content = content.encode()
        self.status_code = status_code

    def __bool__(self):
        return self.status_code < 400


STATEMENT_ACCESS = """<FlexStatementResponse timestamp='16 May, 2026 11:41 AM EDT'>
<Status>Success</Status>
<ReferenceCode>abc123</ReferenceCode>
<Url>https://example.test/get</Url>
</FlexStatementResponse>"""

TEMPORARY_ERROR = """<FlexStatementResponse timestamp='16 May, 2026 11:41 AM EDT'>
<Status>Fail</Status>
<ErrorCode>1001</ErrorCode>
<ErrorMessage>Statement could not be generated at this time. Please try again shortly.</ErrorMessage>
</FlexStatementResponse>"""

FLEX_QUERY = """<FlexQueryResponse queryName='Activity' type='AF'></FlexQueryResponse>"""


def test_submit_request_rejects_http_error_without_retrying_forever(monkeypatch):
    calls = []

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeResponse("<html>not found</html>", status_code=404)

    monkeypatch.setattr(ibkr.client.requests, "get", fake_get)

    with pytest.raises(client.BadResponseError):
        ibkr._submit_flex_request("https://example.test/missing", "token", "query")

    assert len(calls) == 1


def test_download_retries_temporary_send_request_error(monkeypatch):
    responses = [
        FakeResponse(TEMPORARY_ERROR),
        FakeResponse(STATEMENT_ACCESS),
        FakeResponse(FLEX_QUERY),
    ]

    def fake_get(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(ibkr.client.requests, "get", fake_get)
    monkeypatch.setattr(ibkr.time, "sleep", lambda seconds: None)

    content = ibkr._download_flex_statement("token", "query", max_tries=3)

    assert b"FlexQueryResponse" in content
    assert responses == []


def test_download_uses_current_send_request_endpoint(monkeypatch):
    requested_urls = []

    def fake_get(url, **kwargs):
        requested_urls.append(url)
        if len(requested_urls) == 1:
            return FakeResponse(STATEMENT_ACCESS)
        return FakeResponse(FLEX_QUERY)

    monkeypatch.setattr(ibkr.client.requests, "get", fake_get)

    ibkr._download_flex_statement("token", "query", max_tries=1)

    assert requested_urls[0] == ibkr.IBKR_SEND_REQUEST_URL


def test_download_passes_period_override_to_send_request(monkeypatch):
    request_params = []

    def fake_get(url, **kwargs):
        request_params.append(kwargs["params"])
        if len(request_params) == 1:
            return FakeResponse(STATEMENT_ACCESS)
        return FakeResponse(FLEX_QUERY)

    monkeypatch.setattr(ibkr.client.requests, "get", fake_get)

    ibkr._download_flex_statement("token", "query", max_tries=1, period_days=60)

    assert request_params[0]["p"] == "60"
    assert "p" not in request_params[1]
