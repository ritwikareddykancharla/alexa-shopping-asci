"""Amazon Bedrock Converse client supporting Nova Lite and Nova Micro."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, Optional

NOVA_LITE_MODEL = os.environ.get("CONVERSAVOICE_NOVA_LITE_MODEL", "us.amazon.nova-lite-v1:0")
NOVA_MICRO_MODEL = os.environ.get("CONVERSAVOICE_NOVA_MICRO_MODEL", "us.amazon.nova-micro-v1:0")
DEFAULT_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))

_CLIENT_LOCK = threading.Lock()
_CACHED_CLIENTS: Dict[str, Any] = {}


def _get_bedrock_client(region: str) -> Any:
    key = f"{region}:{os.environ.get('AWS_PROFILE', '')}:{bool(os.environ.get('AWS_LAMBDA_FUNCTION_NAME'))}"
    with _CLIENT_LOCK:
        if key in _CACHED_CLIENTS:
            return _CACHED_CLIENTS[key]
        import boto3  # type: ignore

        if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
            session = boto3.Session(region_name=region)
        else:
            profile = os.environ.get("CONVERSAVOICE_BEDROCK_PROFILE", os.environ.get("AWS_PROFILE", "portfolio"))
            try:
                session = boto3.Session(profile_name=profile, region_name=region)
            except Exception:
                session = boto3.Session(region_name=region)
        client = session.client("bedrock-runtime", region_name=region)
        _CACHED_CLIENTS[key] = client
        return client


def invoke_bedrock_converse(
    system_prompt: str,
    user_prompt: str,
    model_id: str = NOVA_LITE_MODEL,
    max_tokens: int = 280,
    temperature: float = 0.1,
) -> Optional[Dict[str, Any]]:
    """Invoke Amazon Bedrock Converse API and capture server latency and token counts."""
    if os.environ.get("CONVERSAVOICE_OFFLINE_LLM") == "1":
        return None

    region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", DEFAULT_REGION))
    t0 = time.perf_counter()

    try:
        client = _get_bedrock_client(region)
        resp = client.converse(
            modelId=model_id,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
        )
        round_trip_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        server_ms = float(resp.get("metrics", {}).get("latencyMs") or round_trip_ms)
        usage = resp.get("usage", {})
        content_blocks = resp.get("output", {}).get("message", {}).get("content", [])
        text = ""
        for blk in content_blocks:
            if "text" in blk:
                text += blk["text"]
        return {
            "text": text.strip(),
            "engine": f"aws-bedrock ({model_id})",
            "model_id": model_id,
            "latency_ms": round(server_ms, 2),
            "round_trip_ms": round_trip_ms,
            "input_tokens": int(usage.get("inputTokens", 0)),
            "output_tokens": int(usage.get("outputTokens", 0)),
            "total_tokens": int(usage.get("totalTokens", 0)),
        }
    except Exception:
        pass

    aws_bin = shutil.which("aws")
    if not aws_bin or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        return None

    env = dict(os.environ)
    env.pop("AWS_ACCESS_KEY_ID", None)
    env.pop("AWS_SECRET_ACCESS_KEY", None)
    env.pop("AWS_SESSION_TOKEN", None)
    env["AWS_PROFILE"] = os.environ.get("CONVERSAVOICE_BEDROCK_PROFILE", os.environ.get("AWS_PROFILE", "portfolio"))
    env["AWS_DEFAULT_REGION"] = region

    cmd = [
        aws_bin,
        "bedrock-runtime",
        "converse",
        "--region",
        region,
        "--model-id",
        model_id,
        "--system",
        json.dumps([{"text": system_prompt}]),
        "--messages",
        json.dumps([{"role": "user", "content": [{"text": user_prompt}]}]),
        "--inference-config",
        json.dumps({"maxTokens": max_tokens, "temperature": temperature}),
        "--output",
        "json",
    ]
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=15.0)
        round_trip_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            server_ms = float(data.get("metrics", {}).get("latencyMs") or round_trip_ms)
            usage = data.get("usage", {})
            text = data["output"]["message"]["content"][0]["text"]
            return {
                "text": text.strip(),
                "engine": f"aws-bedrock ({model_id})",
                "model_id": model_id,
                "latency_ms": round(server_ms, 2),
                "round_trip_ms": round_trip_ms,
                "input_tokens": int(usage.get("inputTokens", 0)),
                "output_tokens": int(usage.get("outputTokens", 0)),
                "total_tokens": int(usage.get("totalTokens", 0)),
            }
    except Exception:
        pass

    return None


def extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object cleanly from an LLM response."""
    if not raw_text:
        return None
    cleaned = raw_text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    else:
        brace = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if brace:
            cleaned = brace.group(1)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None
