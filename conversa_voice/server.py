"""Local HTTP Server & AWS CloudFront `/api/*` Lambda Function URL Handler for ConversaVoice."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import platform
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import parse_qs, urlparse

from .asci_cascade import ASCICascadeEngine
from .bedrock_client import NOVA_LITE_MODEL, NOVA_MICRO_MODEL
from .household_privacy import SPEAKER_PROFILES, HouseholdPrivacyManager
from .voice_shopping import RETAIL_CATALOG, VoiceShoppingEngine

ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT_DIR / "web"
DOCS_DIR = ROOT_DIR / "docs"
BENCHMARK_JSON_PATH = ROOT_DIR / "benchmarks" / "latest_results.json"

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
}


class ConversaVoiceControlPlane:
    """Thread-safe coordinator uniting Voice Shopping, ASCI Safety Cascade, and Household Privacy."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.asci = ASCICascadeEngine()
        self.shopping = VoiceShoppingEngine(asci_engine=self.asci)
        self.privacy = HouseholdPrivacyManager()

    def load_benchmarks(self) -> Dict[str, Any]:
        if BENCHMARK_JSON_PATH.is_file():
            try:
                return json.loads(BENCHMARK_JSON_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def get_state_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            priv = self.privacy.get_privacy_snapshot()
            sessions_summary = {}
            for p_id in SPEAKER_PROFILES:
                sess = self.shopping.get_session(p_id)
                sessions_summary[p_id] = {
                    "stage": sess.stage,
                    "active_family": sess.active_family,
                    "attributes": sess.attributes,
                    "size_pref": sess.size_pref,
                    "max_price_usd": sess.max_price_usd,
                    "awaiting_confirmation_asin": sess.awaiting_confirmation_asin,
                    "requires_adult_pin": sess.requires_adult_pin,
                    "cart_items": sess.cart_items,
                    "confirmed_orders": sess.confirmed_orders,
                }
            return {
                "service": "conversa-voice",
                "catalog_count": len(RETAIL_CATALOG),
                "catalog_preview": RETAIL_CATALOG[:10],
                "privacy": priv,
                "shopping_sessions": sessions_summary,
                "benchmarks": self.load_benchmarks(),
            }

    def process_voice_turn(self, body: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            utterance = str(body.get("utterance") or "").strip()
            if not utterance:
                raise ValueError("utterance is required")
            profile_id = str(body.get("profile_id") or body.get("profile") or "Alice").strip()
            if profile_id not in SPEAKER_PROFILES:
                # Map tier names cleanly if caller passed "Adult Verified" or "Amazon Kids"
                tier_to_prof = {
                    "adult verified": "Alice",
                    "amazon kids": "Leo-Kids",
                    "kids": "Leo-Kids",
                    "guest": "Guest",
                }
                profile_id = tier_to_prof.get(profile_id.lower(), "Alice")

            spec = SPEAKER_PROFILES[profile_id]
            adult_pin = str(body.get("adult_pin") or "").strip() or None

            # First check cross-profile household isolation boundary
            isolation = self.privacy.evaluate_household_query(profile_id, utterance)
            if isolation["cross_profile_blocked"]:
                from .voice_shopping import compile_ssml_budgeted

                ssml_bundle = compile_ssml_budgeted(isolation["spoken_explanation"])
                return {
                    "profile_id": profile_id,
                    "profile_tier": spec["tier"],
                    "dialogue_stage": "CROSS_PROFILE_ISOLATED",
                    "isolation": isolation,
                    "asr": {
                        "raw_transcript": utterance,
                        "normalized_transcript": utterance,
                        "corrections_applied": [],
                        "phonetic_signature": "",
                        "resolved_concept": None,
                        "asr_confidence": 0.99,
                        "asr_recovery_ms": 0.05,
                    },
                    "asci": {
                        "stage1_category": "BENIGN_COMMERCE",
                        "final_category": "BENIGN_COMMERCE",
                        "policy_action": "HOUSEHOLD_PARTITION_ENFORCED",
                        "allowed_to_proceed": False,
                        "fast_path_bypassed_llm": True,
                        "stage1_latency_ms": isolation["isolation_check_ms"],
                        "bedrock_latency_ms": 0.0,
                        "rationale": "Cross-speaker query blocked by Voice-ID partition filter (0 bytes leaked).",
                    },
                    "ssml_output": ssml_bundle,
                    "echo_show_card": {
                        "card_type": "PRIVACY_BOUNDARY_CARD",
                        "badge": "0 B CROSS-PROFILE LEAK",
                        "headline": f"Isolated Voice Partition ({profile_id})",
                        "subhead": isolation["spoken_explanation"],
                        "asin": None,
                        "price_display": None,
                        "prime_eta": None,
                    },
                    "cart_count": len(self.shopping.get_session(profile_id).cart_items),
                    "orders_count": len(self.shopping.get_session(profile_id).confirmed_orders),
                    "total_turn_ms": isolation["isolation_check_ms"],
                }

            turn_out = self.shopping.handle_turn(
                utterance=utterance,
                profile_id=profile_id,
                profile_tier=spec["tier"],
                adult_pin=adult_pin,
                expected_adult_pin=spec["adult_pin"] or "4821",
                use_bedrock_phrasing=bool(body.get("use_bedrock", True)),
            )
            turn_out["isolation"] = isolation

            # Persist isolated memory only for opted-in recognized household profiles
            persisted = self.privacy.record_turn_memory(
                profile_id=profile_id,
                category=turn_out["dialogue_stage"].lower(),
                summary=f"{utterance} -> {turn_out['echo_show_card']['headline']}",
                is_sensitive=not turn_out["asci"]["allowed_to_proceed"],
            )
            turn_out["memory_persisted"] = persisted
            return turn_out

    def moderate_asci(self, body: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            utterance = str(body.get("utterance") or "").strip()
            if not utterance:
                raise ValueError("utterance is required")
            profile = str(body.get("profile") or "Adult Verified").strip()
            prior_turns = body.get("prior_turns")
            if not isinstance(prior_turns, list):
                prior_turns = []
            entropy_threshold = body.get("entropy_threshold")
            ent_val = float(entropy_threshold) if entropy_threshold is not None else None
            force_cascade = bool(body.get("force_cascade", False))
            res = self.asci.moderate(
                utterance=utterance,
                active_profile=profile,
                prior_turns=[str(x) for x in prior_turns],
                entropy_threshold=ent_val,
                force_cascade=force_cascade,
            )
            return res.to_dict()

    def audit_dp(self, body: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if body.get("probe_utterance"):
                prof = str(body.get("profile_id") or "Guest").strip()
                probe = str(body.get("probe_utterance") or "").strip()
                iso = self.privacy.evaluate_household_query(prof, probe)
                return {
                    "mode": "household_isolation_probe",
                    "isolation": iso,
                    "privacy_snapshot": self.privacy.get_privacy_snapshot(),
                }
            out = self.privacy.audit_and_step_dp(
                additional_rounds=int(body.get("additional_rounds", 4)),
                noise_multiplier=body.get("noise_multiplier"),
                clip_norm=body.get("clip_norm"),
                sample_rate=body.get("sample_rate"),
                max_epsilon=body.get("max_epsilon"),
            )
            out["privacy_snapshot"] = self.privacy.get_privacy_snapshot()
            return out

    def purge_privacy(self, body: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            target = body.get("profile_id")
            target_str = str(target).strip() if target else None
            receipt = self.privacy.purge_memories(target_profile_id=target_str)
            self.shopping.reset_sessions(
                profile_id=target_str if target_str and target_str != "ALL" else None
            )
            receipt["privacy_snapshot"] = self.privacy.get_privacy_snapshot()
            return receipt

    def reset_all(self) -> None:
        with self._lock:
            self.shopping.reset_sessions()
            self.privacy.reset()


CONTROL_PLANE = ConversaVoiceControlPlane()


def route_api_request(
    method: str,
    path: str,
    query: Dict[str, str],
    body: Dict[str, Any],
    request_id: str = "local",
) -> Tuple[int, Dict[str, Any]]:
    """Shared API router for local HTTP server and AWS Lambda Function URL behind CloudFront."""
    try:
        if method == "GET" and path == "/api/health":
            priv = CONTROL_PLANE.privacy.get_privacy_snapshot()
            return 200, {
                "status": "ok",
                "service": "conversa-voice",
                "delivery": os.environ.get("CONVERSAVOICE_DELIVERY", "cloudfront-s3-oac + lambda-api-origin"),
                "aws_region": os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2")),
                "bedrock_models": {
                    "asci_cascade": NOVA_LITE_MODEL,
                    "voice_shopping": NOVA_MICRO_MODEL,
                },
                "python_version": platform.python_version(),
                "catalog_skus": len(RETAIL_CATALOG),
                "active_profiles": list(SPEAKER_PROFILES.keys()),
                "dp_epsilon_spent": priv["dp_accountant"]["epsilon_spent"],
                "dp_epsilon_max": priv["dp_accountant"]["max_epsilon_budget"],
                "request_id": request_id,
            }

        if method == "GET" and path == "/api/state":
            snap = CONTROL_PLANE.get_state_snapshot()
            snap["request_id"] = request_id
            return 200, snap

        if method == "GET" and path == "/api/benchmarks":
            bench = CONTROL_PLANE.load_benchmarks()
            bench["request_id"] = request_id
            return 200, bench

        if method == "POST" and path == "/api/voice/turn":
            out = CONTROL_PLANE.process_voice_turn(body)
            out["request_id"] = request_id
            return 200, out

        if method == "POST" and path == "/api/asci/moderate":
            out = CONTROL_PLANE.moderate_asci(body)
            out["request_id"] = request_id
            return 200, out

        if method == "POST" and path == "/api/privacy/dp-audit":
            out = CONTROL_PLANE.audit_dp(body)
            out["request_id"] = request_id
            return 200, out

        if method == "POST" and path == "/api/privacy/purge":
            out = CONTROL_PLANE.purge_privacy(body)
            out["request_id"] = request_id
            return 200, out

        if method == "POST" and path == "/api/reset":
            CONTROL_PLANE.reset_all()
            return 200, {
                "status": "reset_complete",
                "service": "conversa-voice",
                "request_id": request_id,
            }

        return 404, {"error": f"Unknown API route: {method} {path}"}
    except KeyError as exc:
        return 404, {"error": str(exc)}
    except ValueError as exc:
        return 400, {"error": str(exc)}
    except Exception as exc:
        return 500, {"error": str(exc)}


def lambda_handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """AWS Lambda Function URL handler mounted behind Amazon CloudFront `/api/*` origin."""
    req_ctx = event.get("requestContext") or {}
    method = ((req_ctx.get("http") or {}).get("method") or event.get("httpMethod") or "GET").upper()
    raw_path = event.get("rawPath") or event.get("path") or "/"
    query_params = {
        k: str(v) for k, v in (event.get("queryStringParameters") or {}).items() if v is not None
    }
    req_id = getattr(context, "aws_request_id", "cloudfront-api")

    if method == "OPTIONS":
        return {"statusCode": 204, "headers": CORS_HEADERS, "body": ""}

    if raw_path.startswith("/api/"):
        body: Dict[str, Any] = {}
        raw_body = event.get("body")
        if raw_body:
            if event.get("isBase64Encoded") and isinstance(raw_body, str):
                raw_body = base64.b64decode(raw_body).decode("utf-8")
            body = json.loads(raw_body) if isinstance(raw_body, str) else dict(raw_body)
        status, payload = route_api_request(method, raw_path, query_params, body, request_id=req_id)
        return {
            "statusCode": status,
            "headers": {
                **CORS_HEADERS,
                "Content-Type": "application/json; charset=utf-8",
                "Cache-Control": "no-store",
            },
            "body": json.dumps(payload, default=str),
        }

    target = WEB_DIR / ("index.html" if raw_path in ("/", "") else raw_path.lstrip("/"))
    if not target.is_file() and raw_path.endswith(".svg"):
        target = DOCS_DIR / Path(raw_path).name
    if target.is_file():
        ctype = mimetypes.guess_type(str(target))[0] or "text/html; charset=utf-8"
        return {
            "statusCode": 200,
            "headers": {**CORS_HEADERS, "Content-Type": ctype},
            "body": target.read_text(encoding="utf-8"),
        }
    return {
        "statusCode": 404,
        "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
        "body": json.dumps({"error": "Not found", "path": raw_path}),
    }


class LocalConversaVoiceHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = {k: v[0] for k, v in parse_qs(parsed.query).items() if v}

        if path.startswith("/api/"):
            status, payload = route_api_request("GET", path, query, {}, request_id="local-dev")
            raw = json.dumps(payload, indent=2, default=str).encode("utf-8")
            self.send_response(status)
            for k, v in CORS_HEADERS.items():
                self.send_header(k, v)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        candidate = (WEB_DIR / rel).resolve()
        if not candidate.is_file() and path.endswith(".svg"):
            candidate = (DOCS_DIR / Path(path).name).resolve()
        if not candidate.is_file():
            candidate = WEB_DIR / "index.html"

        data = candidate.read_bytes()
        ctype = mimetypes.guess_type(str(candidate))[0] or "text/html"
        if ctype.startswith("text/") or ctype in ("application/javascript", "image/svg+xml"):
            ctype += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or "0")
        raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        body = json.loads(raw_body) if raw_body.strip() else {}
        status, payload = route_api_request("POST", parsed.path, {}, body, request_id="local-dev")
        raw = json.dumps(payload, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def run_local_server(port: int = 3099) -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", port), LocalConversaVoiceHandler)
    print(f"ConversaVoice running at http://localhost:{port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    port_num = int(os.environ.get("PORT", "3099"))
    run_local_server(port=port_num)
