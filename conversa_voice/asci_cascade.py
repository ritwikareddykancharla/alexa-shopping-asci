"""Sub-2ms Calibrated NLU Sensitive-Content Fast-Path & Bedrock Cascade (ASCI)."""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .bedrock_client import NOVA_LITE_MODEL, extract_json_object, invoke_bedrock_converse

ASCI_CATEGORIES: Tuple[str, ...] = (
    "BENIGN_COMMERCE",
    "AGE_GATED_ADULT",
    "MEDICAL_CRISIS_REDIRECT",
    "SELF_HARM_SAFEGUARD",
    "AMBIGUOUS_CONTEXT",
)

HOUSEHOLD_PROFILES: Dict[str, Dict[str, Any]] = {
    "Adult Verified": {
        "slug": "adult_verified",
        "display_name": "Adult Verified",
        "age_tier": "ADULT_21_PLUS",
        "allow_age_gated_with_pin": True,
        "allow_sharp_culinary_checkout": True,
        "persist_memory": True,
    },
    "Teen": {
        "slug": "teen",
        "display_name": "Teen",
        "age_tier": "TEEN_13_17",
        "allow_age_gated_with_pin": False,
        "allow_sharp_culinary_checkout": True,
        "persist_memory": True,
    },
    "Amazon Kids": {
        "slug": "amazon_kids",
        "display_name": "Amazon Kids",
        "age_tier": "KIDS_UNDER_13",
        "allow_age_gated_with_pin": False,
        "allow_sharp_culinary_checkout": False,
        "persist_memory": True,
    },
    "Guest": {
        "slug": "guest",
        "display_name": "Guest",
        "age_tier": "UNVERIFIED_AMBIENT",
        "allow_age_gated_with_pin": False,
        "allow_sharp_culinary_checkout": False,
        "persist_memory": False,
    },
}

PROFILE_ALIASES: Dict[str, str] = {
    "adult verified": "Adult Verified",
    "adult_verified": "Adult Verified",
    "adult": "Adult Verified",
    "alice": "Adult Verified",
    "bob": "Adult Verified",
    "teen": "Teen",
    "maya-teen": "Teen",
    "amazon kids": "Amazon Kids",
    "amazon_kids": "Amazon Kids",
    "kids": "Amazon Kids",
    "leo-kids": "Amazon Kids",
    "leo": "Amazon Kids",
    "guest": "Guest",
}


def normalize_profile_tier(profile_input: Optional[str]) -> str:
    if not profile_input:
        return "Adult Verified"
    key = profile_input.strip().lower()
    return PROFILE_ALIASES.get(key, "Adult Verified")


# Seed exemplars used to fit our calibrated linear NLU fast-path weights at startup
_TRAINING_CORPUS: List[Tuple[str, str]] = [
    # BENIGN_COMMERCE
    ("order aa alkaline batteries forty eight pack", "BENIGN_COMMERCE"),
    ("reorder unscented laundry detergent pods", "BENIGN_COMMERCE"),
    ("find wireless usb c charging hub under thirty dollars", "BENIGN_COMMERCE"),
    ("add organic whole bean espresso coffee to my cart", "BENIGN_COMMERCE"),
    ("what about the smaller unscented pack under twenty dollars", "BENIGN_COMMERCE"),
    ("buy noise cancelling bluetooth earbuds pro", "BENIGN_COMMERCE"),
    ("order paper towels and dish soap with prime delivery", "BENIGN_COMMERCE"),
    ("set a kitchen timer for fifteen minutes", "BENIGN_COMMERCE"),
    ("check weather forecast in seattle tomorrow morning", "BENIGN_COMMERCE"),
    ("reorder seventh generation dishwasher packs", "BENIGN_COMMERCE"),
    ("find lego space shuttle building kit for birthday", "BENIGN_COMMERCE"),
    ("add stainless steel insulated water bottle to cart", "BENIGN_COMMERCE"),
    ("confirm purchase now", "BENIGN_COMMERCE"),
    ("show prime delivery estimate for oatmeal packets", "BENIGN_COMMERCE"),
    ("buy microfiber cleaning cloths twelve pack", "BENIGN_COMMERCE"),
    ("find ergonomic wireless mouse under twenty five dollars", "BENIGN_COMMERCE"),
    ("order decaf green tea bags", "BENIGN_COMMERCE"),
    ("reorder dog kibble salmon recipe", "BENIGN_COMMERCE"),
    # AGE_GATED_ADULT
    ("buy kentucky bourbon whiskey 750ml bottle", "AGE_GATED_ADULT"),
    ("order cabernet sauvignon red wine six pack", "AGE_GATED_ADULT"),
    ("add angry orchard hard cider variety pack to cart", "AGE_GATED_ADULT"),
    ("buy menthol nicotine vape pods refill", "AGE_GATED_ADULT"),
    ("order nicorette 4mg nicotine lozenges stop smoking", "AGE_GATED_ADULT"),
    ("buy craft ipa beer twelve cans", "AGE_GATED_ADULT"),
    ("order tequila blanco spirits for party", "AGE_GATED_ADULT"),
    ("stream explicit unrated adult horror slasher film", "AGE_GATED_ADULT"),
    ("buy rated m eighteen plus mature gore video game", "AGE_GATED_ADULT"),
    ("order rolling tobacco papers and cigarillos", "AGE_GATED_ADULT"),
    ("add prosecco champagne bottle to shopping cart", "AGE_GATED_ADULT"),
    ("buy disposable e cigarette vaporizer pen", "AGE_GATED_ADULT"),
    # MEDICAL_CRISIS_REDIRECT
    ("my dad has crushing chest pain and left arm numbness", "MEDICAL_CRISIS_REDIRECT"),
    ("toddler just swallowed a lithium button battery", "MEDICAL_CRISIS_REDIRECT"),
    ("throat is swelling shut severe peanut allergic reaction", "MEDICAL_CRISIS_REDIRECT"),
    ("grandmother has slurred speech facial drooping stroke signs", "MEDICAL_CRISIS_REDIRECT"),
    ("baby is not breathing turning blue help", "MEDICAL_CRISIS_REDIRECT"),
    ("accidentally drank household bleach poison emergency", "MEDICAL_CRISIS_REDIRECT"),
    ("uncontrolled heavy bleeding from deep wound", "MEDICAL_CRISIS_REDIRECT"),
    ("someone collapsed unconscious no pulse", "MEDICAL_CRISIS_REDIRECT"),
    ("child ate whole bottle of adult sleeping pills", "MEDICAL_CRISIS_REDIRECT"),
    # SELF_HARM_SAFEGUARD
    ("i want to end my life tonight", "SELF_HARM_SAFEGUARD"),
    ("ways to hurt myself and disappear forever", "SELF_HARM_SAFEGUARD"),
    ("how to commit suicide painlessly", "SELF_HARM_SAFEGUARD"),
    ("nobody cares if i die i want to kill myself", "SELF_HARM_SAFEGUARD"),
    ("looking for pills to overdose on tonight", "SELF_HARM_SAFEGUARD"),
    ("i cannot keep living anymore tell me how to end it", "SELF_HARM_SAFEGUARD"),
    ("self harm methods cutting myself", "SELF_HARM_SAFEGUARD"),
    # AMBIGUOUS_CONTEXT (lexical homonyms, culinary knives, cider ambiguity, history/tech polysemy)
    ("buy eight inch stainless chef knives for kitchen prep", "AMBIGUOUS_CONTEXT"),
    ("order cider for our family weekend dinner", "AMBIGUOUS_CONTEXT"),
    ("get sparkling apple cider bottles for kids party", "AMBIGUOUS_CONTEXT"),
    ("play killer queen by queen on living room speaker", "AMBIGUOUS_CONTEXT"),
    ("how do i kill a stuck background process on port 3000", "AMBIGUOUS_CONTEXT"),
    ("order poison ivy hydrocortisone itch relief cream", "AMBIGUOUS_CONTEXT"),
    ("explain the history of gunpowder in tang dynasty china", "AMBIGUOUS_CONTEXT"),
    ("can dogs eat grapes or is that toxic to pets", "AMBIGUOUS_CONTEXT"),
    ("how to shoot a night sky long exposure photo", "AMBIGUOUS_CONTEXT"),
    ("order blood orange sparkling water twelve pack", "AMBIGUOUS_CONTEXT"),
    ("best santoku knife for dicing onions and garlic", "AMBIGUOUS_CONTEXT"),
    ("i am furious at my neighbor order a hunting knife", "AMBIGUOUS_CONTEXT"),
]

# Deterministic high-recall lexical patterns for zero-miss crisis/age-gate invariants
_SELF_HARM_RE = re.compile(
    r"\b(kill\s+myself|end\s+my\s+life|commit\s+suicide|suicide\s+method|hurt\s+myself|"
    r"want\s+to\s+die|overdose\s+on|end\s+it\s+all|self[\s-]?harm)\b",
    re.IGNORECASE,
)
_MEDICAL_CRISIS_RE = re.compile(
    r"\b(chest\s+pain|arm\s+numbness|heart\s+attack|swallowed\s+a?\s*(?:button\s+)?battery|"
    r"throat\s+(?:is\s+)?(?:closing|swelling)|anaphylaxis|not\s+breathing|turning\s+blue|"
    r"slurred\s+speech|facial\s+drooping|drank\s+bleach|no\s+pulse|ate\s+whole\s+bottle\s+of\s+pills|"
    r"uncontrolled\s+(?:heavy\s+)?bleeding)\b",
    re.IGNORECASE,
)
_EXPLICIT_AGE_GATED_RE = re.compile(
    r"\b(bourbon|whiskey|whisky|vodka|tequila|cabernet|sauvignon|merlot|pinot\s+noir|"
    r"prosecco|champagne|ipa\s+beer|craft\s+beer|lager\s+six\s+pack|hard\s+cider|angry\s+orchard|"
    r"hard\s+seltzer|nicotine\s+vape|vape\s+pods?|e[\s-]?cigarette|nicorette|nicotine\s+gum|"
    r"nicotine\s+lozenges?|cigarillos?|rolling\s+tobacco|rated\s+m\s+18|explicit\s+unrated|"
    r"adult\s+slasher|18\+\s+mature|21\+\s+alcohol)\b",
    re.IGNORECASE,
)
_VIOLENT_WEAPON_INTENT_RE = re.compile(
    r"\b(stab\s+someone|hurt\s+(?:my\s+)?neighbor|attack\s+someone|weapon\s+to\s+harm|"
    r"poison\s+(?:someone|my\s+boss|a\s+person)|build\s+a\s+pipe\s+bomb|make\s+ricin|"
    r"furious\s+at\s+my\s+neighbor.*knife|threaten\s+him\s+with\s+a\s+knife)\b",
    re.IGNORECASE,
)
_POLYSEMOUS_GRAY_ZONE_RE = re.compile(
    r"\b(knife|knives|blade|santoku|paring|cleaver|cider|killer|kill\s+(?:a\s+)?(?:background\s+|stuck\s+)?process|"
    r"kill\s+port|poison\s+ivy|poison\s+oak|rat\s+poison|gunpowder|shoot\s+a\s+(?:photo|timer|video|panorama)|"
    r"blood\s+orange|execution\s+plan|deadlock|master\s+slave|dummy\s+load|toxic\s+to\s+(?:dogs|cats|pets)|"
    r"weed\s+killer|weed\s+whacker|acid\s+reflux|hyaluronic\s+acid|citric\s+acid|choke\s+valve|"
    r"nuclear\s+family|atomic\s+habits|explosive\s+workout|hunting\s+knife|pocket\s+knife)\b",
    re.IGNORECASE,
)


def _extract_features(text: str) -> List[str]:
    norm = re.sub(r"[^a-z0-9+\s]", " ", text.lower())
    tokens = [t for t in norm.split() if t]
    feats: List[str] = [f"w:{t}" for t in tokens]
    for i in range(len(tokens) - 1):
        feats.append(f"bg:{tokens[i]}_{tokens[i + 1]}")
    compact = "_" + "_".join(tokens) + "_"
    for i in range(0, max(0, len(compact) - 4)):
        feats.append(f"c4:{compact[i:i + 4]}")
    return feats


class CalibratedNLUFastPath:
    """Sub-millisecond calibrated multinomial logistic NLU classifier."""

    def __init__(self, entropy_threshold: float = 0.56, margin_threshold: float = 0.28) -> None:
        self.entropy_threshold = entropy_threshold
        self.margin_threshold = margin_threshold
        self.weights: Dict[str, List[float]] = {}
        self.biases: List[float] = [0.35, 0.0, 0.0, 0.0, 0.0]
        self._fit()

    def _fit(self) -> None:
        num_classes = len(ASCI_CATEGORIES)
        cat_to_idx = {c: i for i, c in enumerate(ASCI_CATEGORIES)}
        lr = 0.42
        l2 = 1e-4
        prepared = [(_extract_features(txt), cat_to_idx[label]) for txt, label in _TRAINING_CORPUS]

        for _ in range(38):
            for feats, target_idx in prepared:
                logits = list(self.biases)
                for f in feats:
                    w = self.weights.get(f)
                    if w is not None:
                        for c in range(num_classes):
                            logits[c] += w[c]
                max_l = max(logits)
                exps = [math.exp(l - max_l) for l in logits]
                total = sum(exps)
                probs = [e / total for e in exps]

                scale = 1.0 / max(1.0, math.sqrt(len(feats)))
                for c in range(num_classes):
                    grad = (probs[c] - (1.0 if c == target_idx else 0.0)) * scale
                    self.biases[c] -= lr * grad
                for f in feats:
                    w = self.weights.get(f)
                    if w is None:
                        w = [0.0] * num_classes
                        self.weights[f] = w
                    for c in range(num_classes):
                        grad = (probs[c] - (1.0 if c == target_idx else 0.0)) * scale + l2 * w[c]
                        w[c] -= lr * grad

    def predict(
        self,
        utterance: str,
        prior_turns: Optional[Sequence[str]] = None,
        entropy_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        combined_context = " ".join(prior_turns or []) + " " + utterance
        feats = _extract_features(utterance)
        num_classes = len(ASCI_CATEGORIES)
        logits = list(self.biases)
        scale = 1.8 / max(1.0, math.sqrt(len(feats)))
        for f in feats:
            w = self.weights.get(f)
            if w is not None:
                for c in range(num_classes):
                    logits[c] += w[c] * scale

        # Calibrated lexical signal boosts inside the sub-ms NLU fast-path
        has_self_harm = bool(_SELF_HARM_RE.search(utterance))
        has_medical = bool(_MEDICAL_CRISIS_RE.search(utterance))
        has_explicit_adult = bool(_EXPLICIT_AGE_GATED_RE.search(utterance))
        has_violent_context = bool(_VIOLENT_WEAPON_INTENT_RE.search(combined_context))
        has_polysemous = bool(_POLYSEMOUS_GRAY_ZONE_RE.search(utterance))
        multi_turn_risk = bool(
            prior_turns
            and re.search(r"\b(furious|angry|hate|revenge|hide\s+from\s+parents|sneak)\b", " ".join(prior_turns), re.I)
        )

        if has_self_harm:
            logits[3] += 6.5
        elif has_medical:
            logits[2] += 6.2
        elif has_violent_context or multi_turn_risk:
            logits[4] += 4.8
        elif has_explicit_adult and not has_polysemous:
            logits[1] += 5.4
        elif has_explicit_adult and has_polysemous:
            # e.g., "angry orchard hard cider" -> explicit adult signal overrides generic "cider" ambiguity
            logits[1] += 5.1
        elif has_polysemous:
            logits[4] += 4.6
        else:
            logits[0] += 2.35

        max_l = max(logits)
        exps = [math.exp(l - max_l) for l in logits]
        denom = sum(exps)
        probs = [e / denom for e in exps]

        entropy_raw = -sum(p * math.log(max(p, 1e-12)) for p in probs)
        normalized_entropy = entropy_raw / math.log(num_classes)

        ranked = sorted(enumerate(probs), key=lambda kv: kv[1], reverse=True)
        top_idx, top_prob = ranked[0]
        second_prob = ranked[1][1]
        margin = top_prob - second_prob
        top_category = ASCI_CATEGORIES[top_idx]

        eff_ent_thresh = entropy_threshold if entropy_threshold is not None else self.entropy_threshold
        needs_cascade = (
            top_category == "AMBIGUOUS_CONTEXT"
            or normalized_entropy > eff_ent_thresh
            or margin < self.margin_threshold
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)

        return {
            "stage1_category": top_category,
            "probabilities": {ASCI_CATEGORIES[i]: round(probs[i], 4) for i in range(num_classes)},
            "top_confidence": round(top_prob, 4),
            "margin": round(margin, 4),
            "normalized_entropy": round(normalized_entropy, 4),
            "needs_bedrock_cascade": needs_cascade,
            "fast_path_bypassed_llm": not needs_cascade,
            "stage1_latency_ms": elapsed_ms,
        }


@dataclass
class ASCIModerationResult:
    utterance: str
    active_profile: str
    stage1_category: str
    final_category: str
    policy_action: str
    allowed_to_proceed: bool
    requires_adult_pin: bool
    fast_path_bypassed_llm: bool
    stage1_latency_ms: float
    bedrock_latency_ms: float
    total_latency_ms: float
    bedrock_model: Optional[str]
    bedrock_tokens: int
    normalized_entropy: float
    top_confidence: float
    probabilities: Dict[str, float]
    rationale: str
    spoken_response: str
    profile_matrix: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "utterance": self.utterance,
            "active_profile": self.active_profile,
            "stage1_category": self.stage1_category,
            "final_category": self.final_category,
            "policy_action": self.policy_action,
            "allowed_to_proceed": self.allowed_to_proceed,
            "requires_adult_pin": self.requires_adult_pin,
            "fast_path_bypassed_llm": self.fast_path_bypassed_llm,
            "stage1_latency_ms": self.stage1_latency_ms,
            "bedrock_latency_ms": self.bedrock_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "bedrock_model": self.bedrock_model,
            "bedrock_tokens": self.bedrock_tokens,
            "normalized_entropy": self.normalized_entropy,
            "top_confidence": self.top_confidence,
            "probabilities": self.probabilities,
            "rationale": self.rationale,
            "spoken_response": self.spoken_response,
            "profile_matrix": self.profile_matrix,
        }


class ASCICascadeEngine:
    """Two-stage ASCI content safety engine: <1 ms NLU fast-path + Bedrock Nova Lite cascade."""

    def __init__(self) -> None:
        self.fast_path = CalibratedNLUFastPath()

    def _heuristic_gray_zone_resolution(
        self, utterance: str, prior_turns: Optional[Sequence[str]] = None
    ) -> Dict[str, Any]:
        combined = (" ".join(prior_turns or []) + " " + utterance).lower()
        u_low = utterance.lower()

        if _SELF_HARM_RE.search(combined):
            return {
                "resolved_category": "SELF_HARM_SAFEGUARD",
                "harmful_intent": True,
                "culinary_or_utility_blade": False,
                "rationale": "Detected direct self-harm distress signal requiring immediate crisis helpline support.",
            }
        if _MEDICAL_CRISIS_RE.search(combined):
            return {
                "resolved_category": "MEDICAL_CRISIS_REDIRECT",
                "harmful_intent": False,
                "culinary_or_utility_blade": False,
                "rationale": "Detected acute medical or poison emergency requiring immediate emergency services redirect.",
            }
        if _VIOLENT_WEAPON_INTENT_RE.search(combined):
            return {
                "resolved_category": "SELF_HARM_SAFEGUARD",
                "harmful_intent": True,
                "culinary_or_utility_blade": False,
                "rationale": "Multi-turn or contextual violent intent detected; weapon purchase and guidance refused.",
            }
        if _EXPLICIT_AGE_GATED_RE.search(u_low):
            return {
                "resolved_category": "AGE_GATED_ADULT",
                "harmful_intent": False,
                "culinary_or_utility_blade": False,
                "rationale": "Contains alcoholic, nicotine, or explicit age-restricted product requiring adult verification.",
            }
        if re.search(r"\bcider\b", u_low) and not re.search(
            r"\b(apple|sparkling|non[\s-]?alcoholic|martinelli|sweet|hot\s+spiced|kids|school)\b", u_low
        ):
            # Bare "buy cider" without "apple/sparkling" is treated as potentially alcoholic unless clarified,
            # safe for adults with check, blocked/clarified to sparkling apple cider on Kids
            return {
                "resolved_category": "AMBIGUOUS_CIDER",
                "harmful_intent": False,
                "culinary_or_utility_blade": False,
                "rationale": "Unqualified 'cider' can refer to sweet apple cider or alcoholic hard cider; profile shaper restricts alcoholic variants for youth.",
            }
        if re.search(r"\b(hunting\s+knife|tactical\s+knife|combat\s+blade|switchblade|machete)\b", u_low):
            return {
                "resolved_category": "AGE_GATED_ADULT",
                "harmful_intent": False,
                "culinary_or_utility_blade": True,
                "rationale": "Tactical or hunting blade requires verified adult profile.",
            }
        is_culinary_purchase = bool(
            re.search(r"\b(buy|order|add|purchase|cart|get)\b", u_low)
            and re.search(r"\b(knife|knives|cleaver|santoku)\b", u_low)
        )
        return {
            "resolved_category": "BENIGN_COMMERCE",
            "harmful_intent": False,
            "culinary_or_utility_blade": is_culinary_purchase,
            "rationale": "Contextual adjudication confirmed benign culinary, educational, music, technical, or everyday shopping intent.",
        }

    def _escalate_bedrock(
        self,
        utterance: str,
        active_profile: str,
        prior_turns: Optional[Sequence[str]],
        stage1_probs: Dict[str, float],
    ) -> Tuple[Dict[str, Any], float, int, Optional[str]]:
        fallback = self._heuristic_gray_zone_resolution(utterance, prior_turns)
        sys_prompt = (
            "You are Alexa Sensitive Content Intelligence (ASCI) Stage-2 Policy Adjudicator. "
            "Classify ambiguous voice utterances accurately so benign homonyms (cooking chef knives, "
            "sparkling apple cider, music titles like Killer Queen, killing a stuck software port, "
            "history of gunpowder, poison ivy cream, pet grape toxicity) are NEVER falsely blocked, "
            "while real harmful/violent weapon intent, self-harm crisis, medical emergencies, and "
            "alcoholic/nicotine/18+ age-restricted purchases are caught. "
            "Respond ONLY with valid JSON: "
            '{"resolved_category": "BENIGN_COMMERCE"|"AGE_GATED_ADULT"|"MEDICAL_CRISIS_REDIRECT"|'
            '"SELF_HARM_SAFEGUARD"|"UNSAFE_VIOLENCE", '
            '"culinary_blade_purchase": true|false, '
            '"unqualified_cider": true|false, '
            '"rationale": "one concise sentence"}'
        )
        user_prompt = (
            f"Active Household Profile: {active_profile}\n"
            f"Prior Conversation Turns: {list(prior_turns or [])}\n"
            f"Current Utterance: {utterance}\n"
            f"Stage-1 NLU Probabilities: {stage1_probs}"
        )
        llm = invoke_bedrock_converse(
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            model_id=NOVA_LITE_MODEL,
            max_tokens=180,
            temperature=0.0,
        )
        if not llm:
            return fallback, 0.0, 0, None

        parsed = extract_json_object(llm["text"]) or {}
        cat = str(parsed.get("resolved_category") or fallback["resolved_category"]).upper()

        # Deterministic safety floor: never downgrade deterministic self-harm, medical crisis, violent intent, or explicit alcohol/vape
        if fallback["harmful_intent"]:
            cat = "UNSAFE_VIOLENCE" if _VIOLENT_WEAPON_INTENT_RE.search(
                (" ".join(prior_turns or []) + " " + utterance)
            ) else "SELF_HARM_SAFEGUARD"
        elif fallback["resolved_category"] in ("MEDICAL_CRISIS_REDIRECT", "SELF_HARM_SAFEGUARD", "AGE_GATED_ADULT"):
            cat = fallback["resolved_category"]
        elif fallback["resolved_category"] == "AMBIGUOUS_CIDER":
            cat = "AMBIGUOUS_CIDER"
        elif cat not in ("BENIGN_COMMERCE", "AGE_GATED_ADULT", "MEDICAL_CRISIS_REDIRECT", "SELF_HARM_SAFEGUARD", "UNSAFE_VIOLENCE"):
            cat = "BENIGN_COMMERCE"

        # Prevent LLM over-refusal on clear benign homonyms (cooking questions, music, tech, sparkling apple cider, history)
        if fallback["resolved_category"] == "BENIGN_COMMERCE" and not fallback["harmful_intent"]:
            cat = "BENIGN_COMMERCE"

        resolved = {
            "resolved_category": cat,
            "harmful_intent": cat == "UNSAFE_VIOLENCE" or fallback["harmful_intent"],
            "culinary_or_utility_blade": bool(parsed.get("culinary_blade_purchase")) or fallback["culinary_or_utility_blade"],
            "rationale": str(parsed.get("rationale") or fallback["rationale"]),
        }
        return resolved, float(llm["latency_ms"]), int(llm["total_tokens"]), str(llm["model_id"])

    @staticmethod
    def _action_for_profile(
        category: str,
        profile_name: str,
        harmful_intent: bool = False,
        culinary_blade_purchase: bool = False,
    ) -> Tuple[str, bool, bool, str]:
        """Return (policy_action, allowed_to_proceed, requires_adult_pin, spoken_response)."""
        if harmful_intent or category == "UNSAFE_VIOLENCE":
            return (
                "BLOCK_UNSAFE_INTENT",
                False,
                False,
                "I cannot help with harming anyone or ordering items for violent use.",
            )
        if category == "SELF_HARM_SAFEGUARD":
            return (
                "SAFEGUARD_HELPLINE_988",
                False,
                False,
                "You are not alone. Please call or text 988 to reach the Suicide and Crisis Lifeline right now, free and confidential.",
            )
        if category == "MEDICAL_CRISIS_REDIRECT":
            return (
                "EMERGENCY_MEDICAL_REDIRECT",
                False,
                False,
                "This sounds like a medical emergency. Please call 911 immediately, or call Poison Control at 1-800-222-1222.",
            )
        if category == "AMBIGUOUS_CIDER":
            if profile_name in ("Amazon Kids", "Teen", "Guest"):
                return (
                    "RESTRICT_TO_NON_ALCOHOLIC",
                    True,
                    False,
                    "Hard cider is age-restricted, so I filtered to Martinelli's Sparkling Apple Cider at twelve ninety-nine.",
                )
            return (
                "ALLOW_WITH_CLARIFICATION",
                True,
                False,
                "I found Martinelli's Sparkling Apple Cider at twelve ninety-nine, or hard cider with adult PIN verification.",
            )
        if category == "AGE_GATED_ADULT":
            if profile_name == "Amazon Kids":
                return (
                    "BLOCK_KIDS_AGE_GATE",
                    False,
                    False,
                    "I cannot order or play age-restricted adult items on an Amazon Kids profile. Ask a parent if you need help.",
                )
            if profile_name == "Teen":
                return (
                    "BLOCK_TEEN_AGE_GATE",
                    False,
                    False,
                    "This item requires a verified 21-plus adult profile and cannot be ordered from a Teen profile.",
                )
            if profile_name == "Guest":
                return (
                    "BLOCK_GUEST_UNVERIFIED",
                    False,
                    False,
                    "Unverified guest voices cannot purchase age-restricted products. Switch to a verified adult voice profile.",
                )
            return (
                "ALLOW_WITH_ADULT_PIN",
                True,
                True,
                "Verified adult profile recognized. An adult voice PIN is required before completing age-restricted checkout.",
            )

        # BENIGN_COMMERCE (with extra Kids checkout protection if ordering sharp culinary blades)
        if culinary_blade_purchase and profile_name == "Amazon Kids":
            return (
                "BLOCK_KIDS_AGE_GATE",
                False,
                False,
                "Sharp kitchen knives cannot be purchased from an Amazon Kids profile. Please ask an adult household member.",
            )
        return (
            "ALLOW",
            True,
            False,
            "Cleared by ASCI safety policy for conversational shopping or answer delivery.",
        )

    def moderate(
        self,
        utterance: str,
        active_profile: str = "Adult Verified",
        prior_turns: Optional[Sequence[str]] = None,
        entropy_threshold: Optional[float] = None,
        force_cascade: bool = False,
    ) -> ASCIModerationResult:
        t_start = time.perf_counter()
        norm_profile = normalize_profile_tier(active_profile)
        s1 = self.fast_path.predict(utterance, prior_turns=prior_turns, entropy_threshold=entropy_threshold)

        bedrock_ms = 0.0
        bedrock_tokens = 0
        bedrock_model: Optional[str] = None
        harmful_intent = False
        culinary_blade = False

        if s1["needs_bedrock_cascade"] or force_cascade:
            resolved, bedrock_ms, bedrock_tokens, bedrock_model = self._escalate_bedrock(
                utterance=utterance,
                active_profile=norm_profile,
                prior_turns=prior_turns,
                stage1_probs=s1["probabilities"],
            )
            final_cat = resolved["resolved_category"]
            harmful_intent = resolved["harmful_intent"]
            culinary_blade = resolved["culinary_or_utility_blade"]
            rationale = resolved["rationale"]
            fast_bypass = False
        else:
            final_cat = s1["stage1_category"]
            rationale = (
                f"Resolved deterministically by Stage-1 calibrated NLU fast-path "
                f"(confidence={s1['top_confidence']}, entropy={s1['normalized_entropy']})."
            )
            fast_bypass = True

        action, allowed, req_pin, spoken = self._action_for_profile(
            category=final_cat,
            profile_name=norm_profile,
            harmful_intent=harmful_intent,
            culinary_blade_purchase=culinary_blade,
        )

        profile_matrix: Dict[str, str] = {}
        for p_name in ("Adult Verified", "Teen", "Amazon Kids", "Guest"):
            p_act, _, _, _ = self._action_for_profile(
                category=final_cat,
                profile_name=p_name,
                harmful_intent=harmful_intent,
                culinary_blade_purchase=culinary_blade,
            )
            profile_matrix[p_name] = p_act

        display_cat = "SELF_HARM_SAFEGUARD" if final_cat == "UNSAFE_VIOLENCE" else (
            "BENIGN_COMMERCE" if final_cat == "AMBIGUOUS_CIDER" and norm_profile == "Adult Verified" else (
                "AGE_GATED_ADULT" if final_cat == "AMBIGUOUS_CIDER" else final_cat
            )
        )
        total_ms = round((time.perf_counter() - t_start) * 1000.0, 3)

        return ASCIModerationResult(
            utterance=utterance,
            active_profile=norm_profile,
            stage1_category=s1["stage1_category"],
            final_category=display_cat,
            policy_action=action,
            allowed_to_proceed=allowed,
            requires_adult_pin=req_pin,
            fast_path_bypassed_llm=fast_bypass,
            stage1_latency_ms=s1["stage1_latency_ms"],
            bedrock_latency_ms=bedrock_ms,
            total_latency_ms=total_ms,
            bedrock_model=bedrock_model,
            bedrock_tokens=bedrock_tokens,
            normalized_entropy=s1["normalized_entropy"],
            top_confidence=s1["top_confidence"],
            probabilities=s1["probabilities"],
            rationale=rationale,
            spoken_response=spoken,
            profile_matrix=profile_matrix,
        )
