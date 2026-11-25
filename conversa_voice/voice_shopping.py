"""Multi-Turn Voice Shopping State Machine, Phonetic ASR Recovery & <=40-Word SSML Compiler."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .asci_cascade import ASCICascadeEngine, normalize_profile_tier
from .bedrock_client import NOVA_MICRO_MODEL, extract_json_object, invoke_bedrock_converse

SPOKEN_WORD_BUDGET = 40

# Rich Alexa Retail Catalog (exact prices, Prime ETAs, size/scent attributes, age restrictions)
RETAIL_CATALOG: List[Dict[str, Any]] = [
    {
        "asin": "B0CV-USBC-01",
        "title": "Anker 7-in-1 USB-C Hub (100W PD, 4K HDMI)",
        "short_name": "Anker 7-in-1 USB-C Hub",
        "canonical_concept": "USB-C Hub",
        "family": "usb_c_hub",
        "brand": "Anker",
        "price_usd": 29.99,
        "prime_eta": "Prime Overnight by 7 AM",
        "rating": 4.7,
        "size_tier": "standard",
        "attributes": ["usb-c", "7-in-1", "4k hdmi", "100w"],
        "age_restricted": False,
        "keywords": ["usb-c hub", "usb c hub", "anker hub", "dongle", "adapter", "charging block"],
    },
    {
        "asin": "B0CV-USBC-02",
        "title": "Anker Compact 4-Port Slim USB-C Hub",
        "short_name": "Anker Compact 4-Port USB-C Hub",
        "canonical_concept": "USB-C Hub",
        "family": "usb_c_hub",
        "brand": "Anker",
        "price_usd": 16.99,
        "prime_eta": "Prime Tomorrow",
        "rating": 4.6,
        "size_tier": "smaller",
        "attributes": ["usb-c", "compact", "smaller", "slim"],
        "age_restricted": False,
        "keywords": ["usb-c hub", "smaller usb c hub", "compact hub", "slim usb hub"],
    },
    {
        "asin": "B0CV-BUDS-03",
        "title": "SoundCore Earbuds Pro 3 (Adaptive ANC, Spatial Audio)",
        "short_name": "SoundCore Earbuds Pro 3",
        "canonical_concept": "Earbuds Pro 3",
        "family": "wireless_earbuds",
        "brand": "SoundCore",
        "price_usd": 149.99,
        "prime_eta": "Prime Same-Day Today",
        "rating": 4.8,
        "size_tier": "standard",
        "attributes": ["wireless", "noise cancelling", "pro 3", "spatial"],
        "age_restricted": False,
        "keywords": ["earbuds pro 3", "airpods pro 3", "wireless earbuds", "noise cancelling earbuds"],
    },
    {
        "asin": "B0CV-BUDS-04",
        "title": "SoundCore Earbuds Mini 3 Compact True Wireless",
        "short_name": "SoundCore Earbuds Mini 3",
        "canonical_concept": "Earbuds Pro 3",
        "family": "wireless_earbuds",
        "brand": "SoundCore",
        "price_usd": 39.99,
        "prime_eta": "Prime Tomorrow",
        "rating": 4.5,
        "size_tier": "smaller",
        "attributes": ["wireless", "smaller", "compact", "mini"],
        "age_restricted": False,
        "keywords": ["earbuds mini", "smaller earbuds", "cheaper earbuds", "compact earbuds"],
    },
    {
        "asin": "B0CV-TIDE-01",
        "title": "Tide Pods Free & Gentle Unscented Laundry Pacs (81-Count Tub)",
        "short_name": "Tide Pods Free & Gentle 81-Count",
        "canonical_concept": "Tide Pods Unscented",
        "family": "laundry_pods",
        "brand": "Tide",
        "price_usd": 24.99,
        "prime_eta": "Prime Tomorrow by 10 AM",
        "rating": 4.8,
        "size_tier": "larger",
        "attributes": ["unscented", "free & gentle", "hypoallergenic", "81-count"],
        "age_restricted": False,
        "keywords": ["tide pods unscented", "tide pods", "laundry pods", "free and gentle", "detergent pods"],
    },
    {
        "asin": "B0CV-TIDE-02",
        "title": "Tide Pods Free & Gentle Unscented Laundry Pacs (42-Count Bag)",
        "short_name": "Tide Pods Free & Gentle 42-Count",
        "canonical_concept": "Tide Pods Unscented",
        "family": "laundry_pods",
        "brand": "Tide",
        "price_usd": 15.49,
        "prime_eta": "Prime Overnight by 8 AM",
        "rating": 4.8,
        "size_tier": "smaller",
        "attributes": ["unscented", "smaller", "free & gentle", "hypoallergenic", "42-count"],
        "age_restricted": False,
        "keywords": ["tide pods unscented", "smaller tide pods", "unscented laundry pods", "42 count"],
    },
    {
        "asin": "B0CV-DISH-01",
        "title": "Seventh Generation Free & Clear Unscented Dishwasher Packs (45-Count)",
        "short_name": "Seventh Generation Dishwasher Packs",
        "canonical_concept": "Organic Dishwasher Pods",
        "family": "dishwasher_pods",
        "brand": "Seventh Generation",
        "price_usd": 16.99,
        "prime_eta": "Prime Tomorrow",
        "rating": 4.7,
        "size_tier": "smaller",
        "attributes": ["unscented", "organic", "plant-based", "free & clear"],
        "age_restricted": False,
        "keywords": ["dishwasher pods", "seventh generation", "dish packs", "organic dishwasher"],
    },
    {
        "asin": "B0CV-CIDR-01",
        "title": "Martinelli's Gold Medal Sparkling Apple Cider (4-Pack, Non-Alcoholic)",
        "short_name": "Martinelli's Sparkling Apple Cider",
        "canonical_concept": "Sparkling Apple Cider",
        "family": "apple_cider",
        "brand": "Martinelli's",
        "price_usd": 12.99,
        "prime_eta": "Prime Fresh Today",
        "rating": 4.9,
        "size_tier": "smaller",
        "attributes": ["non-alcoholic", "sparkling", "apple", "kid-safe"],
        "age_restricted": False,
        "keywords": ["sparkling apple cider", "apple cider", "sweet cider", "martinelli cider"],
    },
    {
        "asin": "B0CV-CIDR-21",
        "title": "Angry Orchard Crisp Apple Hard Cider (6-Pack Bottles, 5.0% ABV)",
        "short_name": "Angry Orchard Hard Cider 6-Pack",
        "canonical_concept": "Hard Cider (21+)",
        "family": "hard_cider",
        "brand": "Angry Orchard",
        "price_usd": 14.99,
        "prime_eta": "Prime 2-Hour Scheduled (21+ ID Check)",
        "rating": 4.6,
        "size_tier": "standard",
        "attributes": ["alcoholic", "hard cider", "21+"],
        "age_restricted": True,
        "keywords": ["hard cider", "angry orchard", "alcoholic cider"],
    },
    {
        "asin": "B0CV-KNIF-01",
        "title": "Victorinox Fibrox Pro 8-Inch Stainless Chef's Knife",
        "short_name": "Victorinox 8-Inch Chef's Knife",
        "canonical_concept": "Chef Knives",
        "family": "chef_knives",
        "brand": "Victorinox",
        "price_usd": 44.95,
        "prime_eta": "Prime Tomorrow",
        "rating": 4.9,
        "size_tier": "standard",
        "attributes": ["culinary", "8-inch", "stainless steel", "kitchen"],
        "age_restricted": False,
        "keywords": ["chef knives", "chef knife", "kitchen knife", "santoku knife", "victorinox"],
    },
    {
        "asin": "B0CV-KNIF-02",
        "title": "Victorinox 4-Inch Swiss Classic Kitchen Paring Knife",
        "short_name": "Victorinox 4-Inch Paring Knife",
        "canonical_concept": "Chef Knives",
        "family": "chef_knives",
        "brand": "Victorinox",
        "price_usd": 11.95,
        "prime_eta": "Prime Tomorrow",
        "rating": 4.8,
        "size_tier": "smaller",
        "attributes": ["culinary", "smaller", "4-inch", "paring"],
        "age_restricted": False,
        "keywords": ["smaller knife", "paring knife", "small kitchen knife"],
    },
    {
        "asin": "B0CV-BATT-01",
        "title": "Amazon Basics AA High-Performance Alkaline Batteries (48-Pack)",
        "short_name": "Amazon Basics AA Batteries 48-Pack",
        "canonical_concept": "AA Batteries",
        "family": "batteries",
        "brand": "Amazon Basics",
        "price_usd": 14.49,
        "prime_eta": "Prime Same-Day Today",
        "rating": 4.8,
        "size_tier": "larger",
        "attributes": ["aa", "alkaline", "48-pack"],
        "age_restricted": False,
        "keywords": ["aa batteries", "double a batteries", "alkaline batteries"],
    },
    {
        "asin": "B0CV-BATT-02",
        "title": "Amazon Basics AA Alkaline Batteries (20-Pack)",
        "short_name": "Amazon Basics AA Batteries 20-Pack",
        "canonical_concept": "AA Batteries",
        "family": "batteries",
        "brand": "Amazon Basics",
        "price_usd": 7.99,
        "prime_eta": "Prime Tomorrow",
        "rating": 4.8,
        "size_tier": "smaller",
        "attributes": ["aa", "smaller", "20-pack"],
        "age_restricted": False,
        "keywords": ["smaller aa batteries", "20 pack batteries", "aa batteries"],
    },
    {
        "asin": "B0CV-COFF-01",
        "title": "Lavazza Super Crema Whole Bean Espresso Coffee (2.2 lb Bag)",
        "short_name": "Lavazza Super Crema Espresso 2.2 lb",
        "canonical_concept": "Espresso Beans",
        "family": "coffee_beans",
        "brand": "Lavazza",
        "price_usd": 21.99,
        "prime_eta": "Prime Overnight",
        "rating": 4.7,
        "size_tier": "larger",
        "attributes": ["whole bean", "espresso", "medium roast"],
        "age_restricted": False,
        "keywords": ["espresso beans", "coffee beans", "lavazza coffee", "whole bean coffee"],
    },
    {
        "asin": "B0CV-COFF-02",
        "title": "Lavazza Organic Tierra Unscented Roast Coffee Beans (12 oz Bag)",
        "short_name": "Lavazza Organic Coffee 12 oz",
        "canonical_concept": "Espresso Beans",
        "family": "coffee_beans",
        "brand": "Lavazza",
        "price_usd": 11.49,
        "prime_eta": "Prime Tomorrow",
        "rating": 4.6,
        "size_tier": "smaller",
        "attributes": ["organic", "smaller", "12 oz", "whole bean"],
        "age_restricted": False,
        "keywords": ["smaller coffee beans", "organic coffee", "12 oz espresso"],
    },
    {
        "asin": "B0CV-IVY-01",
        "title": "Cortizone-10 Maximum Strength Poison Ivy Soothing Cream (2 oz)",
        "short_name": "Cortizone-10 Poison Ivy Relief Cream",
        "canonical_concept": "Poison Ivy Relief Cream",
        "family": "otc_skin_care",
        "brand": "Cortizone-10",
        "price_usd": 9.49,
        "prime_eta": "Prime Same-Day Pharmacy",
        "rating": 4.7,
        "size_tier": "smaller",
        "attributes": ["unscented", "hydrocortisone", "itch relief"],
        "age_restricted": False,
        "keywords": ["poison ivy cream", "poison ivy relief", "itch cream", "hydrocortisone"],
    },
    {
        "asin": "B0CV-BRBN-21",
        "title": "Woodford Reserve Kentucky Straight Bourbon Whiskey (750ml)",
        "short_name": "Woodford Reserve Bourbon 750ml",
        "canonical_concept": "Bourbon Whiskey (21+)",
        "family": "spirits_21",
        "brand": "Woodford Reserve",
        "price_usd": 38.99,
        "prime_eta": "Prime Scheduled (21+ Adult Signature)",
        "rating": 4.8,
        "size_tier": "standard",
        "attributes": ["bourbon", "whiskey", "21+"],
        "age_restricted": True,
        "keywords": ["bourbon", "whiskey", "woodford reserve", "kentucky bourbon"],
    },
    {
        "asin": "B0CV-LEGO-01",
        "title": "LEGO Creator 3-in-1 Space Shuttle Adventure Kit",
        "short_name": "LEGO Space Shuttle Kit",
        "canonical_concept": "LEGO Space Shuttle",
        "family": "kids_toys",
        "brand": "LEGO",
        "price_usd": 29.99,
        "prime_eta": "Prime Tomorrow",
        "rating": 4.9,
        "size_tier": "standard",
        "attributes": ["kid-safe", "ages 8+", "stem"],
        "age_restricted": False,
        "keywords": ["lego", "space shuttle", "building kit", "kids toy"],
    },
]

# Multi-word & token-level noisy ASR homophone normalizations
ASR_PHRASE_CORRECTIONS: List[Tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"\busb\s+see\s+hub\b", re.I), "usb-c hub", "usb see hub -> USB-C Hub"),
    (re.compile(r"\busb\s+sea\s+hub\b", re.I), "usb-c hub", "usb sea hub -> USB-C Hub"),
    (re.compile(r"\banchor\s+usb\s+see\b", re.I), "anker usb-c hub", "anchor usb see -> Anker USB-C Hub"),
    (re.compile(r"\bair\s*pods?\s+pro\s+tree\b", re.I), "earbuds pro 3", "air pods pro tree -> Earbuds Pro 3"),
    (re.compile(r"\bear\s*pods?\s+pro\s+tree\b", re.I), "earbuds pro 3", "ear pods pro tree -> Earbuds Pro 3"),
    (re.compile(r"\bear\s*buds?\s+pro\s+free\b", re.I), "earbuds pro 3", "ear buds pro free -> Earbuds Pro 3"),
    (re.compile(r"\btied\s+pods?\b", re.I), "tide pods", "tied pods -> Tide Pods"),
    (re.compile(r"\btime\s+pods?\s+unscented\b", re.I), "tide pods unscented", "time pods -> Tide Pods"),
    (re.compile(r"\bfree\s+in\s+gentle\b", re.I), "free & gentle", "free in gentle -> Free & Gentle"),
    (re.compile(r"\bapple\s+sider\b", re.I), "apple cider", "apple sider -> Apple Cider"),
    (re.compile(r"\bhard\s+sider\b", re.I), "hard cider", "hard sider -> Hard Cider"),
    (re.compile(r"\bshef\s+knives\b", re.I), "chef knives", "shef knives -> Chef Knives"),
    (re.compile(r"\bchef\s+nighs\b", re.I), "chef knives", "chef nighs -> Chef Knives"),
    (re.compile(r"\bdouble\s+eh\s+batter\s*ease\b", re.I), "aa batteries", "double eh batter ease -> AA Batteries"),
    (re.compile(r"\baa\s+batter\s*ease\b", re.I), "aa batteries", "aa batter ease -> AA Batteries"),
    (re.compile(r"\bexpresso\s+beans\b", re.I), "espresso beans", "expresso beans -> Espresso Beans"),
    (re.compile(r"\blava\s+za\b", re.I), "lavazza", "lava za -> Lavazza"),
    (re.compile(r"\bseven\s+generation\b", re.I), "seventh generation", "seven generation -> Seventh Generation"),
    (re.compile(r"\bwood\s+ford\s+bourbon\b", re.I), "woodford bourbon", "wood ford bourbon -> Woodford Bourbon"),
    (re.compile(r"\blet\s+go\s+space\s+shuttle\b", re.I), "lego space shuttle", "let go space shuttle -> LEGO Space Shuttle"),
]

SPOKEN_NUMBER_WORDS: Dict[str, float] = {
    "ten": 10.0,
    "twelve": 12.0,
    "fifteen": 15.0,
    "eighteen": 18.0,
    "twenty": 20.0,
    "twenty five": 25.0,
    "twenty-five": 25.0,
    "thirty": 30.0,
    "thirty five": 35.0,
    "forty": 40.0,
    "forty five": 45.0,
    "fifty": 50.0,
    "hundred": 100.0,
}


def phonetic_fingerprint(word: str) -> str:
    """Compact Metaphone-inspired phonetic code for ASR token alignment."""
    w = re.sub(r"[^a-z0-9]", "", word.lower())
    if not w:
        return ""
    if w.isdigit():
        return w
    w = re.sub(r"^kn", "n", w)
    w = re.sub(r"^ph", "f", w)
    w = re.sub(r"^wr", "r", w)
    w = w.replace("sh", "x").replace("ch", "x").replace("th", "0").replace("ck", "k")
    w = w.replace("ee", "i").replace("ea", "i").replace("oo", "u")
    first = w[0]
    rest = re.sub(r"[aeiouyhw]", "", w[1:])
    dedup = [first]
    for ch in rest:
        if ch != dedup[-1]:
            dedup.append(ch)
    return "".join(dedup)[:6]


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


class PhoneticASRResolver:
    """Recovers canonical retail entities from noisy/homophone ASR transcripts."""

    def resolve(self, raw_utterance: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        normalized = raw_utterance.strip()
        corrections: List[str] = []

        for pattern, replacement, label in ASR_PHRASE_CORRECTIONS:
            if pattern.search(normalized):
                normalized = pattern.sub(replacement, normalized)
                corrections.append(label)

        tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", normalized.lower())
        phonetic_codes = [phonetic_fingerprint(tok) for tok in tokens[:10]]

        # Score against catalog canonical concepts via exact + phonetic + edit similarity
        best_concept: Optional[str] = None
        best_family: Optional[str] = None
        best_score = 0.0

        norm_lower = normalized.lower()
        for item in RETAIL_CATALOG:
            score = 0.0
            for kw in item["keywords"]:
                if kw in norm_lower:
                    score = max(score, 0.94 + 0.01 * len(kw.split()))
                else:
                    kw_tokens = kw.split()
                    kw_phon = [phonetic_fingerprint(k) for k in kw_tokens]
                    matches = 0
                    for kp in kw_phon:
                        if kp and any(
                            kp == tp or (_levenshtein(kp, tp) <= 1 and len(kp) >= 3)
                            for tp in phonetic_codes
                        ):
                            matches += 1
                    if kw_phon and matches > 0:
                        score = max(score, 0.72 * (matches / len(kw_phon)))
            if score > best_score:
                best_score = score
                best_concept = item["canonical_concept"]
                best_family = item["family"]

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)
        return {
            "raw_transcript": raw_utterance,
            "normalized_transcript": normalized,
            "corrections_applied": corrections,
            "phonetic_signature": " ".join(c for c in phonetic_codes if c),
            "resolved_concept": best_concept if best_score >= 0.62 else None,
            "resolved_family": best_family if best_score >= 0.62 else None,
            "asr_confidence": round(min(0.99, best_score if best_score > 0 else 0.88), 3),
            "asr_recovery_ms": elapsed_ms,
        }


def extract_price_ceiling(text: str) -> Optional[float]:
    low = text.lower()
    m_num = re.search(r"\b(?:under|below|less\s+than|max|up\s+to)\s*\$?\s*(\d+(?:\.\d{1,2})?)\b", low)
    if m_num:
        return float(m_num.group(1))
    for word_phrase, val in sorted(SPOKEN_NUMBER_WORDS.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b(?:under|below|less\s+than|up\s+to)\s+{re.escape(word_phrase)}\b", low):
            return val
    return None


def count_spoken_words(ssml_or_text: str) -> int:
    """Count spoken words after stripping SSML XML tags."""
    stripped = re.sub(r"<[^>]+>", " ", ssml_or_text)
    words = [w for w in re.split(r"\s+", stripped.strip()) if re.search(r"[A-Za-z0-9]", w)]
    return len(words)


def compile_ssml_budgeted(spoken_text: str, max_words: int = SPOKEN_WORD_BUDGET, rate: str = "102%") -> Dict[str, Any]:
    """Deterministically enforce Alexa's <= 40-word spoken brevity rule and wrap in valid SSML."""
    clean = re.sub(r"<[^>]+>", " ", spoken_text)
    clean = clean.replace("—", ", ").replace("–", ", ")
    clean = re.sub(r"\[[^\]]*\]", "", clean)
    clean = re.sub(r"\*+", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    words = clean.split()
    truncated = False
    if len(words) > max_words:
        truncated = True
        clipped = words[:max_words]
        joined = " ".join(clipped)
        # Prefer ending cleanly on a sentence boundary if available after word 18
        sentences = re.findall(r"[^.!?]+[.!?]", joined)
        if sentences and len(" ".join(sentences).split()) >= 16:
            clean = " ".join(s.strip() for s in sentences)
        else:
            clean = joined.rstrip(",;:") + "."
    elif clean and clean[-1] not in ".!?":
        clean += "."

    final_word_count = count_spoken_words(clean)
    escaped_xml = (
        clean.replace("&", "and")
        .replace("<", "")
        .replace(">", "")
    )
    final_word_count = count_spoken_words(escaped_xml)
    if final_word_count > max_words:
        escaped_xml = " ".join(escaped_xml.split()[:max_words]).rstrip(",;:") + "."
        final_word_count = count_spoken_words(escaped_xml)

    ssml = f'<speak><prosody rate="{rate}">{escaped_xml}</prosody><break time="250ms"/></speak>'
    estimated_tts_sec = round(final_word_count / 2.65, 2)
    return {
        "plain_spoken": escaped_xml,
        "ssml": ssml,
        "spoken_word_count": final_word_count,
        "word_budget_limit": max_words,
        "within_word_budget": final_word_count <= max_words,
        "was_compressed": truncated,
        "estimated_tts_seconds": estimated_tts_sec,
    }


@dataclass
class ShoppingDialogueState:
    profile_id: str
    active_family: Optional[str] = None
    attributes: List[str] = field(default_factory=list)
    size_pref: Optional[str] = None
    max_price_usd: Optional[float] = None
    candidate_asin: Optional[str] = None
    awaiting_confirmation_asin: Optional[str] = None
    requires_adult_pin: bool = False
    cart_items: List[Dict[str, Any]] = field(default_factory=list)
    confirmed_orders: List[Dict[str, Any]] = field(default_factory=list)
    turn_transcripts: List[str] = field(default_factory=list)
    stage: str = "IDLE"


class VoiceShoppingEngine:
    """Stateful multi-turn Alexa shopping session manager with Bedrock Nova Micro phrasing."""

    def __init__(self, asci_engine: Optional[ASCICascadeEngine] = None) -> None:
        self.asci = asci_engine or ASCICascadeEngine()
        self.asr = PhoneticASRResolver()
        self.sessions: Dict[str, ShoppingDialogueState] = {}

    def get_session(self, profile_id: str) -> ShoppingDialogueState:
        key = profile_id.strip()
        if key not in self.sessions:
            self.sessions[key] = ShoppingDialogueState(profile_id=key)
        return self.sessions[key]

    def reset_sessions(self, profile_id: Optional[str] = None) -> None:
        if profile_id:
            self.sessions.pop(profile_id.strip(), None)
        else:
            self.sessions.clear()

    def _match_catalog(self, state: ShoppingDialogueState, norm_text: str, restrict_non_alcoholic: bool = False) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        candidates = [
            sku for sku in RETAIL_CATALOG
            if (not restrict_non_alcoholic or not sku["age_restricted"])
        ]
        if state.active_family:
            fam_cands = [sku for sku in candidates if sku["family"] == state.active_family]
            if not fam_cands and state.active_family == "hard_cider" and restrict_non_alcoholic:
                fam_cands = [sku for sku in candidates if sku["family"] == "apple_cider"]
            if fam_cands:
                candidates = fam_cands

        scored: List[Tuple[float, Dict[str, Any]]] = []
        low = norm_text.lower()
        for sku in candidates:
            s = 0.0
            if state.max_price_usd is not None:
                if sku["price_usd"] <= state.max_price_usd:
                    s += 3.0
                else:
                    s -= 4.0
            if state.size_pref and sku["size_tier"] == state.size_pref:
                s += 2.5
            for attr in state.attributes:
                if any(attr in a for a in sku["attributes"]) or attr in sku["title"].lower():
                    s += 2.0
            for kw in sku["keywords"]:
                if kw in low:
                    s += 1.8
            s += sku["rating"] * 0.1
            scored.append((s, sku))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        if not scored:
            return None, []
        top_sku = scored[0][1]
        alternates = [pair[1] for pair in scored[1:3]]
        return top_sku, alternates

    def _draft_spoken_with_bedrock(
        self,
        utterance: str,
        top_sku: Dict[str, Any],
        alternates: Sequence[Dict[str, Any]],
        dialogue_stage: str,
        requires_pin: bool,
    ) -> Tuple[str, float, int, Optional[str]]:
        alt_text = (
            f"Alternate: {alternates[0]['short_name']} at ${alternates[0]['price_usd']:.2f}."
            if alternates
            else "No alternate needed."
        )
        deterministic_line = self._deterministic_spoken(top_sku, alternates, dialogue_stage, requires_pin)

        sys_prompt = (
            "You are Alexa Shopping Core's voice response composer. "
            "Write ONE natural spoken response in STRICTLY 32 words or fewer. "
            "Use the exact dollar price and Prime delivery window provided. "
            "Never invent prices, never use bullet points, markdown, or em-dashes. "
            "If stage is AWAITING_CONFIRMATION, ask the customer to say confirm purchase."
        )
        user_prompt = (
            f"Customer said: {utterance}\n"
            f"Dialogue stage: {dialogue_stage}\n"
            f"Top match: {top_sku['short_name']} priced at ${top_sku['price_usd']:.2f} ({top_sku['prime_eta']})\n"
            f"{alt_text}\n"
            f"Requires adult PIN: {requires_pin}"
        )
        llm = invoke_bedrock_converse(
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            model_id=NOVA_MICRO_MODEL,
            max_tokens=95,
            temperature=0.1,
        )
        if not llm or not llm["text"]:
            return deterministic_line, 0.0, 0, None

        draft = llm["text"].strip().strip('"')
        # Verify exact price grounding (either numeric 15.49 or deterministic fallback if LLM omitted/altered price)
        price_str = f"{top_sku['price_usd']:.2f}"
        any_digits = re.findall(r"\$\s*(\d+\.\d{2})", draft)
        if any_digits and price_str not in any_digits:
            return deterministic_line, float(llm["latency_ms"]), int(llm["total_tokens"]), str(llm["model_id"])
        if price_str not in draft and str(int(top_sku["price_usd"])) not in draft:
            return deterministic_line, float(llm["latency_ms"]), int(llm["total_tokens"]), str(llm["model_id"])
        return draft, float(llm["latency_ms"]), int(llm["total_tokens"]), str(llm["model_id"])

    @staticmethod
    def _deterministic_spoken(
        top_sku: Dict[str, Any],
        alternates: Sequence[Dict[str, Any]],
        dialogue_stage: str,
        requires_pin: bool,
    ) -> str:
        price = f"${top_sku['price_usd']:.2f}"
        if dialogue_stage == "AWAITING_CONFIRMATION":
            pin_clause = " with your four-digit adult PIN" if requires_pin else ""
            return (
                f"Ready to add {top_sku['short_name']} for {price} with {top_sku['prime_eta']}. "
                f"Say confirm purchase{pin_clause} to place this order."
            )
        if alternates:
            alt = alternates[0]
            return (
                f"Top pick is {top_sku['short_name']} at {price} with {top_sku['prime_eta']}. "
                f"{alt['short_name']} is ${alt['price_usd']:.2f}. Want to add the top pick?"
            )
        return (
            f"I found {top_sku['short_name']} for {price} with {top_sku['prime_eta']}. "
            f"Would you like to add it to your cart?"
        )

    def handle_turn(
        self,
        utterance: str,
        profile_id: str = "Alice",
        profile_tier: Optional[str] = None,
        adult_pin: Optional[str] = None,
        expected_adult_pin: str = "4821",
        use_bedrock_phrasing: bool = True,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        tier = normalize_profile_tier(profile_tier or profile_id)
        state = self.get_session(profile_id)

        # Step 1: ASCI Sensitive-Content Moderation (Stage-1 <1ms NLU + Stage-2 Nova Lite if gray-zone)
        moderation = self.asci.moderate(
            utterance=utterance,
            active_profile=tier,
            prior_turns=state.turn_transcripts[-3:],
        )

        # Step 2: Phonetic ASR Disambiguation
        asr_info = self.asr.resolve(utterance)
        norm_text = asr_info["normalized_transcript"]
        low = norm_text.lower()

        if not moderation.allowed_to_proceed:
            state.stage = "BLOCKED_SAFETY"
            ssml_bundle = compile_ssml_budgeted(moderation.spoken_response)
            return {
                "profile_id": profile_id,
                "profile_tier": tier,
                "dialogue_stage": state.stage,
                "asr": asr_info,
                "asci": moderation.to_dict(),
                "slots": {
                    "active_family": state.active_family,
                    "attributes": list(state.attributes),
                    "size_pref": state.size_pref,
                    "max_price_usd": state.max_price_usd,
                },
                "ssml_output": ssml_bundle,
                "echo_show_card": {
                    "card_type": "SAFETY_INTERVENTION",
                    "badge": moderation.policy_action,
                    "headline": "Protected by Alexa Sensitive Content Intelligence",
                    "subhead": moderation.rationale,
                    "asin": None,
                    "price_display": None,
                    "prime_eta": None,
                },
                "cart_count": len(state.cart_items),
                "orders_count": len(state.confirmed_orders),
                "shopping_llm_latency_ms": 0.0,
                "shopping_llm_model": None,
                "total_turn_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

        # Step 3: Check for Explicit Voice Purchase Confirmation Gate
        is_confirm_intent = bool(
            re.search(r"\b(confirm\s+purchase|confirm\s+order|yes\s+place\s+order|yes\s+buy\s+it)\b", low)
        )
        if is_confirm_intent and state.awaiting_confirmation_asin:
            pending_sku = next(
                (s for s in RETAIL_CATALOG if s["asin"] == state.awaiting_confirmation_asin),
                None,
            )
            if tier in ("Amazon Kids", "Guest"):
                spoken = (
                    f"Voice purchasing is disabled for {tier} profiles. "
                    f"An adult household member can review items in the Alexa app."
                )
                state.stage = "CHECKOUT_BLOCKED_PROFILE"
                ssml_bundle = compile_ssml_budgeted(spoken)
                return {
                    "profile_id": profile_id,
                    "profile_tier": tier,
                    "dialogue_stage": state.stage,
                    "asr": asr_info,
                    "asci": moderation.to_dict(),
                    "slots": {
                        "active_family": state.active_family,
                        "attributes": list(state.attributes),
                        "size_pref": state.size_pref,
                        "max_price_usd": state.max_price_usd,
                    },
                    "ssml_output": ssml_bundle,
                    "echo_show_card": {
                        "card_type": "PROFILE_PURCHASE_LOCK",
                        "badge": "ADULT VOICE REQUIRED",
                        "headline": pending_sku["title"] if pending_sku else "Pending Item",
                        "subhead": spoken,
                        "asin": state.awaiting_confirmation_asin,
                        "price_display": f"${pending_sku['price_usd']:.2f}" if pending_sku else None,
                        "prime_eta": pending_sku["prime_eta"] if pending_sku else None,
                    },
                    "cart_count": len(state.cart_items),
                    "orders_count": len(state.confirmed_orders),
                    "shopping_llm_latency_ms": 0.0,
                    "shopping_llm_model": None,
                    "total_turn_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }

            if (state.requires_adult_pin or (pending_sku and pending_sku["age_restricted"])) and (
                str(adult_pin or "").strip() != expected_adult_pin
            ):
                spoken = (
                    f"Please provide your valid four-digit adult PIN to confirm "
                    f"{pending_sku['short_name'] if pending_sku else 'this age-gated order'}."
                )
                state.stage = "AWAITING_ADULT_PIN"
                ssml_bundle = compile_ssml_budgeted(spoken)
                return {
                    "profile_id": profile_id,
                    "profile_tier": tier,
                    "dialogue_stage": state.stage,
                    "asr": asr_info,
                    "asci": moderation.to_dict(),
                    "slots": {
                        "active_family": state.active_family,
                        "attributes": list(state.attributes),
                        "size_pref": state.size_pref,
                        "max_price_usd": state.max_price_usd,
                    },
                    "ssml_output": ssml_bundle,
                    "echo_show_card": {
                        "card_type": "ADULT_PIN_REQUIRED",
                        "badge": "PIN VERIFICATION NEEDED",
                        "headline": pending_sku["title"] if pending_sku else "Age-Gated Checkout",
                        "subhead": "Enter or speak your 4-digit household adult PIN to authorize.",
                        "asin": state.awaiting_confirmation_asin,
                        "price_display": f"${pending_sku['price_usd']:.2f}" if pending_sku else None,
                        "prime_eta": pending_sku["prime_eta"] if pending_sku else None,
                    },
                    "cart_count": len(state.cart_items),
                    "orders_count": len(state.confirmed_orders),
                    "shopping_llm_latency_ms": 0.0,
                    "shopping_llm_model": None,
                    "total_turn_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }

            if pending_sku:
                order_record = {
                    "order_id": f"AMZ-{1001 + len(state.confirmed_orders)}",
                    "asin": pending_sku["asin"],
                    "title": pending_sku["title"],
                    "price_usd": pending_sku["price_usd"],
                    "prime_eta": pending_sku["prime_eta"],
                }
                state.confirmed_orders.append(order_record)
                state.cart_items.append(order_record)
                state.awaiting_confirmation_asin = None
                state.requires_adult_pin = False
                state.stage = "CONFIRMED_PURCHASE"
                state.turn_transcripts.append(utterance)
                spoken = (
                    f"Order confirmed for {pending_sku['short_name']} at "
                    f"${pending_sku['price_usd']:.2f}, arriving via {pending_sku['prime_eta']}."
                )
                ssml_bundle = compile_ssml_budgeted(spoken)
                return {
                    "profile_id": profile_id,
                    "profile_tier": tier,
                    "dialogue_stage": state.stage,
                    "asr": asr_info,
                    "asci": moderation.to_dict(),
                    "slots": {
                        "active_family": state.active_family,
                        "attributes": list(state.attributes),
                        "size_pref": state.size_pref,
                        "max_price_usd": state.max_price_usd,
                    },
                    "matched_sku": pending_sku,
                    "order_receipt": order_record,
                    "ssml_output": ssml_bundle,
                    "echo_show_card": {
                        "card_type": "ORDER_CONFIRMED",
                        "badge": f"CONFIRMED · {order_record['order_id']}",
                        "headline": pending_sku["title"],
                        "subhead": f"Arriving {pending_sku['prime_eta']} · Voice Confirmation Verified",
                        "asin": pending_sku["asin"],
                        "price_display": f"${pending_sku['price_usd']:.2f}",
                        "prime_eta": pending_sku["prime_eta"],
                    },
                    "cart_count": len(state.cart_items),
                    "orders_count": len(state.confirmed_orders),
                    "shopping_llm_latency_ms": 0.0,
                    "shopping_llm_model": None,
                    "total_turn_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }

        # Step 4: Update Multi-Turn Conversational Slots (Carry-over + Refinement)
        if asr_info["resolved_family"] and asr_info["resolved_family"] != state.active_family:
            state.active_family = asr_info["resolved_family"]
            state.attributes = []
            state.size_pref = None
            state.max_price_usd = None

        for attr_kw in ("unscented", "organic", "compact", "wireless", "sparkling", "non-alcoholic"):
            if attr_kw in low and attr_kw not in state.attributes:
                state.attributes.append(attr_kw)

        if re.search(r"\b(smaller|small|compact|mini|travel|42[\s-]?count|20[\s-]?pack|12\s*oz)\b", low):
            state.size_pref = "smaller"
        elif re.search(r"\b(larger|large|bulk|tub|81[\s-]?count|48[\s-]?pack)\b", low):
            state.size_pref = "larger"

        price_cap = extract_price_ceiling(low)
        if price_cap is not None:
            state.max_price_usd = price_cap

        restrict_non_alc = moderation.policy_action == "RESTRICT_TO_NON_ALCOHOLIC" or tier in (
            "Amazon Kids",
            "Teen",
            "Guest",
        )
        top_sku, alternates = self._match_catalog(state, norm_text, restrict_non_alcoholic=restrict_non_alc)

        # Non-shopping general informational question cleared by ASCI (e.g., history of gunpowder, kill port 3000)
        if top_sku is None or (
            state.active_family is None
            and not re.search(r"\b(buy|order|add|cart|shop|price|smaller|cheaper|under|reorder|pack|pods|hub)\b", low)
        ):
            state.stage = "INFORMATIONAL_ANSWER"
            state.turn_transcripts.append(utterance)
            spoken, llm_ms, llm_tok, llm_model = self._answer_general_voice_query(utterance, use_bedrock_phrasing)
            ssml_bundle = compile_ssml_budgeted(spoken)
            return {
                "profile_id": profile_id,
                "profile_tier": tier,
                "dialogue_stage": state.stage,
                "asr": asr_info,
                "asci": moderation.to_dict(),
                "slots": {
                    "active_family": state.active_family,
                    "attributes": list(state.attributes),
                    "size_pref": state.size_pref,
                    "max_price_usd": state.max_price_usd,
                },
                "matched_sku": None,
                "ssml_output": ssml_bundle,
                "echo_show_card": {
                    "card_type": "VOICE_ANSWER",
                    "badge": "ASCI CLEARED · BENIGN",
                    "headline": "Voice Answer Delivered",
                    "subhead": ssml_bundle["plain_spoken"],
                    "asin": None,
                    "price_display": None,
                    "prime_eta": None,
                },
                "cart_count": len(state.cart_items),
                "orders_count": len(state.confirmed_orders),
                "shopping_llm_latency_ms": llm_ms,
                "shopping_llm_tokens": llm_tok,
                "shopping_llm_model": llm_model,
                "total_turn_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

        state.candidate_asin = top_sku["asin"]
        wants_checkout = bool(
            re.search(r"\b(add\s+(?:it|that|this|to\s+cart)|buy\s+now|checkout|place\s+order|order\s+(?:it|that)\s+now)\b", low)
        )
        is_refinement = bool(
            state.size_pref or state.max_price_usd is not None or re.search(r"\b(what\s+about|instead|cheaper|smaller|unscented\s+one)\b", low)
        )

        if wants_checkout:
            state.stage = "AWAITING_CONFIRMATION"
            state.awaiting_confirmation_asin = top_sku["asin"]
            state.requires_adult_pin = bool(moderation.requires_adult_pin or top_sku["age_restricted"])
        elif is_refinement:
            state.stage = "VARIANT_NARROWING"
            state.awaiting_confirmation_asin = top_sku["asin"]
            state.requires_adult_pin = bool(moderation.requires_adult_pin or top_sku["age_restricted"])
        else:
            state.stage = "DISCOVERY"
            state.awaiting_confirmation_asin = top_sku["asin"]
            state.requires_adult_pin = bool(moderation.requires_adult_pin or top_sku["age_restricted"])

        state.turn_transcripts.append(utterance)

        if use_bedrock_phrasing:
            spoken, llm_ms, llm_tok, llm_model = self._draft_spoken_with_bedrock(
                utterance=norm_text,
                top_sku=top_sku,
                alternates=alternates,
                dialogue_stage=state.stage,
                requires_pin=state.requires_adult_pin,
            )
        else:
            spoken = self._deterministic_spoken(top_sku, alternates, state.stage, state.requires_adult_pin)
            llm_ms, llm_tok, llm_model = 0.0, 0, None

        ssml_bundle = compile_ssml_budgeted(spoken)
        alt_summary = (
            f"Compare: {alternates[0]['short_name']} (${alternates[0]['price_usd']:.2f})"
            if alternates
            else "Exact catalog match"
        )

        return {
            "profile_id": profile_id,
            "profile_tier": tier,
            "dialogue_stage": state.stage,
            "asr": asr_info,
            "asci": moderation.to_dict(),
            "slots": {
                "active_family": state.active_family,
                "attributes": list(state.attributes),
                "size_pref": state.size_pref,
                "max_price_usd": state.max_price_usd,
                "candidate_asin": state.candidate_asin,
            },
            "matched_sku": top_sku,
            "alternates": alternates,
            "ssml_output": ssml_bundle,
            "echo_show_card": {
                "card_type": "ECHO_SHOW_PRODUCT_CARD",
                "badge": state.stage.replace("_", " "),
                "headline": top_sku["title"],
                "subhead": f"{top_sku['prime_eta']} · ★ {top_sku['rating']} · {alt_summary}",
                "asin": top_sku["asin"],
                "price_display": f"${top_sku['price_usd']:.2f}",
                "prime_eta": top_sku["prime_eta"],
                "requires_adult_pin": state.requires_adult_pin,
            },
            "cart_count": len(state.cart_items),
            "orders_count": len(state.confirmed_orders),
            "shopping_llm_latency_ms": llm_ms,
            "shopping_llm_tokens": llm_tok,
            "shopping_llm_model": llm_model,
            "total_turn_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    @staticmethod
    def _answer_general_voice_query(utterance: str, use_bedrock: bool = True) -> Tuple[str, float, int, Optional[str]]:
        fallback = "Here is a quick answer: that request is safe and ready on your Echo speaker."
        low = utterance.lower()
        if "gunpowder" in low:
            fallback = "Gunpowder was invented in ninth-century China by Tang dynasty alchemists mixing saltpeter, sulfur, and charcoal."
        elif "port 3000" in low or "background process" in low:
            fallback = "Run lsof dash i colon 3000 to find the process ID, then run kill with that PID."
        elif "killer queen" in low:
            fallback = "Playing Killer Queen by Queen from their 1974 album Sheer Heart Attack on Living Room Echo."
        elif "grapes" in low and ("dog" in low or "pet" in low):
            fallback = "No, dogs should never eat grapes or raisins because even small amounts can cause acute kidney failure."

        if not use_bedrock:
            return fallback, 0.0, 0, None

        llm = invoke_bedrock_converse(
            system_prompt=(
                "You are Alexa answering a safe household question aloud. "
                "Reply in ONE concise sentence of 28 words or fewer, plain spoken English, no markdown, no em-dashes."
            ),
            user_prompt=utterance,
            model_id=NOVA_MICRO_MODEL,
            max_tokens=75,
            temperature=0.1,
        )
        if not llm or not llm["text"]:
            return fallback, 0.0, 0, None
        return llm["text"].strip(), float(llm["latency_ms"]), int(llm["total_tokens"]), str(llm["model_id"])
