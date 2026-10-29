from __future__ import annotations

import ast
import json
import os
import re
import zlib
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

from pipeline.project_paths import env_or_project_path, project_path

TARGET_STATE = os.getenv("TARGET_STATE_OVERRIDE", "AZ").strip().upper() or "AZ"
PARQUET_BASE_DIR = env_or_project_path("PARQUET_BASE_DIR", "data/parquet")

STAGE09_BUCKET5_DIR = os.getenv("SOURCE_STAGE09_BUCKET5_DIR", "").strip()
STAGE09_BUCKET10_DIR = os.getenv("SOURCE_STAGE09_BUCKET10_DIR", "").strip()

INDEX_MAPS_ROOT = env_or_project_path("OUTPUT_09_INDEX_MAPS_ROOT_DIR", "data/output/09_index_maps")
INTERACTION_ROOT = env_or_project_path("OUTPUT_09_INTERACTION_V2_WEIGHTED_ROOT_DIR", "data/output/09_interaction_v2_weighted")
MERCHANT_CARD_ROOT = env_or_project_path("OUTPUT_09_MERCHANT_CARD_ROOT_DIR", "data/output/09_merchant_semantic_card")
TEXT_EMBED_ROOT = env_or_project_path(
    "OUTPUT_09_CANDIDATE_WISE_TEXT_EMBEDDINGS_V1_ROOT_DIR",
    "data/output/09_candidate_wise_text_embeddings_v1",
)
USER_SCHEMA_ROOT = env_or_project_path(
    "OUTPUT_09_USER_SCHEMA_PROJECTION_V1_ROOT_DIR",
    "data/output/09_user_schema_projection_v1",
)
MERCHANT_EVIDENCE_ROOT = env_or_project_path(
    "OUTPUT_09_MERCHANT_EVIDENCE_ASSETS_V1_ROOT_DIR",
    "data/output/09_merchant_evidence_assets_v1",
)

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
EMBED_DIM = int(os.getenv("AZ_SUPPORT_TEXT_EMBED_DIM", "96").strip() or 96)
MAX_TAGS_PER_DOMAIN = int(os.getenv("AZ_SUPPORT_MAX_TAGS_PER_DOMAIN", "12").strip() or 12)

CUISINE_SYNONYMS = {
    "mexican": "mexican",
    "tacos": "mexican",
    "tex-mex": "mexican",
    "pizza": "pizza",
    "italian": "italian",
    "pasta": "italian",
    "american": "american",
    "burgers": "american",
    "breakfast": "breakfast_brunch",
    "brunch": "breakfast_brunch",
    "cafes": "cafe",
    "coffee": "cafe",
    "coffee & tea": "cafe",
    "sushi bars": "sushi",
    "sushi": "sushi",
    "japanese": "japanese",
    "ramen": "japanese",
    "chinese": "chinese",
    "dim sum": "chinese",
    "thai": "thai",
    "indian": "indian",
    "mediterranean": "mediterranean",
    "greek": "mediterranean",
    "middle eastern": "middle_eastern",
    "korean": "korean",
    "vietnamese": "vietnamese",
    "sandwiches": "sandwiches",
    "seafood": "seafood",
    "steakhouses": "steakhouse",
    "barbeque": "bbq",
    "bbq": "bbq",
    "vegetarian": "vegetarian",
    "vegan": "vegan",
    "desserts": "dessert",
    "bakeries": "bakery",
    "chicken wings": "wings",
    "sports bars": "sports_bar",
    "bars": "bar",
    "cocktail bars": "bar",
}
SCENE_CATEGORY_MAP = {
    "family": ["family_scene"],
    "kids": ["family_scene"],
    "nightlife": ["nightlife_scene"],
    "bars": ["nightlife_scene", "group_scene"],
    "lounges": ["nightlife_scene", "date_scene"],
    "date": ["date_scene"],
    "romantic": ["date_scene"],
    "fast food": ["fast_casual_scene"],
    "food trucks": ["fast_casual_scene"],
    "cafes": ["sitdown_scene"],
    "coffee": ["sitdown_scene"],
    "restaurants": ["sitdown_scene", "group_scene"],
}
PROPERTY_KEYWORDS = {
    "delivery": ["delivery"],
    "takeout": ["takeout"],
    "reservations": ["reservations"],
    "reservation": ["reservations"],
    "patio": ["outdoor_seating"],
    "outdoor seating": ["outdoor_seating"],
    "drive-thru": ["drive_thru"],
    "drive thru": ["drive_thru"],
}
TIME_KEYWORDS = {
    "breakfast": ["breakfast"],
    "brunch": ["breakfast", "brunch"],
    "lunch": ["lunch"],
    "dinner": ["dinner"],
    "late night": ["late_night"],
    "late-night": ["late_night"],
    "happy hour": ["happy_hour"],
}
SERVICE_KEYWORDS = {
    "friendly": ["friendly_service"],
    "helpful": ["helpful_service"],
    "quick": ["quick_service"],
    "fast": ["quick_service"],
    "attentive": ["attentive_service"],
    "clean": ["cleanliness"],
}
COMPLAINT_KEYWORDS = {
    "rude": ["rude_service"],
    "slow": ["slow_service"],
    "dirty": ["dirty"],
    "expensive": ["overpriced"],
    "cold": ["cold_food"],
    "bland": ["bland_food"],
    "wait": ["long_wait"],
}
DISH_KEYWORDS = {
    "burger": ["burger"],
    "taco": ["tacos"],
    "pizza": ["pizza"],
    "sushi": ["sushi"],
    "ramen": ["ramen"],
    "bbq": ["bbq"],
    "burrito": ["burrito"],
    "fries": ["fries"],
    "coffee": ["coffee"],
    "brunch": ["brunch"],
    "wings": ["wings"],
    "salad": ["salad"],
}
RECOMMEND_RE = re.compile(r"\b(recommend|must try|favorite|best|love|worth it)\b", re.I)
TIME_RE = re.compile(r"\b(breakfast|brunch|lunch|dinner|late night|happy hour|weekend)\b", re.I)
DISH_RE = re.compile(r"\b(burger|taco|pizza|sushi|ramen|burrito|fries|wings|salad|coffee)\b", re.I)


def now_run_id(tag: str) -> str:
    return f"{RUN_TS}_full_{tag}"


def resolve_path(raw: str):
    path = project_path(raw) if not os.path.isabs(raw) else raw
    return pd.io.common.stringify_path(path)


def require_bucket_dir(raw: str, bucket: int):
    if not raw:
        raise FileNotFoundError(f"SOURCE_STAGE09_BUCKET{bucket}_DIR is required")
    path = resolve_path(raw)
    if not os.path.exists(path):
        raise FileNotFoundError(f"bucket dir not found: {path}")
    return path


def parquet_dir(name: str):
    path = PARQUET_BASE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"parquet dir not found: {path}")
    return path


def safe_json_write(path, payload: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def normalize_tag(value: object) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def parse_categories(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    parts = value if isinstance(value, list) else [part.strip() for part in str(value).split(",")]
    return [str(part).strip() for part in parts if str(part).strip()]


def parse_nested_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if value in (None, "", np.nan):
        return {}
    try:
        parsed = ast.literal_eval(str(value))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def attr_flag(attrs: object, key: str) -> float:
    parsed = parse_nested_dict(attrs)
    val = parsed.get(key)
    if isinstance(val, bool):
        return float(val)
    if val is None:
        return 0.0
    return 1.0 if str(val).strip().lower() in {"true", "1", "yes"} else 0.0


def iter_hours(hours: object) -> list[tuple[str, int, int]]:
    parsed = parse_nested_dict(hours)
    rows: list[tuple[str, int, int]] = []
    for day, span in parsed.items():
        if not span:
            continue
        try:
            start_text, end_text = str(span).split("-", 1)
            start_hour = int(start_text.split(":", 1)[0])
            end_hour = int(end_text.split(":", 1)[0])
        except Exception:
            continue
        rows.append((str(day), start_hour, end_hour))
    return rows


def geo_cell_3dp(lat: object, lon: object) -> str:
    try:
        return f"{float(lat):.3f}_{float(lon):.3f}"
    except Exception:
        return ""


def extract_domain_tags(text: str, keyword_map: dict[str, list[str]]) -> list[str]:
    lower = (text or "").lower()
    out: list[str] = []
    for key, tags in keyword_map.items():
        if key in lower:
            out.extend(tags)
    return sorted(set(out))


def pick_top(counter: Counter[str], topn: int = MAX_TAGS_PER_DOMAIN) -> list[str]:
    return [tag for tag, _ in counter.most_common(topn)]


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den_safe = den.replace(0, np.nan)
    return (num / den_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def hash_embed(tokens: list[str], dim: int = EMBED_DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in tokens:
        token = normalize_tag(tok)
        if not token:
            continue
        h = zlib.crc32(token.encode("utf-8"))
        idx = h % dim
        sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def build_bucket_maps(bucket_dirs: dict[int, str], out_dir: str) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    user_maps: dict[int, pd.DataFrame] = {}
    item_maps: dict[int, pd.DataFrame] = {}
    for bucket, bucket_dir in bucket_dirs.items():
        truth = pd.read_parquet(os.path.join(bucket_dir, "truth.parquet"), columns=["user_idx", "user_id"])
        cand = pd.read_parquet(os.path.join(bucket_dir, "candidates_pretrim.parquet"), columns=["item_idx", "business_id"])
        user_map = truth.drop_duplicates().sort_values(["user_idx", "user_id"]).reset_index(drop=True)
        item_map = cand.dropna().drop_duplicates().sort_values(["item_idx", "business_id"]).reset_index(drop=True)
        bucket_out = os.path.join(out_dir, f"bucket_{bucket}")
        os.makedirs(bucket_out, exist_ok=True)
        user_map.to_parquet(os.path.join(bucket_out, "user_index_map.parquet"), index=False)
        item_map.to_parquet(os.path.join(bucket_out, "item_index_map.parquet"), index=False)
        user_maps[bucket] = user_map
        item_maps[bucket] = item_map
    return user_maps, item_maps


def build_merchant_assets(az_business: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    az_ids = set(az_business["business_id"].astype(str))

    review = pd.read_parquet(parquet_dir("yelp_academic_dataset_review"), columns=["business_id", "stars", "text"])
    review = review.loc[review["business_id"].astype(str).isin(az_ids)].copy()
    review["stars"] = pd.to_numeric(review["stars"], errors="coerce").fillna(0.0)
    review["text"] = review["text"].fillna("").astype(str)
    review["review_positive"] = (review["stars"] >= 4.0).astype(int)
    review["review_negative"] = (review["stars"] <= 2.0).astype(int)

    tip = pd.read_parquet(parquet_dir("yelp_academic_dataset_tip"), columns=["business_id", "text"])
    tip = tip.loc[tip["business_id"].astype(str).isin(az_ids)].copy()
    tip["text"] = tip["text"].fillna("").astype(str)

    checkin = pd.read_parquet(parquet_dir("yelp_academic_dataset_checkin"), columns=["business_id", "date"])
    checkin = checkin.loc[checkin["business_id"].astype(str).isin(az_ids)].copy()

    review_agg = (
        review.groupby("business_id", as_index=False)
        .agg(
            audit_review_count=("business_id", "size"),
            positive_review_count=("review_positive", "sum"),
            negative_review_count=("review_negative", "sum"),
        )
        .reset_index(drop=True)
    )
    review_agg["evidence_positive_per_review"] = safe_div(review_agg["positive_review_count"], review_agg["audit_review_count"])
    review_agg["evidence_negative_per_review"] = safe_div(review_agg["negative_review_count"], review_agg["audit_review_count"])

    tip["has_recommend_cue"] = tip["text"].str.contains(RECOMMEND_RE).astype(int)
    tip["has_time_cue"] = tip["text"].str.contains(TIME_RE).astype(int)
    tip["has_dish_cue"] = tip["text"].str.contains(DISH_RE).astype(int)
    tip_agg = (
        tip.groupby("business_id", as_index=False)
        .agg(
            audit_tip_rows=("business_id", "size"),
            recommend_tip_rows=("has_recommend_cue", "sum"),
            time_tip_rows=("has_time_cue", "sum"),
            dish_tip_rows=("has_dish_cue", "sum"),
        )
        .reset_index(drop=True)
    )
    tip_agg["tip_recommend_share"] = safe_div(tip_agg["recommend_tip_rows"], tip_agg["audit_tip_rows"])
    tip_agg["tip_time_share"] = safe_div(tip_agg["time_tip_rows"], tip_agg["audit_tip_rows"])
    tip_agg["tip_dish_share"] = safe_div(tip_agg["dish_tip_rows"], tip_agg["audit_tip_rows"])

    checkin_rows: list[dict[str, object]] = []
    for row in checkin.itertuples(index=False):
        entries = [part.strip() for part in str(row.date or "").split(",") if part.strip()]
        total = weekend = breakfast = lunch = dinner = late = 0
        for entry in entries:
            try:
                ts = pd.to_datetime(entry)
            except Exception:
                continue
            total += 1
            hour = int(ts.hour)
            if ts.dayofweek >= 5:
                weekend += 1
            if 5 <= hour < 11:
                breakfast += 1
            elif 11 <= hour < 15:
                lunch += 1
            elif 17 <= hour < 22:
                dinner += 1
            elif hour >= 22 or hour < 5:
                late += 1
        checkin_rows.append(
            {
                "business_id": row.business_id,
                "audit_checkin_rows": total,
                "weekend_share": float(weekend / total) if total else 0.0,
                "breakfast_share": float(breakfast / total) if total else 0.0,
                "lunch_share": float(lunch / total) if total else 0.0,
                "dinner_share": float(dinner / total) if total else 0.0,
                "late_share": float(late / total) if total else 0.0,
            }
        )
    checkin_agg = pd.DataFrame(checkin_rows)

    tip_text_map = tip.groupby("business_id")["text"].apply(lambda s: " ".join(s.head(50).tolist())).to_dict()
    review_pos_map = review.loc[review["review_positive"] == 1].groupby("business_id")["text"].apply(lambda s: " ".join(s.head(80).tolist())).to_dict()
    review_neg_map = review.loc[review["review_negative"] == 1].groupby("business_id")["text"].apply(lambda s: " ".join(s.head(80).tolist())).to_dict()

    merchant_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []
    for row in az_business.itertuples(index=False):
        categories = parse_categories(row.categories)
        normalized_categories = [normalize_tag(v) for v in categories]
        category_text = " ".join(categories).lower()
        cuisine_tags: list[str] = []
        for cat in categories:
            key = str(cat).strip().lower()
            if key in CUISINE_SYNONYMS:
                cuisine_tags.append(CUISINE_SYNONYMS[key])
        if not cuisine_tags and normalized_categories:
            cuisine_tags.append(normalized_categories[0])
        cuisine_tags = list(dict.fromkeys(cuisine_tags))
        primary_cuisine = cuisine_tags[0] if cuisine_tags else (normalized_categories[0] if normalized_categories else "unknown")
        secondary_cuisine = cuisine_tags[1] if len(cuisine_tags) > 1 else ""

        tip_text = tip_text_map.get(row.business_id, "")
        review_pos_text = review_pos_map.get(row.business_id, "")
        review_neg_text = review_neg_map.get(row.business_id, "")
        scene_tags = extract_domain_tags(category_text, SCENE_CATEGORY_MAP)
        time_tags = sorted(set(extract_domain_tags(category_text, TIME_KEYWORDS)) | set(extract_domain_tags(tip_text, TIME_KEYWORDS)))
        property_tags = sorted(set(extract_domain_tags(category_text, PROPERTY_KEYWORDS)) | set(extract_domain_tags(tip_text, PROPERTY_KEYWORDS)))
        dish_tags = sorted(
            set(extract_domain_tags(category_text, DISH_KEYWORDS))
            | set(extract_domain_tags(tip_text, DISH_KEYWORDS))
            | set(extract_domain_tags(review_pos_text, DISH_KEYWORDS))
        )
        service_tags = sorted(set(extract_domain_tags(tip_text, SERVICE_KEYWORDS)) | set(extract_domain_tags(review_pos_text, SERVICE_KEYWORDS)))
        complaint_tags = sorted(extract_domain_tags(review_neg_text, COMPLAINT_KEYWORDS))

        hour_rows = iter_hours(row.hours)
        open_weekend = 1.0 if any(day in {"Saturday", "Sunday"} for day, _, _ in hour_rows) else 0.0
        open_late_any = 1.0 if any(end >= 22 or end <= 3 for _, _, end in hour_rows) else 0.0

        merchant_rows.append(
            {
                "business_id": row.business_id,
                "city": row.city,
                "geo_cell_3dp": geo_cell_3dp(getattr(row, "latitude", None), getattr(row, "longitude", None)),
                "merchant_primary_cuisine": primary_cuisine,
                "merchant_primary_cuisine_score": 1.0 if primary_cuisine else 0.0,
                "merchant_secondary_cuisine": secondary_cuisine,
                "merchant_secondary_cuisine_score": 0.65 if secondary_cuisine else 0.0,
                "family_scene_fit": float("family_scene" in scene_tags or attr_flag(row.attributes, "GoodForKids") > 0),
                "group_scene_fit": float("group_scene" in scene_tags or attr_flag(row.attributes, "RestaurantsGoodForGroups") > 0),
                "date_scene_fit": float("date_scene" in scene_tags),
                "nightlife_scene_fit": float("nightlife_scene" in scene_tags),
                "fast_casual_fit": float("fast_casual_scene" in scene_tags or "fast_food" in normalized_categories),
                "sitdown_fit": float("sitdown_scene" in scene_tags or attr_flag(row.attributes, "RestaurantsReservations") > 0),
                "meal_breakfast_fit": float("breakfast" in time_tags or "breakfast_brunch" in cuisine_tags),
                "meal_lunch_fit": float("lunch" in time_tags),
                "meal_dinner_fit": float("dinner" in time_tags or "steakhouse" in cuisine_tags),
                "late_night_fit": float("late_night" in time_tags or open_late_any > 0),
                "attr_delivery": attr_flag(row.attributes, "RestaurantsDelivery"),
                "attr_takeout": attr_flag(row.attributes, "RestaurantsTakeOut"),
                "attr_reservations": attr_flag(row.attributes, "RestaurantsReservations"),
                "open_weekend": open_weekend,
                "open_late_any": open_late_any,
            }
        )
        evidence_rows.append(
            {
                "business_id": row.business_id,
                "evidence_cuisine_tags_v1": cuisine_tags[:MAX_TAGS_PER_DOMAIN],
                "evidence_dish_tags_v1": dish_tags[:MAX_TAGS_PER_DOMAIN],
                "evidence_scene_tags_v1": scene_tags[:MAX_TAGS_PER_DOMAIN],
                "evidence_time_tags_v1": time_tags[:MAX_TAGS_PER_DOMAIN],
                "evidence_property_tags_v1": property_tags[:MAX_TAGS_PER_DOMAIN],
                "evidence_service_tags_v1": service_tags[:MAX_TAGS_PER_DOMAIN],
                "evidence_complaint_tags_v1": complaint_tags[:MAX_TAGS_PER_DOMAIN],
            }
        )

    merchant_df = pd.DataFrame(merchant_rows)
    evidence_df = pd.DataFrame(evidence_rows)
    merchant_df = merchant_df.merge(review_agg, on="business_id", how="left").merge(tip_agg, on="business_id", how="left").merge(checkin_agg, on="business_id", how="left")
    for col in [
        "audit_review_count",
        "positive_review_count",
        "negative_review_count",
        "audit_tip_rows",
        "recommend_tip_rows",
        "time_tip_rows",
        "dish_tip_rows",
        "audit_checkin_rows",
    ]:
        if col in merchant_df.columns:
            merchant_df[col] = pd.to_numeric(merchant_df[col], errors="coerce").fillna(0).astype(int)
    for col in [
        "evidence_positive_per_review",
        "evidence_negative_per_review",
        "tip_recommend_share",
        "tip_time_share",
        "tip_dish_share",
        "weekend_share",
        "breakfast_share",
        "lunch_share",
        "dinner_share",
        "late_share",
    ]:
        if col not in merchant_df.columns:
            merchant_df[col] = 0.0
        merchant_df[col] = pd.to_numeric(merchant_df[col], errors="coerce").fillna(0.0)
    merchant_df = merchant_df.fillna({"merchant_secondary_cuisine": "", "city": "", "geo_cell_3dp": ""})
    merchant_evidence_df = evidence_df.merge(merchant_df[["business_id", "audit_review_count"]], on="business_id", how="left")
    merchant_evidence_df["audit_review_count"] = pd.to_numeric(merchant_evidence_df["audit_review_count"], errors="coerce").fillna(0).astype(int)
    return merchant_df, merchant_evidence_df


def build_interactions(
    bucket_dirs: dict[int, str],
    user_maps: dict[int, pd.DataFrame],
    item_maps: dict[int, pd.DataFrame],
    merchant_df: pd.DataFrame,
) -> pd.DataFrame:
    merchant_cues = merchant_df[["business_id", "tip_recommend_share", "tip_time_share", "tip_dish_share"]].copy()
    frames: list[pd.DataFrame] = []
    for bucket, bucket_dir in bucket_dirs.items():
        hist = pd.read_parquet(os.path.join(bucket_dir, "train_history.parquet"))
        hist = hist.merge(user_maps[bucket], on="user_idx", how="left").merge(item_maps[bucket], on="item_idx", how="left")
        hist = hist.dropna(subset=["user_id", "business_id"]).copy()
        hist["hist_rating"] = pd.to_numeric(hist["hist_rating"], errors="coerce").fillna(0.0)
        hist["hist_recency_w"] = pd.to_numeric(hist["hist_recency_w"], errors="coerce").fillna(0.0)
        hist["event_time"] = pd.to_datetime(hist["hist_ts"], errors="coerce")
        hist["n_train"] = hist.groupby("user_id")["item_idx"].transform("size").astype(int)
        hist = hist.merge(merchant_cues, on="business_id", how="left")
        for cue in ["tip_recommend_share", "tip_time_share", "tip_dish_share"]:
            hist[cue] = pd.to_numeric(hist[cue], errors="coerce").fillna(0.0)
        hist["has_recommend_cue"] = (hist["tip_recommend_share"] >= 0.08).astype(float)
        hist["has_time_cue"] = (hist["tip_time_share"] >= 0.06).astype(float)
        hist["has_dish_cue"] = (hist["tip_dish_share"] >= 0.06).astype(float)
        hist["event_type"] = np.select(
            [hist["hist_rating"] >= 4.0, hist["hist_rating"] <= 2.0, hist["hist_rating"] == 3.0],
            ["review_positive", "review_negative", "review_neutral"],
            default="review_neutral",
        )
        hist["sample_weight_v2"] = hist["hist_recency_w"].clip(lower=0.05) * np.select(
            [hist["event_type"] == "review_positive", hist["event_type"] == "review_negative"],
            [1.00, 0.90],
            default=0.55,
        )
        hist["event_id"] = [f"bucket{bucket}_hist_{i}" for i in range(len(hist))]
        base_cols = ["event_id", "event_type", "user_id", "business_id", "event_time", "sample_weight_v2", "n_train", "has_recommend_cue", "has_time_cue", "has_dish_cue"]
        frames.append(hist[base_cols].copy())

        tip_rows = hist.loc[(hist["has_recommend_cue"] + hist["has_time_cue"] + hist["has_dish_cue"]) > 0].copy()
        if not tip_rows.empty:
            tip_rows["event_type"] = "tip_signal"
            tip_rows["sample_weight_v2"] = tip_rows["hist_recency_w"].clip(lower=0.05) * 0.35 * (1.0 + 0.3 * (tip_rows["has_recommend_cue"] + tip_rows["has_time_cue"] + tip_rows["has_dish_cue"]))
            tip_rows["event_id"] = [f"bucket{bucket}_tip_{i}" for i in range(len(tip_rows))]
            frames.append(tip_rows[base_cols].copy())
    events = pd.concat(frames, ignore_index=True)
    events["sample_weight_v2"] = pd.to_numeric(events["sample_weight_v2"], errors="coerce").fillna(0.0)
    return events.loc[events["sample_weight_v2"] > 0].reset_index(drop=True)


def build_user_schema_and_embeddings(
    events: pd.DataFrame,
    merchant_df: pd.DataFrame,
    merchant_evidence_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    merchant_join = merchant_df.merge(merchant_evidence_df, on="business_id", how="left")
    event_df = events.merge(merchant_join, on="business_id", how="left")

    schema_rows: list[dict[str, object]] = []
    user_ids = sorted(event_df["user_id"].dropna().astype(str).unique().tolist())
    merchant_ids = sorted(merchant_join["business_id"].dropna().astype(str).unique().tolist())
    user_long = np.zeros((len(user_ids), EMBED_DIM), dtype=np.float32)
    user_recent = np.zeros((len(user_ids), EMBED_DIM), dtype=np.float32)
    user_neg = np.zeros((len(user_ids), EMBED_DIM), dtype=np.float32)
    user_ctx = np.zeros((len(user_ids), EMBED_DIM), dtype=np.float32)

    user_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    for user_id, sub in event_df.groupby("user_id", sort=False):
        pos = sub.loc[sub["event_type"].isin(["review_positive", "tip_signal"])].copy()
        neg = sub.loc[sub["event_type"] == "review_negative"].copy()
        recent = sub.sort_values("event_time", ascending=False).head(12)

        pos_cuisine = Counter()
        pos_dish = Counter()
        pos_scene = Counter()
        pos_time = Counter()
        pos_property = Counter()
        pos_service = Counter()
        neg_complaint = Counter()
        neg_cuisine = Counter()
        neg_dish = Counter()

        for row in pos.itertuples(index=False):
            weight = float(getattr(row, "sample_weight_v2", 0.0) or 0.0)
            if getattr(row, "merchant_primary_cuisine", ""):
                pos_cuisine[str(row.merchant_primary_cuisine)] += weight
            for col, counter in [
                ("evidence_dish_tags_v1", pos_dish),
                ("evidence_scene_tags_v1", pos_scene),
                ("evidence_time_tags_v1", pos_time),
                ("evidence_property_tags_v1", pos_property),
                ("evidence_service_tags_v1", pos_service),
            ]:
                for tag in getattr(row, col, []) or []:
                    counter[str(tag)] += weight
        for row in neg.itertuples(index=False):
            weight = float(getattr(row, "sample_weight_v2", 0.0) or 0.0)
            if getattr(row, "merchant_primary_cuisine", ""):
                neg_cuisine[str(row.merchant_primary_cuisine)] += weight
            for col, counter in [("evidence_dish_tags_v1", neg_dish), ("evidence_complaint_tags_v1", neg_complaint)]:
                for tag in getattr(row, col, []) or []:
                    counter[str(tag)] += weight

        schema_rows.append(
            {
                "user_id": user_id,
                "schema_pos_cuisine": pick_top(pos_cuisine),
                "schema_pos_dish": pick_top(pos_dish),
                "schema_pos_scene": pick_top(pos_scene),
                "schema_pos_time": pick_top(pos_time),
                "schema_pos_property": pick_top(pos_property),
                "schema_pos_service": pick_top(pos_service),
                "schema_neg_complaint": pick_top(neg_complaint),
                "schema_neg_cuisine": pick_top(neg_cuisine),
                "schema_neg_dish": pick_top(neg_dish),
            }
        )

        recent_tokens: list[str] = []
        for row in recent.itertuples(index=False):
            recent_tokens.extend([str(getattr(row, "merchant_primary_cuisine", "") or "")])
            recent_tokens.extend((getattr(row, "evidence_time_tags_v1", []) or [])[:3])
            recent_tokens.extend((getattr(row, "evidence_dish_tags_v1", []) or [])[:3])
        user_pos = user_index[str(user_id)]
        user_long[user_pos] = hash_embed(pick_top(pos_cuisine, 24) + pick_top(pos_scene, 16) + pick_top(pos_property, 12))
        user_recent[user_pos] = hash_embed(recent_tokens)
        user_neg[user_pos] = hash_embed(pick_top(neg_cuisine, 16) + pick_top(neg_complaint, 16) + pick_top(neg_dish, 16))
        user_ctx[user_pos] = hash_embed(pick_top(pos_time, 12) + pick_top(pos_property, 12))

    merchant_core = np.zeros((len(merchant_ids), EMBED_DIM), dtype=np.float32)
    merchant_sem = np.zeros((len(merchant_ids), EMBED_DIM), dtype=np.float32)
    merchant_pos = np.zeros((len(merchant_ids), EMBED_DIM), dtype=np.float32)
    merchant_neg = np.zeros((len(merchant_ids), EMBED_DIM), dtype=np.float32)
    merchant_ctx = np.zeros((len(merchant_ids), EMBED_DIM), dtype=np.float32)
    merchant_lookup = merchant_join.set_index("business_id")
    for idx, business_id in enumerate(merchant_ids):
        if business_id not in merchant_lookup.index:
            continue
        rec = merchant_lookup.loc[business_id]
        if isinstance(rec, pd.DataFrame):
            rec = rec.iloc[0]
        core_tokens = [str(rec.get("merchant_primary_cuisine", "") or ""), str(rec.get("merchant_secondary_cuisine", "") or ""), str(rec.get("city", "") or "")]
        sem_tokens = list(rec.get("evidence_cuisine_tags_v1", []) or []) + list(rec.get("evidence_dish_tags_v1", []) or [])
        pos_tokens = list(rec.get("evidence_dish_tags_v1", []) or []) + list(rec.get("evidence_service_tags_v1", []) or [])
        neg_tokens = list(rec.get("evidence_complaint_tags_v1", []) or [])
        ctx_tokens = list(rec.get("evidence_time_tags_v1", []) or []) + list(rec.get("evidence_property_tags_v1", []) or []) + list(rec.get("evidence_scene_tags_v1", []) or [])
        merchant_core[idx] = hash_embed(core_tokens)
        merchant_sem[idx] = hash_embed(sem_tokens)
        merchant_pos[idx] = hash_embed(pos_tokens)
        merchant_neg[idx] = hash_embed(neg_tokens)
        merchant_ctx[idx] = hash_embed(ctx_tokens)

    return (
        pd.DataFrame(schema_rows).sort_values("user_id").reset_index(drop=True),
        pd.DataFrame({"user_id": user_ids}),
        pd.DataFrame({"business_id": merchant_ids}),
        {
            "user_long_pref_emb": user_long,
            "user_recent_intent_emb": user_recent,
            "user_negative_avoid_emb": user_neg,
            "user_context_emb": user_ctx,
        },
        {
            "merchant_core_emb": merchant_core,
            "merchant_semantic_emb": merchant_sem,
            "merchant_pos_emb": merchant_pos,
            "merchant_neg_emb": merchant_neg,
            "merchant_context_emb": merchant_ctx,
        },
    )


def main() -> None:
    bucket_dirs = {5: require_bucket_dir(STAGE09_BUCKET5_DIR, 5), 10: require_bucket_dir(STAGE09_BUCKET10_DIR, 10)}
    business = pd.read_parquet(
        parquet_dir("yelp_academic_dataset_business"),
        columns=["business_id", "name", "city", "state", "latitude", "longitude", "categories", "attributes", "hours"],
    )
    az_business = business.loc[business["state"].astype(str).str.upper() == TARGET_STATE].copy()

    index_run_dir = os.path.join(str(INDEX_MAPS_ROOT), now_run_id("stage09_index_maps_build"))
    os.makedirs(index_run_dir, exist_ok=True)
    user_maps, item_maps = build_bucket_maps(bucket_dirs, index_run_dir)

    merchant_card_df, merchant_evidence_df = build_merchant_assets(az_business)
    merchant_card_run = os.path.join(str(MERCHANT_CARD_ROOT), now_run_id("stage09_merchant_semantic_card_build"))
    merchant_evidence_run = os.path.join(str(MERCHANT_EVIDENCE_ROOT), now_run_id("stage09_merchant_evidence_assets_v1_build"))
    interaction_run = os.path.join(str(INTERACTION_ROOT), now_run_id("stage09_interaction_v2_weight_build"))
    user_schema_run = os.path.join(str(USER_SCHEMA_ROOT), now_run_id("stage09_user_schema_projection_v1_build"))
    text_embed_run = os.path.join(str(TEXT_EMBED_ROOT), now_run_id("stage09_candidate_wise_text_embeddings_v1_build"))
    for out_dir in [merchant_card_run, merchant_evidence_run, interaction_run, user_schema_run, text_embed_run]:
        os.makedirs(out_dir, exist_ok=True)

    merchant_card_df.to_parquet(os.path.join(merchant_card_run, "merchant_semantic_card_v2.parquet"), index=False)
    merchant_evidence_df.to_parquet(os.path.join(merchant_evidence_run, "merchant_evidence_assets_v1.parquet"), index=False)

    interaction_df = build_interactions(bucket_dirs, user_maps, item_maps, merchant_card_df)
    interaction_df.to_parquet(os.path.join(interaction_run, "interaction_v2_weighted.parquet"), index=False)

    user_schema_df, user_ids_df, merchant_ids_df, user_arrays, merchant_arrays = build_user_schema_and_embeddings(
        interaction_df, merchant_card_df, merchant_evidence_df
    )
    user_schema_df.to_parquet(os.path.join(user_schema_run, "user_schema_profile_v1.parquet"), index=False)
    user_ids_df.to_parquet(os.path.join(text_embed_run, "user_text_embedding_ids_v1.parquet"), index=False)
    merchant_ids_df.to_parquet(os.path.join(text_embed_run, "merchant_text_embedding_ids_v1.parquet"), index=False)
    np.savez(os.path.join(text_embed_run, "user_text_embeddings_v1.npz"), **user_arrays)
    np.savez(os.path.join(text_embed_run, "merchant_text_embeddings_v1.npz"), **merchant_arrays)

    summary = {
        "target_state": TARGET_STATE,
        "inputs": {
            "bucket_5_dir": bucket_dirs[5],
            "bucket_10_dir": bucket_dirs[10],
            "parquet_base_dir": str(PARQUET_BASE_DIR),
        },
        "row_counts": {
            "az_business_rows": int(len(az_business)),
            "merchant_card_rows": int(len(merchant_card_df)),
            "merchant_evidence_rows": int(len(merchant_evidence_df)),
            "interaction_rows": int(len(interaction_df)),
            "user_schema_rows": int(len(user_schema_df)),
            "text_user_rows": int(len(user_ids_df)),
            "text_merchant_rows": int(len(merchant_ids_df)),
        },
        "outputs": {
            "index_maps_run_dir": index_run_dir,
            "interaction_run_dir": interaction_run,
            "merchant_card_run_dir": merchant_card_run,
            "merchant_evidence_run_dir": merchant_evidence_run,
            "user_schema_run_dir": user_schema_run,
            "text_embeddings_run_dir": text_embed_run,
        },
    }
    for out_dir in [index_run_dir, interaction_run, merchant_card_run, merchant_evidence_run, user_schema_run, text_embed_run]:
        safe_json_write(os.path.join(out_dir, "run_meta.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
