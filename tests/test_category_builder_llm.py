import json
from pathlib import Path

from src.category_builder import generate_category_time_series, get_embedding_backend


def test_llm_backend_end_to_end(tmp_path):
    emb_src = Path(__file__).resolve().parent / "fixtures" / "llm_embeddings.json"
    backend = get_embedding_backend("llm", lm_embedding_path=emb_src, lm_lowercase_tokens=True)

    features_root = tmp_path / "features"
    paths = {"figs": str(tmp_path / "figs")}
    categories_cfg = {
        "category_set": "toy",
        "sets": {
            "toy": {
                "alpha": {"seeds": ["hello"]},
                "beta": {"seeds": ["world"]},
            }
        },
        "category_score_method": "similarity",
        "allow_single_seed": True,
        "expansion": {"enabled": False},
    }
    events = [
        ("hello", 0.0, 0.9),
        ("world", 1.0, 1.9),
    ]

    result = generate_category_time_series(
        subject="SUBJ",
        story="STORY",
        cfg_base={},
        categories_cfg_base=categories_cfg,
        cluster_csv_path="",
        temporal_weighting="proportional",
        prototype_weight_power=1.0,
        smoothing_seconds=0.0,
        smoothing_method="moving_average",
        gaussian_sigma_seconds=None,
        smoothing_pad="edge",
        seconds_bin_width=1.0,
        features_root=features_root,
        paths=paths,
        TR=1.0,
        embedding_backend=backend,
        save_outputs=True,
        word_events=events,
    )

    df = result["category_df_selected"]
    assert "cat_alpha" in df.columns and "cat_beta" in df.columns
    assert df.loc[0, "cat_alpha"] > 0.9
    assert df.loc[0, "cat_beta"] < 0.1
    assert df.loc[1, "cat_beta"] > 0.9
    assert df.loc[1, "cat_alpha"] < 0.1

    definition_path = features_root / "subjects" / "SUBJ" / "STORY" / "category_definition.json"
    assert definition_path.exists()
    meta = json.loads(definition_path.read_text())
    backend_meta = meta.get("_embedding_backend")
    assert backend_meta is not None
    assert backend_meta["name"] == "llm"
    assert backend_meta["embedding_dim"] == 2
    assert backend_meta["lowercase_tokens"] is True
