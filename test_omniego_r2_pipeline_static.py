from pathlib import Path
import ast

PIPELINE = Path(__file__).with_name("run_omniego_r2_pipeline.py")


def _tree():
    assert PIPELINE.exists(), "run_omniego_r2_pipeline.py should define the unified OmniEgo-R2 pipeline"
    return ast.parse(PIPELINE.read_text(encoding="utf-8"))


def _defined_names(tree):
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def test_exposes_paper_pipeline_stages():
    names = _defined_names(_tree())
    assert "OmniEgoR2Pipeline" in names
    assert "temporal_evidence_normalization" in names
    assert "capability_oriented_router" in names
    assert "role_decomposed_reasoning" in names
    assert "boundary_aware_option_verification" in names
    assert "defensive_answer_calibration" in names


def test_unifies_all_domains_and_datasets():
    text = PIPELINE.read_text(encoding="utf-8")
    for token in [
        "Animal", "XSports", "Industry", "Surgery",
        "EgoPet", "ExtrameSportFPV", "ENIGMA", "CholecTrack20", "EgoSurgery",
    ]:
        assert token in text


def test_reuses_original_domain_prompts_and_call_paths():
    text = PIPELINE.read_text(encoding="utf-8")
    for module_name in ["run_animal", "run_xsports", "run_industry", "run_surgery"]:
        assert module_name in text
    assert "process_single_task_with_agents" in text
    assert "process_single_task" in text
    assert "SURGERY_EXPERT_PROMPT" in text
