import pytest

from st_common.licensing import license_status
from st_inference.models import REGISTRY, available_models, create_model
from st_inference.models.base import MattingModel


def test_every_registered_model_is_licence_cleared():
    for spec in REGISTRY.values():
        assert license_status(spec.repo_id) == "allowed", spec.name
        assert license_status(spec.weights) == "allowed", spec.name


def test_registry_specs_are_complete():
    for name, spec in REGISTRY.items():
        assert spec.name == name
        assert spec.precision in {"fp16", "fp32", "amp"}
        assert spec.download["type"] in {"hf", "url"}
        if spec.download["type"] == "url":
            assert spec.download["md5"]


def test_create_model_is_lazy_and_overridable(tmp_path):
    m = create_model("birefnet_lite", models_dir=tmp_path, device="cpu", precision="fp32", input_size=(512, 512))
    assert isinstance(m, MattingModel)
    assert m.model is None and m.input_size == (512, 512) and m.precision == "fp32"
    with pytest.raises(FileNotFoundError):
        m.load()


def test_unknown_model():
    with pytest.raises(KeyError):
        create_model("rmbg2")


def test_available_models_reports_missing_weights(tmp_path):
    avail = available_models(tmp_path)
    assert set(avail) == set(REGISTRY) and not any(avail.values())


def test_blocked_weights_refused_even_if_path_is_renamed(tmp_path):
    from st_common.licensing import LicenseViolation

    m = create_model("birefnet_lite", models_dir=tmp_path, device="cpu")
    m.weights_path = tmp_path / "RMBG-2.0"
    m.weights_path.mkdir()
    with pytest.raises(LicenseViolation):
        m.load()
