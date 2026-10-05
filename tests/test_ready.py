"""The bootstrap can see the published recognizer checksum."""

from app.config import get_settings


def test_recognition_download_is_configured():
    settings = get_settings()
    assert settings.recognition_model.name == "medium.safetensors"
    assert settings.recognition_md5 == "e3411a453ce3b9e9efae3f8631d85762"
    assert "21788410" in settings.recognition_url
    assert settings.segmentation_model.name == "blla.mlmodel"
    assert settings.muharaf_model.name == "muharaf_rec_best.mlmodel"
