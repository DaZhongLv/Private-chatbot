# tests/test_services.py

import requests
from private_chatbot.services import get_ollama_models


def test_get_ollama_models_success(mocker):
    # Arrange: mock response object
    mock_response = mocker.Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "models": [
            {"name": "llama3:8b", "modified_at": "...", "size": 4769434144},
            {"name": "phi3:latest", "modified_at": "...", "size": 2323232323},
        ]
    }

    # Patch: patch requests.get where it's used (in services module)
    mocker.patch("private_chatbot.services.requests.get", return_value=mock_response)

    # Act
    models = get_ollama_models()

    # Assert
    assert models == ["llama3:8b", "phi3:latest"]


def test_get_ollama_models_failure(mocker):
    # Arrange: requests.get raises a RequestException
    mocker.patch(
        "private_chatbot.services.requests.get",
        side_effect=requests.exceptions.RequestException("boom"),
    )

    # Act
    models = get_ollama_models()

    # Assert
    assert models == []

