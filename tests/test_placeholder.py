def test_scaffold_imports() -> None:
    import copilot

    assert "features" in copilot.__all__
