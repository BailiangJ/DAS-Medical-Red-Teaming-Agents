def test_package_imports():
    import med_red_team

    assert med_red_team.__version__.startswith("2.")
