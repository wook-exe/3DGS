from src.feature_flags import evaluate_features, is_feature_enabled


def test_feature_flag_uses_default_value():
    assert is_feature_enabled("model_status_sidebar", env={}) is True
    assert is_feature_enabled("enhanced_dashboard_cards", env={}) is False


def test_feature_flag_can_be_enabled_by_environment_variable():
    env = {"FF_ENHANCED_DASHBOARD_CARDS": "true"}

    assert is_feature_enabled("enhanced_dashboard_cards", env=env) is True


def test_feature_flag_can_target_user_ids():
    env = {"FF_CANARY_RELEASE_PANEL_USERS": "student-001,qa-user"}

    assert is_feature_enabled("canary_release_panel", user_id="student-001", env=env) is True
    assert is_feature_enabled("canary_release_panel", user_id="student-999", env=env) is False


def test_evaluate_features_returns_all_flags():
    features = evaluate_features(user_id="student-001", env={})

    assert {"enhanced_dashboard_cards", "model_status_sidebar"} <= set(features)
