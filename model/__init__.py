from model.flow_matching import RiemannianFlowDet


def build_model(config) -> RiemannianFlowDet:
    """
    Purpose: Instantiate RiemannianFlowDet from a config object or dict.
    Inputs:
        config: dict or namespace with optional keys:
                hidden_dim, num_layers, num_heads, mlp_ratio,
                num_queries, backbone_pretrained, trajectory
    Outputs:
        model: RiemannianFlowDet
    """
    def _get(key, default):
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    return RiemannianFlowDet(
        dim                 = _get("hidden_dim",          256),
        depth               = _get("num_layers",          6),
        num_heads           = _get("num_heads",           8),
        mlp_ratio           = _get("mlp_ratio",           4),
        num_queries         = _get("num_queries",         300),
        backbone_pretrained = _get("backbone_pretrained", True),
        backbone_type       = _get("backbone_type",       "fpn"),
        dinov2_model        = _get("dinov2_model",        "dinov2_vits14"),
        dinov2_freeze       = _get("dinov2_freeze",       False),
        trajectory_type     = _get("trajectory",          "riemannian"),
        ot_coupling         = _get("ot_coupling",         False),
    )
