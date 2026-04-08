"""
Core POPO algorithm implementation.

Modules:
    config      - POPOConfig (extends GRPOConfig)
    trainer     - POPOTrainer (extends GRPOTrainer)
    loss        - Loss components (NLL, similarity, entropy)
    weights     - Bounded importance weight computation
    ema         - EMA target policy manager
    predictor   - Predictor MLP h_phi
    callbacks   - Training callbacks (EMA update)
"""
