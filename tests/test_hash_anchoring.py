"""
Unit tests for Merkle tree, HashAnchoringService, and AuditLogger pluggable backend.

Verifies:
1. MerkleTree root computation and proof verification.
2. HashAnchoringService batching and proof storage.
3. AuditLogger with audit_backend="blockchain" / "hash_anchoring".
"""

from datetime import datetime
import os
import shutil
import tempfile
import pytest
from audit.hash_anchoring import MerkleTree, HashAnchoringService
from audit.audit_logger import AuditLogger, AuditEventType, SignalAudit, OptimizationAudit, ParameterChangeAudit


def test_merkle_tree_basic():
    tree = MerkleTree()
    data1 = {"event": "signal_generated", "price": 50000}
    data2 = {"event": "parameter_change", "param": "rsi_buy", "value": 30}
    
    idx1, hash1 = tree.add_data(data1)
    idx2, hash2 = tree.add_data(data2)
    
    root = tree.get_root()
    assert len(root) == 64
    
    proof1 = tree.get_proof(idx1)
    assert tree.verify_proof(hash1, proof1, root) is True


def test_hash_anchoring_service():
    temp_dir = tempfile.mkdtemp()
    try:
        service = HashAnchoringService(storage_path=temp_dir, auto_anchor=False)
        e1 = {"session_id": "test_1", "action": "buy"}
        e2 = {"session_id": "test_1", "action": "sell"}
        
        _, h1 = service.add_event(e1)
        _, h2 = service.add_event(e2)
        
        tx = service.anchor_batch()
        assert tx is None or isinstance(tx, str)
        
        proof = service.get_proof(h1)
        assert proof is not None
        assert proof.verify() is True
    finally:
        shutil.rmtree(temp_dir)


def test_audit_logger_blockchain_backend():
    temp_dir = tempfile.mkdtemp()
    try:
        logger = AuditLogger(
            output_dir=temp_dir,
            log_to_file=True,
            log_to_console=True,
            audit_backend="blockchain"
        )
        assert logger.anchoring_service is not None
        
        # Log event
        evt = logger.log_event(
            AuditEventType.PARAMETER_CHANGE,
            {"param": "rsi_lookback", "old": 14, "new": 20},
            explanation="Optimized RSI lookback"
        )
        assert evt.to_json() is not None
        
        # Log signal
        sig_audit = SignalAudit(
            timestamp=datetime.now(),
            strategy_name="rsi_mean_reversion",
            signal_type="buy",
            confidence=0.95,
            price=50000.0,
            primary_reason="RSI oversold",
            secondary_reasons=["EMA bullish trend"],
            market_condition="trending_bullish",
            parameter_source="ml"
        )
        logger.log_signal(sig_audit)
        
        # Log optimization
        opt_audit = OptimizationAudit(
            timestamp=datetime.now(),
            strategy_name="rsi_mean_reversion",
            optimizer_type="bayesian_tpe",
            n_trials=30,
            best_objective=1.95,
            elapsed_time=0.5,
            human_params={"rsi_buy_threshold": 30},
            ml_params={"rsi_buy_threshold": 25},
            parameter_changes={"rsi_buy_threshold": {"human": 30, "ml": 25}},
            improvement=0.15,
            converged=True,
            convergence_reason="Max iterations reached"
        )
        logger.log_optimization(opt_audit)
        
        # Log parameter change and regime change
        param_audit = ParameterChangeAudit(
            timestamp=datetime.now(),
            strategy_name="rsi_mean_reversion",
            parameter_name="rsi_lookback",
            old_value=14,
            new_value=20,
            change_pct=42.8,
            source="ml",
            reason="Bayesian optimization"
        )
        logger.log_parameter_change(param_audit)
        logger.log_event(
            AuditEventType.REGIME_CHANGE,
            {"old_regime": "trending_bullish", "new_regime": "ranging"},
            explanation="Regime shift detected"
        )
        
        stats = logger.get_statistics()
        assert stats["total_events"] > 0
        assert stats["session_id"] == logger.session_id
        
        # Exception handling in log_event
        logger.anchoring_service.add_event = lambda ev: (_ for _ in ()).throw(RuntimeError("Mock error"))
        logger.log_event(AuditEventType.ERROR, {"error": "test"})
        
        logger.close()
        
        # Verify anchors directory has records
        anchor_dir = os.path.join(temp_dir, "anchors")
        assert os.path.exists(anchor_dir)
    finally:
        shutil.rmtree(temp_dir)

