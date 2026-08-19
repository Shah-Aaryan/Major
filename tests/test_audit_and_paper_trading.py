"""
Unit tests for paper trading state persistence and audit trail verification.
"""

import pytest
import os
import json
import hashlib
from datetime import datetime

from realtime.paper_trader import PaperTrader, PositionSide, PaperPosition
from audit.verify_audit import AuditVerifier, VerificationStatus
from audit.hash_anchoring import HashAnchoringService, ChainConfig, MerkleTree


def test_paper_trader_state_persistence(tmp_path):
    trader = PaperTrader(initial_capital=10000.0, max_positions=3)
    
    pos = PaperPosition(
        id="pos_1",
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        entry_price=50000.0,
        entry_time=datetime.now(),
        quantity=0.1,
        current_price=51000.0
    )
    trader.positions["BTCUSDT"] = pos
    trader.cash = 5000.0
    
    save_path = str(tmp_path / "paper_state.json")
    trader.save_state(save_path)
    assert os.path.exists(save_path)
    
    trader2 = PaperTrader(initial_capital=10000.0)
    trader2.load_state(save_path)
    
    assert trader2.cash == 5000.0
    assert "BTCUSDT" in trader2.positions
    assert trader2.positions["BTCUSDT"].entry_price == 50000.0
    assert trader2.positions["BTCUSDT"].quantity == 0.1


def test_audit_hash_anchoring_and_merkle():
    tree = MerkleTree()
    h1 = hashlib.sha256(b"event_1").hexdigest()
    h2 = hashlib.sha256(b"event_2").hexdigest()
    
    tree.add_leaf(h1)
    tree.add_leaf(h2)
    
    root = tree.get_root()
    assert root is not None
    assert len(root) == 64
    
    proof = tree.get_proof(0)
    assert proof is not None
    assert tree.verify_proof(h1, proof, root)

    anchor_service = HashAnchoringService(chain=ChainConfig.SEPOLIA)
    leaf = anchor_service.add_event({"session": "test_123", "action": "opt_start"})
    assert leaf is not None
