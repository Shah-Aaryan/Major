"""
Unit tests for Merkle tree, HashAnchoringService, and AuditLogger pluggable backend.

Verifies:
1. MerkleTree root computation and proof verification.
2. HashAnchoringService batching and proof storage.
3. AuditLogger with audit_backend="blockchain" / "hash_anchoring".
"""

import os
import shutil
import tempfile
import pytest
from audit.hash_anchoring import MerkleTree, HashAnchoringService
from audit.audit_logger import AuditLogger, AuditEventType


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
            audit_backend="blockchain"
        )
        assert logger.anchoring_service is not None
        
        logger.log_event(
            AuditEventType.PARAMETER_CHANGE,
            {"param": "rsi_lookback", "old": 14, "new": 20},
            explanation="Optimized RSI lookback"
        )
        logger.close()
        
        # Verify anchors directory has records
        anchor_dir = os.path.join(temp_dir, "anchors")
        assert os.path.exists(anchor_dir)
    finally:
        shutil.rmtree(temp_dir)
