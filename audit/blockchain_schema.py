"""
Blockchain Audit Schema (B1 - P1)

Defines the on-chain data structure for audit record anchoring.

Schema:
    session_id: str        - Unique session identifier
    strategy_hash: str     - SHA256 of strategy definition
    parameter_hash: str    - SHA256 of final parameters
    timestamp: int         - Unix timestamp
    objective_value: float - Primary objective achieved
    merkle_root: str       - Root of Merkle tree of all events
    
Target: Polygon PoS or Sepolia testnet (hash anchoring only)
"""

import hashlib
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class AuditRecordType(Enum):
    """Types of audit records for blockchain anchoring."""
    SESSION_SUMMARY = "session_summary"
    PARAMETER_UPDATE = "parameter_update"
    OPTIMIZATION_RESULT = "optimization_result"
    TRADE_EXECUTION = "trade_execution"
    DAILY_SUMMARY = "daily_summary"


@dataclass
class BlockchainAuditRecord:
    """
    A single audit record for blockchain anchoring.
    
    This is the canonical schema for all audit data that
    gets hashed and anchored on-chain.
    """
    # Core identifiers
    session_id: str
    record_type: AuditRecordType
    sequence_number: int  # Order within session
    
    # Content hashes
    strategy_hash: str  # SHA256 of strategy code/definition
    parameter_hash: str  # SHA256 of parameter values
    
    # Timing
    timestamp: int  # Unix timestamp (seconds)
    timestamp_iso: str  # Human-readable ISO format
    
    # Results (if applicable)
    objective_name: Optional[str] = None
    objective_value: Optional[float] = None
    
    # Aggregation
    merkle_root: Optional[str] = None  # Root of event Merkle tree
    event_count: int = 0
    
    # Verification
    previous_hash: Optional[str] = None  # Chain linkage
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def compute_hash(self) -> str:
        """
        Compute SHA256 hash of this record.
        
        The hash is computed from a canonical JSON representation
        to ensure deterministic hashing.
        """
        # Create canonical representation
        canonical = {
            "session_id": self.session_id,
            "record_type": self.record_type.value,
            "sequence_number": self.sequence_number,
            "strategy_hash": self.strategy_hash,
            "parameter_hash": self.parameter_hash,
            "timestamp": self.timestamp,
            "objective_name": self.objective_name,
            "objective_value": self.objective_value,
            "merkle_root": self.merkle_root,
            "previous_hash": self.previous_hash,
        }
        
        # Sort keys for determinism
        canonical_json = json.dumps(canonical, sort_keys=True, separators=(',', ':'))
        
        return hashlib.sha256(canonical_json.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "record_type": self.record_type.value,
            "sequence_number": self.sequence_number,
            "strategy_hash": self.strategy_hash,
            "parameter_hash": self.parameter_hash,
            "timestamp": self.timestamp,
            "timestamp_iso": self.timestamp_iso,
            "objective_name": self.objective_name,
            "objective_value": self.objective_value,
            "merkle_root": self.merkle_root,
            "event_count": self.event_count,
            "previous_hash": self.previous_hash,
            "metadata": self.metadata,
            "record_hash": self.compute_hash()
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BlockchainAuditRecord':
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            record_type=AuditRecordType(data["record_type"]),
            sequence_number=data["sequence_number"],
            strategy_hash=data["strategy_hash"],
            parameter_hash=data["parameter_hash"],
            timestamp=data["timestamp"],
            timestamp_iso=data.get("timestamp_iso", ""),
            objective_name=data.get("objective_name"),
            objective_value=data.get("objective_value"),
            merkle_root=data.get("merkle_root"),
            event_count=data.get("event_count", 0),
            previous_hash=data.get("previous_hash"),
            metadata=data.get("metadata", {})
        )


@dataclass
class OnChainAuditData:
    """
    Minimal data structure for on-chain storage.
    
    This is what gets stored on Polygon/Sepolia.
    Optimized for gas efficiency (only hashes).
    """
    session_hash: str      # SHA256 of session_id
    content_hash: str      # SHA256 of full audit record
    merkle_root: str       # Root of event Merkle tree
    timestamp: int         # Unix timestamp
    
    # Packed data (single bytes32)
    # objective_value is scaled to 18 decimals
    objective_scaled: int
    
    def to_tuple(self) -> tuple:
        """Convert to tuple for smart contract call."""
        return (
            bytes.fromhex(self.session_hash),
            bytes.fromhex(self.content_hash),
            bytes.fromhex(self.merkle_root),
            self.timestamp,
            self.objective_scaled
        )
    
    @classmethod
    def from_audit_record(cls, record: BlockchainAuditRecord) -> 'OnChainAuditData':
        """Create from audit record."""
        # Hash session_id
        session_hash = hashlib.sha256(record.session_id.encode()).hexdigest()
        
        # Content hash is the record hash
        content_hash = record.compute_hash()
        
        # Scale objective to 18 decimals (like Ethereum wei)
        objective = record.objective_value or 0
        objective_scaled = int(objective * 10**18)
        
        return cls(
            session_hash=session_hash,
            content_hash=content_hash,
            merkle_root=record.merkle_root or "0" * 64,
            timestamp=record.timestamp,
            objective_scaled=objective_scaled
        )


# ═══════════════════════════════════════════════════════════════════════════════
# HASH UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_strategy_hash(strategy_definition: str) -> str:
    """
    Compute SHA256 hash of strategy definition.
    
    Args:
        strategy_definition: Strategy code or DSL text
        
    Returns:
        Hex-encoded SHA256 hash
    """
    # Normalize whitespace
    normalized = " ".join(strategy_definition.split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def compute_parameter_hash(parameters: Dict[str, Any]) -> str:
    """
    Compute SHA256 hash of parameters.
    
    Args:
        parameters: Dictionary of parameter values
        
    Returns:
        Hex-encoded SHA256 hash
    """
    # Canonical JSON representation
    canonical = json.dumps(parameters, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()


def compute_event_hash(event: Dict[str, Any]) -> str:
    """
    Compute SHA256 hash of audit event.
    
    Args:
        event: Event dictionary
        
    Returns:
        Hex-encoded SHA256 hash
    """
    canonical = json.dumps(event, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

class AuditRecordBuilder:
    """
    Builder for creating blockchain audit records.
    
    Usage:
        builder = AuditRecordBuilder(session_id="sess_123")
        
        # Add strategy info
        builder.set_strategy("Buy when RSI < 30")
        
        # Add parameters
        builder.set_parameters({"rsi_period": 14, "rsi_threshold": 30})
        
        # Set results
        builder.set_objective("sharpe_ratio", 1.85)
        
        # Build record
        record = builder.build()
    """
    
    def __init__(self, session_id: str):
        """Initialize builder with session ID."""
        self.session_id = session_id
        self._sequence = 0
        self._strategy_definition: Optional[str] = None
        self._parameters: Dict[str, Any] = {}
        self._objective_name: Optional[str] = None
        self._objective_value: Optional[float] = None
        self._merkle_root: Optional[str] = None
        self._event_count: int = 0
        self._previous_hash: Optional[str] = None
        self._metadata: Dict[str, Any] = {}
        self._record_type: AuditRecordType = AuditRecordType.SESSION_SUMMARY
    
    def set_sequence(self, seq: int) -> 'AuditRecordBuilder':
        """Set sequence number."""
        self._sequence = seq
        return self
    
    def set_record_type(self, record_type: AuditRecordType) -> 'AuditRecordBuilder':
        """Set record type."""
        self._record_type = record_type
        return self
    
    def set_strategy(self, strategy_definition: str) -> 'AuditRecordBuilder':
        """Set strategy definition."""
        self._strategy_definition = strategy_definition
        return self
    
    def set_parameters(self, parameters: Dict[str, Any]) -> 'AuditRecordBuilder':
        """Set parameter values."""
        self._parameters = parameters
        return self
    
    def set_objective(self, name: str, value: float) -> 'AuditRecordBuilder':
        """Set objective result."""
        self._objective_name = name
        self._objective_value = value
        return self
    
    def set_merkle_root(self, root: str, event_count: int = 0) -> 'AuditRecordBuilder':
        """Set Merkle root of events."""
        self._merkle_root = root
        self._event_count = event_count
        return self
    
    def set_previous_hash(self, hash: str) -> 'AuditRecordBuilder':
        """Set previous record hash for chaining."""
        self._previous_hash = hash
        return self
    
    def add_metadata(self, key: str, value: Any) -> 'AuditRecordBuilder':
        """Add metadata field."""
        self._metadata[key] = value
        return self
    
    def build(self) -> BlockchainAuditRecord:
        """Build the audit record."""
        now = datetime.utcnow()
        
        # Compute hashes
        strategy_hash = (
            compute_strategy_hash(self._strategy_definition)
            if self._strategy_definition else "0" * 64
        )
        parameter_hash = (
            compute_parameter_hash(self._parameters)
            if self._parameters else "0" * 64
        )
        
        return BlockchainAuditRecord(
            session_id=self.session_id,
            record_type=self._record_type,
            sequence_number=self._sequence,
            strategy_hash=strategy_hash,
            parameter_hash=parameter_hash,
            timestamp=int(now.timestamp()),
            timestamp_iso=now.isoformat() + "Z",
            objective_name=self._objective_name,
            objective_value=self._objective_value,
            merkle_root=self._merkle_root,
            event_count=self._event_count,
            previous_hash=self._previous_hash,
            metadata=self._metadata
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AuditSession:
    """
    Complete audit session with all records.
    
    A session contains multiple audit records that are
    linked together via previous_hash chain.
    """
    session_id: str
    created_at: datetime
    records: List[BlockchainAuditRecord] = field(default_factory=list)
    
    # Session metadata
    strategy_name: str = ""
    user_id: str = ""
    
    # State tracking
    anchored: bool = False
    anchor_tx_hash: Optional[str] = None
    anchor_block: Optional[int] = None
    
    def add_record(self, record: BlockchainAuditRecord) -> str:
        """
        Add record to session with proper chaining.
        
        Returns:
            Hash of the added record
        """
        # Set previous hash
        if self.records:
            record.previous_hash = self.records[-1].compute_hash()
        
        # Update sequence
        record.sequence_number = len(self.records)
        
        self.records.append(record)
        return record.compute_hash()
    
    def get_latest_hash(self) -> Optional[str]:
        """Get hash of most recent record."""
        if self.records:
            return self.records[-1].compute_hash()
        return None
    
    def verify_chain(self) -> bool:
        """Verify the hash chain integrity."""
        for i in range(1, len(self.records)):
            expected_prev = self.records[i - 1].compute_hash()
            actual_prev = self.records[i].previous_hash
            
            if expected_prev != actual_prev:
                logger.error(f"Chain broken at record {i}: "
                           f"expected {expected_prev[:16]}..., got {actual_prev[:16] if actual_prev else 'None'}...")
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "strategy_name": self.strategy_name,
            "user_id": self.user_id,
            "record_count": len(self.records),
            "records": [r.to_dict() for r in self.records],
            "chain_valid": self.verify_chain(),
            "anchored": self.anchored,
            "anchor_tx_hash": self.anchor_tx_hash,
            "anchor_block": self.anchor_block
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, filepath: str) -> None:
        """Save session to JSON file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
        logger.info(f"Saved audit session to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'AuditSession':
        """Load session from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        session = cls(
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            strategy_name=data.get("strategy_name", ""),
            user_id=data.get("user_id", "")
        )
        
        for record_data in data.get("records", []):
            record = BlockchainAuditRecord.from_dict(record_data)
            session.records.append(record)
        
        session.anchored = data.get("anchored", False)
        session.anchor_tx_hash = data.get("anchor_tx_hash")
        session.anchor_block = data.get("anchor_block")
        
        return session


# ═══════════════════════════════════════════════════════════════════════════════
# SMART CONTRACT ABI (for integration)
# ═══════════════════════════════════════════════════════════════════════════════

AUDIT_CONTRACT_ABI = [
    {
        "name": "anchorAudit",
        "type": "function",
        "inputs": [
            {"name": "sessionHash", "type": "bytes32"},
            {"name": "contentHash", "type": "bytes32"},
            {"name": "merkleRoot", "type": "bytes32"},
            {"name": "timestamp", "type": "uint256"},
            {"name": "objective", "type": "int256"}  # scaled by 10^18
        ],
        "outputs": [{"name": "anchorId", "type": "uint256"}]
    },
    {
        "name": "verifyAudit",
        "type": "function",
        "inputs": [
            {"name": "sessionHash", "type": "bytes32"},
            {"name": "contentHash", "type": "bytes32"}
        ],
        "outputs": [
            {"name": "exists", "type": "bool"},
            {"name": "timestamp", "type": "uint256"}
        ]
    },
    {
        "name": "getAudit",
        "type": "function",
        "inputs": [{"name": "anchorId", "type": "uint256"}],
        "outputs": [
            {"name": "sessionHash", "type": "bytes32"},
            {"name": "contentHash", "type": "bytes32"},
            {"name": "merkleRoot", "type": "bytes32"},
            {"name": "timestamp", "type": "uint256"},
            {"name": "objective", "type": "int256"}
        ]
    },
    {
        "name": "AuditAnchored",
        "type": "event",
        "inputs": [
            {"name": "anchorId", "type": "uint256", "indexed": True},
            {"name": "sessionHash", "type": "bytes32", "indexed": True},
            {"name": "contentHash", "type": "bytes32"},
            {"name": "timestamp", "type": "uint256"}
        ]
    }
]

# Solidity contract template (for deployment)
AUDIT_CONTRACT_SOLIDITY = '''
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title BeyondAlgoAudit
 * @notice Anchors ML trading research audit records on-chain
 * @dev Only stores hashes - full data lives off-chain
 */
contract BeyondAlgoAudit {
    struct AuditRecord {
        bytes32 sessionHash;
        bytes32 contentHash;
        bytes32 merkleRoot;
        uint256 timestamp;
        int256 objective;  // scaled by 10^18
    }
    
    mapping(uint256 => AuditRecord) public audits;
    mapping(bytes32 => uint256[]) public sessionAudits;
    uint256 public auditCount;
    
    event AuditAnchored(
        uint256 indexed anchorId,
        bytes32 indexed sessionHash,
        bytes32 contentHash,
        uint256 timestamp
    );
    
    function anchorAudit(
        bytes32 sessionHash,
        bytes32 contentHash,
        bytes32 merkleRoot,
        uint256 timestamp,
        int256 objective
    ) external returns (uint256 anchorId) {
        anchorId = ++auditCount;
        
        audits[anchorId] = AuditRecord({
            sessionHash: sessionHash,
            contentHash: contentHash,
            merkleRoot: merkleRoot,
            timestamp: timestamp,
            objective: objective
        });
        
        sessionAudits[sessionHash].push(anchorId);
        
        emit AuditAnchored(anchorId, sessionHash, contentHash, timestamp);
    }
    
    function verifyAudit(
        bytes32 sessionHash,
        bytes32 contentHash
    ) external view returns (bool exists, uint256 timestamp) {
        uint256[] storage anchors = sessionAudits[sessionHash];
        
        for (uint256 i = 0; i < anchors.length; i++) {
            if (audits[anchors[i]].contentHash == contentHash) {
                return (true, audits[anchors[i]].timestamp);
            }
        }
        
        return (false, 0);
    }
    
    function getAudit(uint256 anchorId) external view returns (
        bytes32 sessionHash,
        bytes32 contentHash,
        bytes32 merkleRoot,
        uint256 timestamp,
        int256 objective
    ) {
        AuditRecord storage record = audits[anchorId];
        return (
            record.sessionHash,
            record.contentHash,
            record.merkleRoot,
            record.timestamp,
            record.objective
        );
    }
    
    function getSessionAuditCount(bytes32 sessionHash) external view returns (uint256) {
        return sessionAudits[sessionHash].length;
    }
}
'''


if __name__ == "__main__":
    # Example usage
    print("Blockchain Audit Schema Example")
    print("=" * 50)
    
    # Create a session
    session = AuditSession(
        session_id="test_session_001",
        created_at=datetime.utcnow(),
        strategy_name="RSI_MeanReversion"
    )
    
    # Build and add records
    builder = AuditRecordBuilder(session.session_id)
    
    # Initial record
    record1 = (builder
        .set_record_type(AuditRecordType.SESSION_SUMMARY)
        .set_strategy("Buy when RSI < 30, Sell when RSI > 70")
        .set_parameters({"rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70})
        .add_metadata("version", "1.0")
        .build())
    
    session.add_record(record1)
    print(f"Record 1 hash: {record1.compute_hash()[:32]}...")
    
    # Optimization result
    builder2 = AuditRecordBuilder(session.session_id)
    record2 = (builder2
        .set_record_type(AuditRecordType.OPTIMIZATION_RESULT)
        .set_strategy("Buy when RSI < 30, Sell when RSI > 70")
        .set_parameters({"rsi_period": 18, "rsi_oversold": 25, "rsi_overbought": 75})
        .set_objective("sharpe_ratio", 1.85)
        .add_metadata("optimizer", "bayesian")
        .build())
    
    session.add_record(record2)
    print(f"Record 2 hash: {record2.compute_hash()[:32]}...")
    
    # Verify chain
    print(f"\nChain valid: {session.verify_chain()}")
    
    # Show on-chain data
    on_chain = OnChainAuditData.from_audit_record(record2)
    print(f"\nOn-chain data:")
    print(f"  Session hash: {on_chain.session_hash[:32]}...")
    print(f"  Content hash: {on_chain.content_hash[:32]}...")
    print(f"  Timestamp: {on_chain.timestamp}")
    
    # Print Solidity contract
    print("\n" + "=" * 50)
    print("Solidity Contract (first 30 lines):")
    print("=" * 50)
    for i, line in enumerate(AUDIT_CONTRACT_SOLIDITY.strip().split('\n')[:30]):
        print(line)
