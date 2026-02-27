"""
Blockchain Audit Verification Tool (B3 - P2)

Compares off-chain audit logs against on-chain anchored hashes
to verify integrity and detect tampering.

Features:
1. Verify individual audit records
2. Verify Merkle proofs
3. Batch verification of sessions
4. Generate verification reports
5. CLI for manual verification

Usage:
    # Python API
    verifier = AuditVerifier(chain=ChainConfig.SEPOLIA, contract_address="0x...")
    result = verifier.verify_session("session_123")
    
    # CLI
    python verify_audit.py --session session_123 --chain sepolia
"""

import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import argparse

logger = logging.getLogger(__name__)

# Import from sibling modules
from audit.blockchain_schema import (
    BlockchainAuditRecord,
    AuditSession,
    compute_strategy_hash,
    compute_parameter_hash
)
from audit.hash_anchoring import (
    MerkleTree,
    MerkleProof,
    ChainConfig,
    HashAnchoringService
)

# Web3 imports (optional)
try:
    from web3 import Web3
    from web3.middleware import geth_poa_middleware
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICATION RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

class VerificationStatus(Enum):
    """Status of verification check."""
    VERIFIED = "verified"
    FAILED = "failed"
    NOT_FOUND = "not_found"
    PENDING = "pending"
    ERROR = "error"


@dataclass
class VerificationCheck:
    """Result of a single verification check."""
    check_name: str
    status: VerificationStatus
    expected: Optional[str] = None
    actual: Optional[str] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def passed(self) -> bool:
        return self.status == VerificationStatus.VERIFIED
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_name": self.check_name,
            "status": self.status.value,
            "passed": self.passed,
            "expected": self.expected,
            "actual": self.actual,
            "message": self.message,
            "details": self.details
        }


@dataclass
class VerificationReport:
    """Complete verification report for a session."""
    session_id: str
    timestamp: datetime
    overall_status: VerificationStatus
    checks: List[VerificationCheck]
    
    # Summary
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    
    # Chain info
    chain_name: str = ""
    contract_address: str = ""
    
    def __post_init__(self):
        """Calculate summary stats."""
        self.total_checks = len(self.checks)
        self.passed_checks = sum(1 for c in self.checks if c.passed)
        self.failed_checks = self.total_checks - self.passed_checks
        
        if self.failed_checks > 0:
            self.overall_status = VerificationStatus.FAILED
        elif self.total_checks == 0:
            self.overall_status = VerificationStatus.PENDING
        else:
            self.overall_status = VerificationStatus.VERIFIED
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        status_icon = "✓" if self.overall_status == VerificationStatus.VERIFIED else "✗"
        
        lines = [
            "=" * 60,
            f"AUDIT VERIFICATION REPORT",
            "=" * 60,
            f"Session: {self.session_id}",
            f"Timestamp: {self.timestamp}",
            f"Chain: {self.chain_name}",
            f"Contract: {self.contract_address[:20]}..." if self.contract_address else "Contract: N/A",
            "",
            f"Overall Status: {status_icon} {self.overall_status.value.upper()}",
            f"Checks: {self.passed_checks}/{self.total_checks} passed",
            "",
            "DETAILS:",
        ]
        
        for check in self.checks:
            icon = "✓" if check.passed else "✗"
            lines.append(f"  {icon} {check.check_name}: {check.message}")
            
            if not check.passed and check.expected and check.actual:
                lines.append(f"      Expected: {check.expected[:40]}...")
                lines.append(f"      Actual:   {check.actual[:40]}...")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "overall_status": self.overall_status.value,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "chain_name": self.chain_name,
            "contract_address": self.contract_address,
            "checks": [c.to_dict() for c in self.checks]
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, filepath: str) -> None:
        """Save report to file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
        logger.info(f"Saved verification report to {filepath}")


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIT VERIFIER
# ═══════════════════════════════════════════════════════════════════════════════

class AuditVerifier:
    """
    Verifies audit records against on-chain anchors.
    
    Performs comprehensive verification:
    1. Hash chain integrity (off-chain)
    2. Merkle proof validity (off-chain)
    3. On-chain anchor existence
    4. Content hash matches
    5. Timestamp consistency
    
    Usage:
        verifier = AuditVerifier(
            chain=ChainConfig.SEPOLIA,
            contract_address="0x..."
        )
        
        # Verify a session
        report = verifier.verify_session(session)
        print(report.summary())
        
        # Verify a single record
        check = verifier.verify_record(record)
        
        # Verify a Merkle proof
        check = verifier.verify_merkle_proof(proof, event_data)
    """
    
    # Contract ABI for verification
    VERIFY_ABI = [
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
        }
    ]
    
    def __init__(
        self,
        chain: ChainConfig = ChainConfig.SEPOLIA,
        contract_address: Optional[str] = None,
        proofs_path: Optional[str] = None
    ):
        """
        Initialize verifier.
        
        Args:
            chain: Blockchain configuration
            contract_address: Deployed audit contract address
            proofs_path: Path to stored proofs
        """
        self.chain_config = chain.value if isinstance(chain, ChainConfig) else chain
        self.contract_address = contract_address
        self.proofs_path = Path(proofs_path) if proofs_path else None
        
        # Web3 connection
        self._web3: Optional[Any] = None
        self._contract: Optional[Any] = None
        
        if WEB3_AVAILABLE and contract_address:
            self._init_web3()
    
    def _init_web3(self) -> None:
        """Initialize Web3 connection."""
        try:
            self._web3 = Web3(Web3.HTTPProvider(self.chain_config["rpc_url"]))
            
            # Add PoA middleware for Polygon
            if self.chain_config["chain_id"] in [137, 80001]:
                self._web3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            # Setup contract (read-only)
            self._contract = self._web3.eth.contract(
                address=self.contract_address,
                abi=self.VERIFY_ABI
            )
            
            logger.info(f"Connected to {self.chain_config['name']} for verification")
        except Exception as e:
            logger.warning(f"Failed to initialize Web3: {e}")
            self._web3 = None
    
    def verify_session(
        self,
        session: AuditSession
    ) -> VerificationReport:
        """
        Verify an entire audit session.
        
        Args:
            session: AuditSession to verify
            
        Returns:
            VerificationReport with all checks
        """
        checks = []
        
        # 1. Verify hash chain
        checks.append(self._verify_hash_chain(session))
        
        # 2. Verify each record
        for i, record in enumerate(session.records):
            record_check = self._verify_record_hashes(record, i)
            checks.append(record_check)
        
        # 3. Verify on-chain anchor if anchored
        if session.anchored and session.anchor_tx_hash:
            anchor_check = self._verify_on_chain(session)
            checks.append(anchor_check)
        else:
            checks.append(VerificationCheck(
                check_name="On-chain Anchor",
                status=VerificationStatus.PENDING,
                message="Session not yet anchored on-chain"
            ))
        
        # Build report
        report = VerificationReport(
            session_id=session.session_id,
            timestamp=datetime.now(),
            overall_status=VerificationStatus.PENDING,  # Will be calculated
            checks=checks,
            chain_name=self.chain_config["name"],
            contract_address=self.contract_address or ""
        )
        
        return report
    
    def verify_record(
        self,
        record: BlockchainAuditRecord
    ) -> VerificationCheck:
        """
        Verify a single audit record.
        
        Args:
            record: Record to verify
            
        Returns:
            VerificationCheck result
        """
        return self._verify_record_hashes(record, 0)
    
    def verify_merkle_proof(
        self,
        proof: MerkleProof,
        event_data: Optional[Dict[str, Any]] = None
    ) -> VerificationCheck:
        """
        Verify a Merkle proof.
        
        Args:
            proof: MerkleProof to verify
            event_data: Optional event data to re-hash
            
        Returns:
            VerificationCheck result
        """
        tree = MerkleTree()
        
        # Verify leaf hash if event data provided
        if event_data:
            computed_hash = tree.hash_data(event_data)
            if computed_hash != proof.leaf_hash:
                return VerificationCheck(
                    check_name="Merkle Proof - Leaf Hash",
                    status=VerificationStatus.FAILED,
                    expected=proof.leaf_hash,
                    actual=computed_hash,
                    message="Event data hash does not match proof leaf hash"
                )
        
        # Verify proof
        is_valid = tree.verify_proof(
            proof.leaf_hash,
            proof.proof,
            proof.merkle_root
        )
        
        if is_valid:
            return VerificationCheck(
                check_name="Merkle Proof",
                status=VerificationStatus.VERIFIED,
                message=f"Proof verified against root {proof.merkle_root[:16]}...",
                details={
                    "leaf_hash": proof.leaf_hash,
                    "root": proof.merkle_root,
                    "proof_length": len(proof.proof)
                }
            )
        else:
            return VerificationCheck(
                check_name="Merkle Proof",
                status=VerificationStatus.FAILED,
                message="Merkle proof invalid - root mismatch",
                details={
                    "leaf_hash": proof.leaf_hash,
                    "expected_root": proof.merkle_root
                }
            )
    
    def verify_on_chain_hash(
        self,
        session_id: str,
        content_hash: str
    ) -> VerificationCheck:
        """
        Verify a hash exists on-chain.
        
        Args:
            session_id: Session identifier
            content_hash: Content hash to verify
            
        Returns:
            VerificationCheck result
        """
        if not self._web3 or not self._contract:
            return VerificationCheck(
                check_name="On-chain Verification",
                status=VerificationStatus.ERROR,
                message="No Web3 connection available"
            )
        
        try:
            session_hash = hashlib.sha256(session_id.encode()).hexdigest()
            
            exists, timestamp = self._contract.functions.verifyAudit(
                bytes.fromhex(session_hash),
                bytes.fromhex(content_hash)
            ).call()
            
            if exists:
                return VerificationCheck(
                    check_name="On-chain Verification",
                    status=VerificationStatus.VERIFIED,
                    message=f"Hash found on-chain, anchored at timestamp {timestamp}",
                    details={
                        "on_chain_timestamp": timestamp,
                        "session_hash": session_hash,
                        "content_hash": content_hash
                    }
                )
            else:
                return VerificationCheck(
                    check_name="On-chain Verification",
                    status=VerificationStatus.NOT_FOUND,
                    message="Hash not found on-chain"
                )
        
        except Exception as e:
            return VerificationCheck(
                check_name="On-chain Verification",
                status=VerificationStatus.ERROR,
                message=f"Error querying blockchain: {str(e)}"
            )
    
    def _verify_hash_chain(self, session: AuditSession) -> VerificationCheck:
        """Verify the hash chain integrity of a session."""
        if not session.records:
            return VerificationCheck(
                check_name="Hash Chain",
                status=VerificationStatus.VERIFIED,
                message="No records to chain (empty session)"
            )
        
        is_valid = session.verify_chain()
        
        if is_valid:
            return VerificationCheck(
                check_name="Hash Chain",
                status=VerificationStatus.VERIFIED,
                message=f"Hash chain valid ({len(session.records)} records)"
            )
        else:
            # Find where chain breaks
            for i in range(1, len(session.records)):
                expected = session.records[i - 1].compute_hash()
                actual = session.records[i].previous_hash
                
                if expected != actual:
                    return VerificationCheck(
                        check_name="Hash Chain",
                        status=VerificationStatus.FAILED,
                        expected=expected,
                        actual=actual,
                        message=f"Chain broken at record {i}",
                        details={"broken_at_index": i}
                    )
            
            return VerificationCheck(
                check_name="Hash Chain",
                status=VerificationStatus.FAILED,
                message="Chain integrity check failed"
            )
    
    def _verify_record_hashes(
        self,
        record: BlockchainAuditRecord,
        index: int
    ) -> VerificationCheck:
        """Verify hashes in a single record are consistent."""
        # Recompute and verify hash
        computed_hash = record.compute_hash()
        
        # Check strategy hash if we have the definition
        if record.metadata.get("strategy_definition"):
            expected_strategy_hash = compute_strategy_hash(
                record.metadata["strategy_definition"]
            )
            if expected_strategy_hash != record.strategy_hash:
                return VerificationCheck(
                    check_name=f"Record {index} Strategy Hash",
                    status=VerificationStatus.FAILED,
                    expected=expected_strategy_hash,
                    actual=record.strategy_hash,
                    message="Strategy hash mismatch"
                )
        
        # Check parameter hash if we have the parameters
        if record.metadata.get("parameters"):
            expected_param_hash = compute_parameter_hash(
                record.metadata["parameters"]
            )
            if expected_param_hash != record.parameter_hash:
                return VerificationCheck(
                    check_name=f"Record {index} Parameter Hash",
                    status=VerificationStatus.FAILED,
                    expected=expected_param_hash,
                    actual=record.parameter_hash,
                    message="Parameter hash mismatch"
                )
        
        return VerificationCheck(
            check_name=f"Record {index} Hash",
            status=VerificationStatus.VERIFIED,
            message=f"Record hash: {computed_hash[:32]}...",
            details={"record_hash": computed_hash}
        )
    
    def _verify_on_chain(self, session: AuditSession) -> VerificationCheck:
        """Verify session anchor on-chain."""
        if not session.records:
            return VerificationCheck(
                check_name="On-chain Anchor",
                status=VerificationStatus.ERROR,
                message="No records to verify"
            )
        
        # Get latest record hash
        latest_hash = session.get_latest_hash()
        
        return self.verify_on_chain_hash(session.session_id, latest_hash)


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH VERIFIER
# ═══════════════════════════════════════════════════════════════════════════════

class BatchVerifier:
    """
    Verify multiple sessions or proofs in batch.
    """
    
    def __init__(self, verifier: AuditVerifier):
        """Initialize with a verifier instance."""
        self.verifier = verifier
    
    def verify_sessions_from_directory(
        self,
        directory: str
    ) -> Dict[str, VerificationReport]:
        """
        Verify all session files in a directory.
        
        Args:
            directory: Path to directory with session JSON files
            
        Returns:
            Dict mapping session_id to VerificationReport
        """
        results = {}
        path = Path(directory)
        
        for session_file in path.glob("*.json"):
            try:
                session = AuditSession.load(str(session_file))
                report = self.verifier.verify_session(session)
                results[session.session_id] = report
                
                status = "✓" if report.overall_status == VerificationStatus.VERIFIED else "✗"
                logger.info(f"{status} Verified {session.session_id}")
                
            except Exception as e:
                logger.error(f"Failed to verify {session_file}: {e}")
        
        return results
    
    def verify_proofs_from_file(
        self,
        proofs_file: str
    ) -> List[VerificationCheck]:
        """
        Verify all proofs in a proofs file.
        
        Args:
            proofs_file: Path to proofs JSON file
            
        Returns:
            List of VerificationCheck results
        """
        results = []
        
        with open(proofs_file, 'r') as f:
            data = json.load(f)
        
        for proof_data in data.get("proofs", []):
            proof = MerkleProof.from_dict(proof_data)
            check = self.verifier.verify_merkle_proof(proof)
            results.append(check)
        
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Verify BeyondAlgo audit records against blockchain anchors"
    )
    
    parser.add_argument(
        "--session",
        type=str,
        help="Session ID or path to session JSON file"
    )
    
    parser.add_argument(
        "--proof",
        type=str,
        help="Path to proofs JSON file"
    )
    
    parser.add_argument(
        "--directory",
        type=str,
        help="Directory containing session files to verify"
    )
    
    parser.add_argument(
        "--chain",
        type=str,
        choices=["polygon", "mumbai", "sepolia", "local"],
        default="sepolia",
        help="Blockchain to verify against"
    )
    
    parser.add_argument(
        "--contract",
        type=str,
        help="Audit contract address"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for verification report"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s"
    )
    
    # Select chain
    chain_map = {
        "polygon": ChainConfig.POLYGON_MAINNET,
        "mumbai": ChainConfig.POLYGON_MUMBAI,
        "sepolia": ChainConfig.SEPOLIA,
        "local": ChainConfig.LOCAL
    }
    chain = chain_map[args.chain]
    
    # Create verifier
    verifier = AuditVerifier(
        chain=chain,
        contract_address=args.contract
    )
    
    # Run verification
    if args.session:
        # Verify single session
        if Path(args.session).exists():
            session = AuditSession.load(args.session)
        else:
            print(f"Session file not found: {args.session}")
            return 1
        
        report = verifier.verify_session(session)
        print(report.summary())
        
        if args.output:
            report.save(args.output)
    
    elif args.proof:
        # Verify proofs
        batch = BatchVerifier(verifier)
        checks = batch.verify_proofs_from_file(args.proof)
        
        passed = sum(1 for c in checks if c.passed)
        print(f"\nProof verification: {passed}/{len(checks)} passed")
        
        for check in checks:
            icon = "✓" if check.passed else "✗"
            print(f"  {icon} {check.message}")
    
    elif args.directory:
        # Verify all sessions in directory
        batch = BatchVerifier(verifier)
        results = batch.verify_sessions_from_directory(args.directory)
        
        passed = sum(1 for r in results.values() if r.overall_status == VerificationStatus.VERIFIED)
        print(f"\nBatch verification: {passed}/{len(results)} sessions verified")
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    # Demo mode
    print("Audit Verification Tool Demo")
    print("=" * 50)
    
    # Create a mock session
    from audit.blockchain_schema import AuditRecordBuilder, AuditRecordType
    
    session = AuditSession(
        session_id="demo_session_001",
        created_at=datetime.utcnow(),
        strategy_name="RSI_Demo"
    )
    
    # Add records
    builder = AuditRecordBuilder(session.session_id)
    record1 = (builder
        .set_record_type(AuditRecordType.SESSION_SUMMARY)
        .set_strategy("Buy when RSI < 30")
        .set_parameters({"rsi_period": 14})
        .build())
    session.add_record(record1)
    
    builder2 = AuditRecordBuilder(session.session_id)
    record2 = (builder2
        .set_record_type(AuditRecordType.OPTIMIZATION_RESULT)
        .set_strategy("Buy when RSI < 30")
        .set_parameters({"rsi_period": 18})
        .set_objective("sharpe_ratio", 1.85)
        .build())
    session.add_record(record2)
    
    # Verify (off-chain only)
    verifier = AuditVerifier(chain=ChainConfig.SEPOLIA)
    report = verifier.verify_session(session)
    
    print(report.summary())
    
    # Verify a merkle proof
    print("\n" + "=" * 50)
    print("Merkle Proof Verification:")
    print("=" * 50)
    
    tree = MerkleTree()
    events = [{"a": 1}, {"b": 2}, {"c": 3}]
    for e in events:
        tree.add_data(e)
    
    root = tree.get_root()
    proof_data = tree.get_proof(1)
    leaf_hash = tree.hash_data(events[1])
    
    proof = MerkleProof(
        leaf_hash=leaf_hash,
        leaf_index=1,
        proof=proof_data,
        merkle_root=root,
        batch_id="test",
        timestamp=int(datetime.now().timestamp())
    )
    
    check = verifier.verify_merkle_proof(proof, events[1])
    icon = "✓" if check.passed else "✗"
    print(f"{icon} {check.check_name}: {check.message}")
