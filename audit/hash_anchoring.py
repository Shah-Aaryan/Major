"""
Hash Anchoring with Merkle Tree (B2 - P1)

Provides:
1. Merkle tree for batching audit events
2. Hash anchoring service to Polygon PoS / Sepolia
3. Batch submission with configurable frequency
4. Off-chain proof generation

Target chains:
- Polygon PoS (production)
- Sepolia testnet (development)
"""

import hashlib
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

# Web3 imports (optional)
try:
    from web3 import Web3
    from web3.middleware import geth_poa_middleware
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# MERKLE TREE
# ═══════════════════════════════════════════════════════════════════════════════

class MerkleTree:
    """
    Merkle tree implementation for audit event batching.
    
    Each leaf is the hash of an audit event. The root hash
    is anchored on-chain, while leaves and proofs are stored off-chain.
    
    Usage:
        tree = MerkleTree()
        
        # Add events
        tree.add_leaf(event1_hash)
        tree.add_leaf(event2_hash)
        
        # Get root for anchoring
        root = tree.get_root()
        
        # Generate proof for verification
        proof = tree.get_proof(0)  # Proof for first leaf
    """
    
    def __init__(self, leaves: Optional[List[str]] = None):
        """
        Initialize Merkle tree.
        
        Args:
            leaves: Optional list of leaf hashes
        """
        self._leaves: List[str] = []
        self._tree: List[List[str]] = []
        
        if leaves:
            for leaf in leaves:
                self.add_leaf(leaf)
            self._build_tree()
    
    @staticmethod
    def hash_pair(left: str, right: str) -> str:
        """Hash two nodes together."""
        combined = bytes.fromhex(left) + bytes.fromhex(right)
        return hashlib.sha256(combined).hexdigest()
    
    @staticmethod
    def hash_data(data: Any) -> str:
        """Hash arbitrary data to create a leaf."""
        if isinstance(data, dict):
            data = json.dumps(data, sort_keys=True, separators=(',', ':'), default=str)
        elif not isinstance(data, str):
            data = str(data)
        return hashlib.sha256(data.encode()).hexdigest()
    
    def add_leaf(self, leaf_hash: str) -> int:
        """
        Add a leaf to the tree.
        
        Args:
            leaf_hash: SHA256 hash of the data
            
        Returns:
            Index of the added leaf
        """
        self._leaves.append(leaf_hash)
        self._tree = []  # Invalidate tree
        return len(self._leaves) - 1
    
    def add_data(self, data: Any) -> Tuple[int, str]:
        """
        Add data to tree, computing hash automatically.
        
        Args:
            data: Data to hash and add
            
        Returns:
            Tuple of (leaf_index, leaf_hash)
        """
        leaf_hash = self.hash_data(data)
        idx = self.add_leaf(leaf_hash)
        return idx, leaf_hash
    
    def _build_tree(self) -> None:
        """Build the Merkle tree from leaves."""
        if not self._leaves:
            self._tree = []
            return
        
        # Start with leaves
        current_level = self._leaves.copy()
        self._tree = [current_level]
        
        # Build up to root
        while len(current_level) > 1:
            next_level = []
            
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                # If odd number, duplicate last node
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                next_level.append(self.hash_pair(left, right))
            
            current_level = next_level
            self._tree.append(current_level)
    
    def get_root(self) -> str:
        """
        Get the Merkle root hash.
        
        Returns:
            Root hash (64 hex chars), or zeros if empty
        """
        if not self._leaves:
            return "0" * 64
        
        if not self._tree:
            self._build_tree()
        
        return self._tree[-1][0]
    
    def get_proof(self, leaf_index: int) -> List[Tuple[str, str]]:
        """
        Generate Merkle proof for a leaf.
        
        Args:
            leaf_index: Index of the leaf
            
        Returns:
            List of (sibling_hash, position) tuples
            position is 'left' or 'right'
        """
        if not self._tree:
            self._build_tree()
        
        if leaf_index >= len(self._leaves):
            raise IndexError(f"Leaf index {leaf_index} out of range")
        
        proof = []
        idx = leaf_index
        
        for level in self._tree[:-1]:  # Exclude root level
            # Find sibling
            if idx % 2 == 0:  # Current is left
                sibling_idx = idx + 1
                position = "right"
            else:  # Current is right
                sibling_idx = idx - 1
                position = "left"
            
            # Get sibling (handle odd-length levels)
            if sibling_idx < len(level):
                sibling = level[sibling_idx]
            else:
                sibling = level[idx]  # Duplicate last
            
            proof.append((sibling, position))
            
            # Move to parent index
            idx = idx // 2
        
        return proof
    
    def verify_proof(
        self,
        leaf_hash: str,
        proof: List[Tuple[str, str]],
        root: str
    ) -> bool:
        """
        Verify a Merkle proof.
        
        Args:
            leaf_hash: Hash of the leaf data
            proof: Proof from get_proof()
            root: Expected root hash
            
        Returns:
            True if proof is valid
        """
        current = leaf_hash
        
        for sibling, position in proof:
            if position == "left":
                current = self.hash_pair(sibling, current)
            else:
                current = self.hash_pair(current, sibling)
        
        return current == root
    
    @property
    def leaf_count(self) -> int:
        """Number of leaves in tree."""
        return len(self._leaves)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize tree to dictionary."""
        if not self._tree:
            self._build_tree()
        
        return {
            "leaves": self._leaves,
            "root": self.get_root(),
            "tree_depth": len(self._tree),
            "leaf_count": len(self._leaves)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MerkleTree':
        """Deserialize tree from dictionary."""
        return cls(leaves=data.get("leaves", []))


# ═══════════════════════════════════════════════════════════════════════════════
# PROOF STORAGE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MerkleProof:
    """
    Stored proof for an audit event.
    
    Enables verification that an event was included
    in an anchored batch.
    """
    leaf_hash: str
    leaf_index: int
    proof: List[Tuple[str, str]]
    merkle_root: str
    batch_id: str
    timestamp: int
    
    # On-chain anchor info (if anchored)
    anchor_tx_hash: Optional[str] = None
    anchor_block: Optional[int] = None
    chain_id: Optional[int] = None
    
    def verify(self) -> bool:
        """Verify this proof against its root."""
        tree = MerkleTree()
        return tree.verify_proof(self.leaf_hash, self.proof, self.merkle_root)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "leaf_hash": self.leaf_hash,
            "leaf_index": self.leaf_index,
            "proof": self.proof,
            "merkle_root": self.merkle_root,
            "batch_id": self.batch_id,
            "timestamp": self.timestamp,
            "anchor_tx_hash": self.anchor_tx_hash,
            "anchor_block": self.anchor_block,
            "chain_id": self.chain_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MerkleProof':
        return cls(
            leaf_hash=data["leaf_hash"],
            leaf_index=data["leaf_index"],
            proof=[(p[0], p[1]) for p in data["proof"]],
            merkle_root=data["merkle_root"],
            batch_id=data["batch_id"],
            timestamp=data["timestamp"],
            anchor_tx_hash=data.get("anchor_tx_hash"),
            anchor_block=data.get("anchor_block"),
            chain_id=data.get("chain_id")
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CHAIN CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class ChainConfig(Enum):
    """Supported blockchain configurations."""
    
    POLYGON_MAINNET = {
        "chain_id": 137,
        "rpc_url": "https://polygon-rpc.com",
        "explorer": "https://polygonscan.com",
        "name": "Polygon PoS"
    }
    
    POLYGON_MUMBAI = {
        "chain_id": 80001,
        "rpc_url": "https://rpc-mumbai.maticvigil.com",
        "explorer": "https://mumbai.polygonscan.com",
        "name": "Polygon Mumbai"
    }
    
    SEPOLIA = {
        "chain_id": 11155111,
        "rpc_url": "https://rpc.sepolia.org",
        "explorer": "https://sepolia.etherscan.io",
        "name": "Sepolia Testnet"
    }
    
    LOCAL = {
        "chain_id": 31337,
        "rpc_url": "http://127.0.0.1:8545",
        "explorer": "",
        "name": "Local Development"
    }


# ═══════════════════════════════════════════════════════════════════════════════
# HASH ANCHORING SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class HashAnchoringService:
    """
    Service for anchoring audit hashes on-chain.
    
    Supports batching via Merkle trees for gas efficiency.
    
    Usage:
        service = HashAnchoringService(
            chain=ChainConfig.SEPOLIA,
            private_key="0x...",
            contract_address="0x..."
        )
        
        # Add events to batch
        service.add_event(event1)
        service.add_event(event2)
        
        # Anchor when batch is ready
        tx_hash = service.anchor_batch()
        
        # Get proof for verification
        proof = service.get_proof(event_hash)
    """
    
    # Contract ABI for anchoring
    ANCHOR_ABI = [
        {
            "name": "anchorAudit",
            "type": "function",
            "inputs": [
                {"name": "sessionHash", "type": "bytes32"},
                {"name": "contentHash", "type": "bytes32"},
                {"name": "merkleRoot", "type": "bytes32"},
                {"name": "timestamp", "type": "uint256"},
                {"name": "objective", "type": "int256"}
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
        }
    ]
    
    def __init__(
        self,
        chain: ChainConfig = ChainConfig.SEPOLIA,
        private_key: Optional[str] = None,
        contract_address: Optional[str] = None,
        batch_size: int = 100,
        auto_anchor: bool = False,
        storage_path: Optional[str] = None
    ):
        """
        Initialize anchoring service.
        
        Args:
            chain: Blockchain configuration
            private_key: Private key for signing (None = dry run)
            contract_address: Deployed audit contract address
            batch_size: Number of events before auto-anchor
            auto_anchor: Automatically anchor when batch is full
            storage_path: Path for proof storage
        """
        self.chain_config = chain.value if isinstance(chain, ChainConfig) else chain
        self.private_key = private_key
        self.contract_address = contract_address
        self.batch_size = batch_size
        self.auto_anchor = auto_anchor
        self.storage_path = Path(storage_path) if storage_path else None
        
        # Current batch
        self._batch_id = f"batch_{int(time.time())}"
        self._merkle_tree = MerkleTree()
        self._events: List[Dict[str, Any]] = []
        self._proofs: Dict[str, MerkleProof] = {}
        
        # Web3 connection
        self._web3: Optional[Any] = None
        self._contract: Optional[Any] = None
        self._account: Optional[str] = None
        
        if WEB3_AVAILABLE and private_key and contract_address:
            self._init_web3()
    
    def _init_web3(self) -> None:
        """Initialize Web3 connection."""
        try:
            self._web3 = Web3(Web3.HTTPProvider(self.chain_config["rpc_url"]))
            
            # Add PoA middleware for Polygon
            if self.chain_config["chain_id"] in [137, 80001]:
                self._web3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            # Setup account
            self._account = self._web3.eth.account.from_key(self.private_key)
            
            # Setup contract
            self._contract = self._web3.eth.contract(
                address=self.contract_address,
                abi=self.ANCHOR_ABI
            )
            
            logger.info(f"Connected to {self.chain_config['name']}, "
                       f"account: {self._account.address[:10]}...")
        except Exception as e:
            logger.warning(f"Failed to initialize Web3: {e}")
            self._web3 = None
    
    def add_event(
        self,
        event: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Tuple[int, str]:
        """
        Add an event to the current batch.
        
        Args:
            event: Event data to record
            session_id: Optional session ID
            
        Returns:
            Tuple of (leaf_index, leaf_hash)
        """
        # Add session_id if provided
        if session_id:
            event = {**event, "session_id": session_id}
        
        # Add to tree
        idx, leaf_hash = self._merkle_tree.add_data(event)
        self._events.append(event)
        
        logger.debug(f"Added event to batch {self._batch_id}: {leaf_hash[:16]}...")
        
        # Auto anchor if batch is full
        if self.auto_anchor and self._merkle_tree.leaf_count >= self.batch_size:
            self.anchor_batch()
        
        return idx, leaf_hash
    
    def anchor_batch(
        self,
        session_id: str = "",
        objective_value: float = 0.0
    ) -> Optional[str]:
        """
        Anchor the current batch on-chain.
        
        Args:
            session_id: Session identifier
            objective_value: Primary objective value
            
        Returns:
            Transaction hash if successful, None otherwise
        """
        if self._merkle_tree.leaf_count == 0:
            logger.warning("No events to anchor")
            return None
        
        merkle_root = self._merkle_tree.get_root()
        timestamp = int(time.time())
        
        logger.info(f"Anchoring batch {self._batch_id} with {self._merkle_tree.leaf_count} events")
        logger.info(f"Merkle root: {merkle_root[:32]}...")
        
        # Generate proofs for all events BEFORE resetting
        self._generate_proofs(merkle_root, timestamp)
        
        tx_hash = None
        block_number = None
        
        # On-chain anchor
        if self._web3 and self._contract:
            try:
                tx_hash, block_number = self._submit_anchor(
                    session_id=session_id,
                    content_hash=merkle_root,  # Use merkle root as content hash for batch
                    merkle_root=merkle_root,
                    timestamp=timestamp,
                    objective_value=objective_value
                )
                
                # Update proofs with tx info
                for proof in self._proofs.values():
                    if proof.batch_id == self._batch_id:
                        proof.anchor_tx_hash = tx_hash
                        proof.anchor_block = block_number
                        proof.chain_id = self.chain_config["chain_id"]
                
            except Exception as e:
                logger.error(f"Failed to anchor on-chain: {e}")
        else:
            logger.info("No Web3 connection - dry run only")
        
        # Save proofs
        if self.storage_path:
            self._save_proofs()
        
        # Reset for next batch
        old_batch_id = self._batch_id
        self._batch_id = f"batch_{int(time.time())}"
        self._merkle_tree = MerkleTree()
        self._events = []
        
        logger.info(f"Batch {old_batch_id} anchored. TX: {tx_hash or 'dry-run'}")
        
        return tx_hash
    
    def _submit_anchor(
        self,
        session_id: str,
        content_hash: str,
        merkle_root: str,
        timestamp: int,
        objective_value: float
    ) -> Tuple[str, int]:
        """Submit anchor transaction to blockchain."""
        # Hash session_id
        session_hash = hashlib.sha256(session_id.encode()).hexdigest() if session_id else "0" * 64
        
        # Scale objective
        objective_scaled = int(objective_value * 10**18)
        
        # Build transaction
        nonce = self._web3.eth.get_transaction_count(self._account.address)
        
        tx = self._contract.functions.anchorAudit(
            bytes.fromhex(session_hash),
            bytes.fromhex(content_hash),
            bytes.fromhex(merkle_root),
            timestamp,
            objective_scaled
        ).build_transaction({
            'chainId': self.chain_config["chain_id"],
            'gas': 200000,
            'gasPrice': self._web3.eth.gas_price,
            'nonce': nonce
        })
        
        # Sign and send
        signed = self._web3.eth.account.sign_transaction(tx, self.private_key)
        tx_hash = self._web3.eth.send_raw_transaction(signed.rawTransaction)
        
        # Wait for confirmation
        receipt = self._web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        
        return receipt.transactionHash.hex(), receipt.blockNumber
    
    def _generate_proofs(self, merkle_root: str, timestamp: int) -> None:
        """Generate Merkle proofs for all events in batch."""
        for i, event in enumerate(self._events):
            leaf_hash = MerkleTree.hash_data(event)
            proof = self._merkle_tree.get_proof(i)
            
            merkle_proof = MerkleProof(
                leaf_hash=leaf_hash,
                leaf_index=i,
                proof=proof,
                merkle_root=merkle_root,
                batch_id=self._batch_id,
                timestamp=timestamp
            )
            
            self._proofs[leaf_hash] = merkle_proof
    
    def get_proof(self, event_or_hash: Any) -> Optional[MerkleProof]:
        """
        Get proof for an event.
        
        Args:
            event_or_hash: Event dict or leaf hash
            
        Returns:
            MerkleProof if found
        """
        if isinstance(event_or_hash, dict):
            leaf_hash = MerkleTree.hash_data(event_or_hash)
        else:
            leaf_hash = event_or_hash
        
        return self._proofs.get(leaf_hash)
    
    def _save_proofs(self) -> None:
        """Save proofs to storage."""
        if not self.storage_path:
            return
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        proofs_file = self.storage_path / f"proofs_{self._batch_id}.json"
        proofs_data = {
            "batch_id": self._batch_id,
            "timestamp": int(time.time()),
            "proofs": [p.to_dict() for p in self._proofs.values()]
        }
        
        with open(proofs_file, 'w') as f:
            json.dump(proofs_data, f, indent=2)
        
        logger.info(f"Saved {len(self._proofs)} proofs to {proofs_file}")
    
    def load_proofs(self, batch_id: str) -> List[MerkleProof]:
        """Load proofs from storage."""
        if not self.storage_path:
            return []
        
        proofs_file = self.storage_path / f"proofs_{batch_id}.json"
        
        if not proofs_file.exists():
            return []
        
        with open(proofs_file, 'r') as f:
            data = json.load(f)
        
        proofs = [MerkleProof.from_dict(p) for p in data.get("proofs", [])]
        
        # Cache them
        for p in proofs:
            self._proofs[p.leaf_hash] = p
        
        return proofs
    
    @property
    def pending_count(self) -> int:
        """Number of events pending anchoring."""
        return self._merkle_tree.leaf_count
    
    @property
    def current_root(self) -> str:
        """Current Merkle root (before anchoring)."""
        return self._merkle_tree.get_root()


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════════

class BatchAnchorScheduler:
    """
    Scheduler for periodic batch anchoring.
    
    Anchors batches based on:
    - Time interval (e.g., hourly)
    - Event count threshold
    - Manual trigger
    """
    
    def __init__(
        self,
        service: HashAnchoringService,
        interval_seconds: int = 3600,  # 1 hour
        min_events: int = 10
    ):
        """
        Initialize scheduler.
        
        Args:
            service: HashAnchoringService instance
            interval_seconds: Time between anchors
            min_events: Minimum events before anchoring
        """
        self.service = service
        self.interval_seconds = interval_seconds
        self.min_events = min_events
        self._last_anchor = time.time()
    
    def check_and_anchor(
        self,
        session_id: str = "",
        force: bool = False
    ) -> Optional[str]:
        """
        Check if anchoring is needed and do it.
        
        Args:
            session_id: Session ID for anchor
            force: Force anchor regardless of thresholds
            
        Returns:
            TX hash if anchored
        """
        now = time.time()
        event_count = self.service.pending_count
        
        time_trigger = (now - self._last_anchor) >= self.interval_seconds
        count_trigger = event_count >= self.min_events
        
        should_anchor = force or (event_count > 0 and (time_trigger or count_trigger))
        
        if should_anchor:
            tx_hash = self.service.anchor_batch(session_id=session_id)
            self._last_anchor = now
            return tx_hash
        
        return None


if __name__ == "__main__":
    print("Hash Anchoring with Merkle Tree Example")
    print("=" * 50)
    
    # Create Merkle tree
    tree = MerkleTree()
    
    # Add some events
    events = [
        {"type": "signal", "action": "buy", "price": 45000},
        {"type": "signal", "action": "sell", "price": 46500},
        {"type": "param_change", "param": "rsi_period", "old": 14, "new": 18},
        {"type": "optimization", "sharpe": 1.85, "drawdown": 0.12}
    ]
    
    for event in events:
        idx, leaf_hash = tree.add_data(event)
        print(f"Added event {idx}: {leaf_hash[:32]}...")
    
    # Get root
    root = tree.get_root()
    print(f"\nMerkle root: {root}")
    
    # Generate and verify proof
    print("\nProof verification:")
    for i in range(len(events)):
        proof = tree.get_proof(i)
        leaf_hash = tree.hash_data(events[i])
        verified = tree.verify_proof(leaf_hash, proof, root)
        print(f"  Event {i}: {'✓ Valid' if verified else '✗ Invalid'}")
    
    # Demonstrate anchoring service (dry run)
    print("\n" + "=" * 50)
    print("Anchoring Service (dry run):")
    print("=" * 50)
    
    service = HashAnchoringService(
        chain=ChainConfig.SEPOLIA,
        batch_size=10
    )
    
    for event in events:
        service.add_event(event, session_id="test_session")
    
    print(f"Pending events: {service.pending_count}")
    print(f"Current root: {service.current_root[:32]}...")
    
    # Anchor (dry run - no web3 connection)
    tx = service.anchor_batch(session_id="test_session", objective_value=1.85)
    print(f"Anchor result: {tx or 'dry-run (no Web3)'}")
    
    # Get proof
    proof = service.get_proof(events[0])
    if proof:
        print(f"\nProof for event 0:")
        print(f"  Leaf hash: {proof.leaf_hash[:32]}...")
        print(f"  Root: {proof.merkle_root[:32]}...")
        print(f"  Valid: {proof.verify()}")
