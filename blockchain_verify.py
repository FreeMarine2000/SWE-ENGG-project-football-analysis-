from web3 import Web3

print("🔍 Initializing Scout Verification Protocol...")

# 1. Connect to the Network
w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545')) 

# 2. The Contract Address and ABI 
contract_address = Web3.to_checksum_address('0x2F6437dbDdDd3388E75bdBf4A7440f5d3F454065')

contract_abi = [
    {
        "inputs": [{"internalType": "string", "name": "_playerId", "type": "string"}],
        "name": "verifyData",
        "outputs": [
            {"internalType": "string", "name": "", "type": "string"},
            {"internalType": "uint256", "name": "", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

# 3. Initialize Contract
passport_contract = w3.eth.contract(address=contract_address, abi=contract_abi)

# 4. Ask the Blockchain for the Truth
target_player = "mbappe_injury_profile_2026"
print(f"📡 Querying the ledger for: {target_player}...\n")

try:
    result = passport_contract.functions.verifyData(target_player).call()
    
    saved_hash = result[0]
    timestamp = result[1]
    
    print(f"✅ VERIFIED ON-CHAIN:")
    print(f"📄 Authentic Hash: {saved_hash}")
    print(f"🕒 Unix Timestamp: {timestamp}")
except Exception as e:
    print(f"❌ Error: {e}")