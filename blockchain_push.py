from web3 import Web3

print("🔗 Initializing Web3 Blockchain Connection...")

# 1. Connect to Ganache Blockchain Node
w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545')) 

# Check the connection 
if w3.is_connected():
    print("✅ Connected to the local Ganache Network!")
else:
    print("❌ Failed to connect. Is 'npx ganache' running in your other terminal?")
    exit()

# 2. Credentials (Paste from Ganache & Remix)
my_wallet_address = Web3.to_checksum_address('0x0B65FfaE0D210f0fb0175Fec358BF1e8730C8BD3') 
private_key = 'YOUR_PRIVATE_KEY_HERE'
contract_address = Web3.to_checksum_address('0x2F6437dbDdDd3388E75bdBf4A7440f5d3F454065') 

# 3. The ABI
contract_abi = [
	{
		"inputs": [],
		"stateMutability": "nonpayable",
		"type": "constructor"
	},
	{
		"anonymous": False,
		"inputs": [
			{
				"indexed": True,
				"internalType": "string",
				"name": "playerId",
				"type": "string"
			},
			{
				"indexed": True,
				"internalType": "string",
				"name": "dataHash",
				"type": "string"
			},
			{
				"indexed": False,
				"internalType": "uint256",
				"name": "timestamp",
				"type": "uint256"
			}
		],
		"name": "DataStamped",
		"type": "event"
	},
	{
		"inputs": [
			{
				"internalType": "string",
				"name": "_playerId",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "_dataHash",
				"type": "string"
			}
		],
		"name": "logDataHash",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"inputs": [],
		"name": "owner",
		"outputs": [
			{
				"internalType": "address",
				"name": "",
				"type": "address"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "string",
				"name": "_playerId",
				"type": "string"
			}
		],
		"name": "verifyData",
		"outputs": [
			{
				"internalType": "string",
				"name": "",
				"type": "string"
			},
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			}
		],
		"stateMutability": "view",
		"type": "function"
	}
]
# 4. Initialize the Smart Contract in Python
passport_contract = w3.eth.contract(address=contract_address, abi=contract_abi)

def stamp_data_on_chain(player_id, file_hash):
    print(f"\n🔐 Preparing to stamp data for: {player_id}")
    print(f"📄 Hash to secure: {file_hash}")
    
    # Get the current transaction count (Nonce) for your wallet
    nonce = w3.eth.get_transaction_count(my_wallet_address)
    
    #  Smart Contract Transaction
    transaction = passport_contract.functions.logDataHash(player_id, file_hash).build_transaction({
        'chainId': 1337, # 1337 is the default chainId 
        'gas': 2000000,
        'gasPrice': w3.to_wei('50', 'gwei'),
        'nonce': nonce,
    })

    # Cryptographically sign the transaction with Private Key
    print("✍️ Signing transaction...")
    signed_txn = w3.eth.account.sign_transaction(transaction, private_key=private_key)

    # Broadcast it to  blockchain
    print("🚀 Broadcasting to the ledger...")
    tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
    
    print(f"\n🎉 SUCCESS! Data is Immutable.")
    print(f"🧾 Transaction Receipt (TxHash): {w3.to_hex(tx_hash)}")

# --- EXECUTE THE HACK ---
target_player = "mbappe_injury_profile_2026"

target_hash = "a1b2c3d4e5f67890..." 

stamp_data_on_chain(target_player, target_hash)