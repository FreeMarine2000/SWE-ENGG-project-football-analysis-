import os
import base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

print("🛡️ Initializing Post-Quantum Defense Protocol (AES-256-GCM)...")

# 1. Generate 256-bit Quantum-Resistant Key
key = AESGCM.generate_key(bit_length=256)
encoded_key = base64.b64encode(key).decode('utf-8')
print(f"\n🔑 YOUR MASTER DECRYPTION KEY (Save this!):\n{encoded_key}\n")

# Initialize cipher
aesgcm = AESGCM(key)

# 2. Define target files
input_csv = "new-players-data-full.csv" # Change this to your exact CSV filename
encrypted_output = "LOCKED_player_data.enc"

def encrypt_dataset():
    try:
        # Read CSV data
        with open(input_csv, 'rb') as file:
            raw_data = file.read()
            
        print(f"📄 Read {len(raw_data)} bytes from {input_csv}...")

        # Generate a secure 96-bit nonce 
        nonce = os.urandom(12)
        
        # encrypt the data
        encrypted_data = aesgcm.encrypt(nonce, raw_data, None)
        
        # write the nonce + encrypted data to a new file
        with open(encrypted_output, 'wb') as file:
            file.write(nonce + encrypted_data)
            
        print(f"✅ SUCCESS: Dataset encrypted and locked against quantum brute-force.")
        print(f"🔒 Output File: {encrypted_output}")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {input_csv}. Check the filename.")

# Execute the lockdown
encrypt_dataset()