import base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

print("🔓 Initializing Quantum Decryption Protocol...")

# 1. PASTE YOUR MASTER KEY HERE (The massive string from the sheild)
encoded_key = "YOUR_BASE64_KEY_HERE"
key = base64.b64decode(encoded_key)

aesgcm = AESGCM(key)

encrypted_file = "LOCKED_player_data.enc"
decrypted_output = "UNLOCKED_player_data.csv"

def decrypt_dataset():
    try:
        # Read locked file
        with open(encrypted_file, 'rb') as file:
            locked_data = file.read()
            
        #  first 12 bytes are the nonce
        nonce = locked_data[:12]
        ciphertext = locked_data[12:]
        
        # Unlock it
        decrypted_data = aesgcm.decrypt(nonce, ciphertext, None)
        
        # Write back to a readable CSV
        with open(decrypted_output, 'wb') as file:
            file.write(decrypted_data)
            
        print(f"✅ SUCCESS: Data decrypted successfully.")
        print(f"📄 Output File: {decrypted_output}")
        
    except Exception as e:
        print(f"❌ Decryption Failed. Did you use the wrong key or tamper with the file? Error: {e}")

decrypt_dataset()