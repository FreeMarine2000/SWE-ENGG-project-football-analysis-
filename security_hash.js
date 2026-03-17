const crypto = require('crypto');
const fs = require('fs');

// The file you want to secure (change this to test different leagues)
const targetFile = 'la_liga_injuries.csv';

function generateFileHash(filePath) {
    console.log(`\n Initiating Security Protocol for: ${filePath}`);
    
    // 1. Check if the file actually exists first
    if (!fs.existsSync(filePath)) {
        console.error(`❌ Error: Could not find ${filePath}. Did you run the scraper yet?`);
        return;
    }

    try {
        // 2. Read the raw data from your scraped CSV
        const fileBuffer = fs.readFileSync(filePath);

        // 3. Create a SHA-256 Hash (The exact same encryption Bitcoin uses)
        const hashSum = crypto.createHash('sha256');
        hashSum.update(fileBuffer);

        // 4. Output the digital fingerprint in hexadecimal format
        const hexHash = hashSum.digest('hex');

        console.log(`Data Integrity Fingerprint Generated!`);
        console.log(`-----------------------------------------------------`);
        console.log(` File: ${filePath}`);
        console.log(` SHA-256 Hash: ${hexHash}`);
        console.log(`-----------------------------------------------------`);
        console.log(`Save this hash! If the CSV is altered by even one byte, this hash will change completely.`);

        // Optional: Save the hash to a ledger file
        const logEntry = `${new Date().toISOString()} | ${filePath} | ${hexHash}\n`;
        fs.appendFileSync('security_ledger.txt', logEntry);

    } catch (error) {
        console.error("❌ Failed to generate hash:", error);
    }
}

// Run the function hash
generateFileHash(targetFile); 