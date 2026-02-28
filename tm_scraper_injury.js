const puppeteer = require('puppeteer');
const fs = require('fs');

// --- Configuration ---
const BASE_URL = 'https://www.transfermarkt.com';

const LEAGUES = {
    'premier_league': '/premier-league/startseite/wettbewerb/GB1',
    'la_liga': '/laliga/startseite/wettbewerb/ES1',
    'serie_a': '/serie-a/startseite/wettbewerb/IT1',
    'bundesliga': '/bundesliga/startseite/wettbewerb/L1',
    'ligue_1': '/ligue-1/startseite/wettbewerb/FR1'
};

const targetLeague = process.argv[2];

if (!targetLeague || !LEAGUES[targetLeague]) {
    console.error("❌ Please provide a valid league! Options: premier_league, la_liga, serie_a, bundesliga, ligue_1");
    console.error("Example: node tm_scraper.js premier_league");
    process.exit(1);
}

const csvFilename = `${targetLeague}_injuries.csv`;

// Helper: Random delay to mimic human behavior
const randomSleep = (min, max) => new Promise(r => setTimeout(r, Math.floor(Math.random() * (max - min + 1)) + min));

// Helper: Clean text for CSV
const cleanCSV = (text) => {
    if (!text) return '""';
    return `"${text.replace(/"/g, '""').trim()}"`;
};

// Global browser variable so the Emergency Brake can access it
let browser;

// --- EMERGENCY BRAKE (Ctrl+C Handler) ---
process.on('SIGINT', async () => {
    console.log("\n\n🛑 EMERGENCY BRAKE PULLED! (Ctrl+C detected)");
    if (browser) {
        console.log("🧹 Cleaning up and closing the hidden browser...");
        await browser.close();
    }
    console.log("✅ Browser closed. Your CSV is safe. Exiting now.");
    process.exit(0);
});

// --- MAIN SCRAPER ---
async function scrapeLeague() {
    console.log(`🕵️‍♂️ Launching browser to scrape ${targetLeague}...`);
    
    if (!fs.existsSync(csvFilename)) {
        fs.writeFileSync(csvFilename, "League,Team,Player,Season,Injury,From_Date,Until_Date,Days_Missed,Games_Missed\n");
    }

    browser = await puppeteer.launch({
        headless: "new",
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });

    const page = await browser.newPage();
    await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36');
    await page.setViewport({ width: 1366, height: 768 });

    try {
        // 1. NAVIGATE TO LEAGUE PAGE (This was missing!)
        const leagueUrl = BASE_URL + LEAGUES[targetLeague];
        console.log(`🏟️ Navigating to League Page: ${leagueUrl}`);
        await page.goto(leagueUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
        await randomSleep(2000, 4000);

        // 2. EXTRACT TEAMS
        const scrapedTeams = await page.evaluate(() => {
            const teamLinks = [];
            const rows = document.querySelectorAll('table.items td.hauptlink a');
            
            rows.forEach(a => {
                // Ensure it's a team link and ignore empty rows
                if (a.href.includes('/verein/') && a.innerText.trim() !== '') {
                    teamLinks.push({ name: a.innerText.trim(), url: a.href });
                }
            });
            return [...new Map(teamLinks.map(item => [item.url, item])).values()];
        });

        // 3. APPLY HARD CAP (18 for Bundesliga, 20 for others)
        const teamLimit = (targetLeague === 'bundesliga') ? 18 : 20;
        const teams = scrapedTeams.slice(0, teamLimit);

        console.log(`✅ Found ${teams.length} teams.`);

        if (teams.length === 0) {
            console.log("⚠️ No teams found. Transfermarkt might be blocking the connection or layout changed.");
            return;
        }

        // 4. LOOP TEAMS -> PLAYERS -> INJURIES
        for (const team of teams) {
            console.log(`\n👕 Scraping Team: ${team.name}`);
            await page.goto(team.url, { waitUntil: 'domcontentloaded', timeout: 60000 });
            await randomSleep(3000, 5000);

            const players = await page.evaluate(() => {
                const playerLinks = [];
                const rows = document.querySelectorAll('table.items td.hauptlink a');
                rows.forEach(a => {
                    if (a.href.includes('/profil/spieler/')) {
                        const injuryUrl = a.href.replace('/profil/', '/verletzungen/');
                        playerLinks.push({ name: a.innerText.trim(), url: injuryUrl });
                    }
                });
                return [...new Map(playerLinks.map(item => [item.url, item])).values()];
            });

            console.log(`  -> Found ${players.length} players. Fetching injuries...`);

            for (const player of players) {
                await page.goto(player.url, { waitUntil: 'domcontentloaded', timeout: 60000 });
                await randomSleep(2000, 4000); 

                const injuries = await page.evaluate(() => {
                    const data = [];
                    const table = document.querySelector('table.items');
                    if (!table) return data; 

                    const rows = table.querySelectorAll('tbody tr');
                    rows.forEach(row => {
                        const cols = row.querySelectorAll('td');
                        if (cols.length >= 6) {
                            data.push({
                                season: cols[0].innerText,
                                injury: cols[1].innerText,
                                from: cols[2].innerText,
                                until: cols[3].innerText,
                                days: cols[4].innerText,
                                games: cols[5].innerText
                            });
                        }
                    });
                    return data;
                });

                if (injuries.length > 0) {
                    let csvContent = "";
                    injuries.forEach(inj => {
                        const row = [
                            targetLeague,
                            team.name,
                            player.name,
                            inj.season,
                            inj.injury,
                            inj.from,
                            inj.until,
                            inj.days,
                            inj.games
                        ].map(cleanCSV).join(",") + "\n";
                        csvContent += row;
                    });
                    fs.appendFileSync(csvFilename, csvContent);
                    process.stdout.write('+'); 
                } else {
                    process.stdout.write('-'); 
                }
            }
        }
        console.log(`\n🎉 DONE! Saved all data to ${csvFilename}`);

    } catch (error) {
        console.error("\n❌ Scrape interrupted or failed:", error);
    } finally {
        if (browser) await browser.close();
    }
}

scrapeLeague();