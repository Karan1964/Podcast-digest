# Podcast Digest

Daily Telegram digest of new videos from Indian stock-market YouTube channels.
Each stock-market video is transcribed, mined for every stock, theme, framework,
number and lesson, completeness-checked, and sent to Telegram with timestamp links
and the full transcript as a file. Personal-finance videos get a one-line mention.

## Setup (one time)

1. **Secrets** (repo → Settings → Secrets and variables → Actions → New repository secret)
   - `GEMINI_API_KEY`: a new key from Google AI Studio, in a project **with billing off**
   - `TELEGRAM_BOT_TOKEN`: a new bot from @BotFather
   - `TELEGRAM_CHAT_ID`: your chat ID (message the bot once, then open
     `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy `chat.id`)
2. **Actions** tab → Podcast Digest → Run workflow (first run looks back 48 hours).
3. After that it runs daily at about 7:37 AM IST.

## Change channels

Edit `channels.json`: add or remove `{"handle": "...", "name": "..."}`.

## Free-tier design

- Transcription and the completeness check use Flash-Lite (about 500 requests a day free),
  20-minute chunks at low video resolution, 40 seconds apart (tokens-per-minute limit).
- Extraction uses Flash (about 20 requests a day free), one call per video,
  with fallback models if one is out of quota.
- At most 7 hours of video a day (the free limit is 8). Extra videos are queued for the next day.

## Safety (public repo)

- Runs only on schedule or manual trigger; no pull-request triggers.
- Logs show only counts, video IDs and titles. Summaries go only to Telegram.
- `state.json` holds only public video IDs and channel IDs.
