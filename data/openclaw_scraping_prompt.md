# OpenClaw Scraping Task: Parkourtheory.com

## Objective
Scrape detailed trick data from https://parkourtheory.com for 1,721 parkour tricks.

## Important Context
- The site is a React SPA behind Cloudflare protection
- Direct URL navigation does NOT render content
- You MUST navigate through the SPA: load homepage → use search bar → click result
- The content only renders via client-side navigation (React Router)
- Save progress to `~/parkourtheory_detailed.json` every 10 tricks

## Strategy (this is critical — follow exactly)

1. Navigate to https://parkourtheory.com and wait 10 seconds for Cloudflare
2. For each trick name in the list below:
   a. Find the search input on the page
   b. Clear it, type the trick name, press Enter
   c. Wait for "results for" to appear in the page text (up to 15 seconds)
   d. Click the first result link (any `<a>` with href containing `/move/`)
   e. Wait for `h2.detail-header` to appear on the page (up to 20 seconds)
   f. If the page is blank/empty after 20 seconds, go back home (click the logo/home link) and retry
   g. Extract ALL data (see fields below)
   h. Click the home logo (a[href='/']) to go back
   i. Repeat for next trick

3. If search stops working (session expired):
   - Navigate to homepage again
   - Wait 10 seconds
   - Resume from where you left off

## Data to extract from each trick page

For each trick, extract these fields by reading the page content:

- **type**: Text after the bold "Type:" label (e.g., "Flip/Twist", "Roll", "Vault")
- **pronunciation**: Text after "Pronunciation:" (e.g., "[bak -duh-buhl -fool]")
- **description**: Text after "Description:" (e.g., "A double twisting Back Flip.")
- **video_url**: The `src` attribute of any `<source>` tag with "cloudflarestream" in the URL
- **video_source**: Text after "Source:" (performer name)
- **video_time**: Text after "Time:" (timestamp)
- **prerequisites**: Trick names listed under the "Prerequisite" heading
- **subsequents**: Trick names listed under the "Subsequent" heading
- **explore_related**: Trick names in the "Explore" sidebar

## Output format

Save to `~/parkourtheory_detailed.json` as a JSON array:
```json
[
  {
    "name": "Back Double Full",
    "type": "Flip/Twist",
    "pronunciation": "[bak -duh-buhl -fool]",
    "description": "A double twisting Back Flip.",
    "video_url": "https://customer-b2cflnmq20t5mr78.cloudflarestream.com/...",
    "video_source": "Art Rambo",
    "video_time": "",
    "prerequisites": ["Back One and a Half Full"],
    "subsequents": ["Back Full Unwind Full", "Back Two and a Half Full"],
    "explore_related": ["Wall Double Full", "Side Full-Down"],
    "url": "https://parkourtheory.com/move/back_double_full"
  }
]
```

## LOAD EXISTING PROGRESS FIRST
Before starting, load `~/parkourtheory_detailed.json` if it exists.
Skip any trick that already has a "type" or "description" field.
Only scrape tricks that are missing or have an "error" field.

## Trick names to scrape

The complete list of 1,837 trick names is in `~/parkourtheory_tricks.json`.
116 are already done. Scrape the remaining 1,721.
