# Predictions in *Il 20° secolo* (Albert Robida, 1885)

Extracted via a single-call, whole-document map over the entire book (2,573 chunks,
~1.7M chars, well within Gemini Flash-Lite's 1M-token context — see
`rag/summarizer.py:whole_document`). Cost: ~$0.054. Sorted alphabetically within
each theme; a parenthetical gloss is added for invented/obscure terms.

## Transport & Infrastructure

- **Aeronautical Omnibuses** — public airships (*aeronavi-omnibus*) on fixed routes and altitudes, carrying many passengers.
- **Aero-carriages (*Aerocarrozze*)** — personal aerial vehicles, like private taxis.
- **Aero-yachts** — large, luxurious personal airships for extended travel.
- **Aerial Stations/Terminals** — boarding points atop buildings or towers (e.g. Notre Dame).
- **Elevated Viaducts for Transport Tubes** — multi-level structures carrying pneumatic/electric tubes.
- **Pneumatic Transport Tubes** — underground compressed-air tubes for rapid transit of people and goods.
- **Rotating Houses** — buildings with rotating sections to change view or orientation.
- **Submarine Cities/Habitats** — cities built beneath the sea (e.g. *Central-Tubo*).
- **Transatlantic Aerial Travel** — long-distance ocean-crossing airships.
- **Vertical City Architecture** — buildings entered from the top, reflecting aerial-transport-first urban design.

## Communications & Media

- **Automated Public Announcement Systems** (*fono-annunziatore*) — a device that vocally announces visitors at entrances.
- **Concentrated Literature/Education** — condensing classics into short mnemonic verses (*quartine mnemotecniche*) for fast learning.
- **Telefonoscopo** — combined audio+video remote communication, essentially a video call.
- **Electronic Call Bells/Alarm Systems** — home systems summoning specific services (maid, fire brigade) via electric bells.
- **Facsimile/Remote Transmission of Reports** — newspapers sending theatre reviews etc. to subscribers by telephone.
- **Phonograph-Based News/Education Delivery** — phonographs used to deliver art commentary, news, or lessons.
- **Personalized Telephonic News/Wake Calls** — homes woken by telephone for breaking news.
- **Public Telephone Booths** — street kiosks (*colonnetta telefonica*) requiring a subscription key.
- **Simultaneous Multi-Language Theatre** — the same play performed in three languages at once for a mixed audience.

## Warfare & Security

- **Aerial Police/Gendarmerie** (*gendarmeria atmosferica*) — law enforcement patrolling city airspace.
- **Electric Anti-Burglar Traps** — security systems that shock intruders (used in the Ponto bank).
- **Explosive Devices in Urban Warfare** — dynamite bombs and electric torpedoes used in political assassination/unrest.
- **Personal Parachute Belts** — safety gear for aerial-vehicle passengers, deployed by pressing a button.
- **Revolvers and Advanced Firearms** — repeating rifles and "electric rifles" as standard personal/military arms.
- **Submarine Warfare/Transport** — submersible vessels for travel and implied military use.
- **Warfare with Aerial Vehicles** — aerial combat and pursuit as part of military operations.

## Society & Culture

- **Abolition of Capital Punishment** — replaced early in the century by rehabilitation/confinement.
- **Abolition of Traditional Prisons** — replaced by "colonization" and *villeggiatura* (a genteel form of exile/retreat used as punishment).
- **Simplified Marriage Dissolution** — marriages annulled via a legal "case of nullity" rather than divorce proceedings.
- **Aerial Tourism** — recreational sightseeing flights and airship resorts.
- **Automated Domestic Service** — mechanical devices announcing guests, replacing household staff.
- **Centralized Food Delivery** — meals piped to homes via tube from a subscription catering company.
- **Cosmopolitanism/Cultural Fusion** — easy travel and communication blending European nations into one culture.
- **Department Stores with Integrated Services** — huge retail complexes combining shopping, food, and lodging.
- **Electrically Powered Homes** — homes wired for lighting, communication, and automated services.
- **Female Emancipation and Political Participation** — women winning full voting/office rights.
- **Women in Law and Finance** — female lawyers and bankers as an unremarkable norm.
- **Gender-Neutral/Masculinized Naming** — girls given traditionally male-coded names (Nicola, Massimiliana...) reflecting emancipation.
- **Globalized Cuisine** — restaurants serving fully international menus as standard.
- **"Linguaggio Insalata"** — an invented pan-European pidgin ("salad language") blending French, Italian, English, and German, predicted to eventually replace national languages.
- **Marriage Agencies** — formal institutions with catalogs of eligible partners, replacing courtship (*Agenzia universale*).
- **Photographic Art Reproduction** — painting largely replaced by fast photographic reproduction of images.
- **Cyclical/Scheduled Revolutions** — the French government run on a constitution requiring a "regular revolution" every ten years (*vacanze decennali*) as a designed safety valve, laws only valid for 3 months at a time.
- **Historical Revisionism as Entertainment** — a "new school of history" claiming Napoleon and Louis XIV never existed, framed as tabloid-style pseudo-scholarship.

---
*Not included: dozens of near-duplicate restatements of the same ideas across the book's many scenes (the telefonoscopo alone recurs ~20+ times), and one model run that degenerated into repeating "'Insalata' [Theme]" variants — filtered out manually after the raw extraction (see conversation for the debugging story: greedy decoding + no frequency_penalty caused two separate runaway-repetition failures, fixed in `rag/gemini_llm.py`/`rag/summarizer.py`).*
