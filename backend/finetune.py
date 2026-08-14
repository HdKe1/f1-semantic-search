"""
Fine-tune the sentence-transformers model on F1 driver data.

Run from the backend/ directory:
    python finetune.py

Takes ~2-3 minutes on CPU. Produces ./f1_finetuned_model/ which the app
will auto-detect on next startup.

Uses raw PyTorch training to avoid datasets/dill compatibility issues
with Python 3.14.
"""

import json
import math
import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss

# Load drivers
with open("drivers_v2.json") as f:
    drivers = json.load(f)

id_to_vibe_text = {d["id"]: d["vibe_embedding_text"] for d in drivers}

# Start from the base model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Build training pairs
anchors = []
positives = []

# --- Source 1: auto-generated pairs from curated fields ---
for d in drivers:
    positive_text = d["vibe_embedding_text"]
    curated_terms = (
        d.get("memes", [])
        + d.get("aliases", [])
        + d.get("nicknames", [])
        + d.get("search_keywords", [])
    )
    for term in curated_terms:
        anchors.append(term)
        positives.append(positive_text)

# --- Source 2: hand-written paraphrase pairs ---
paraphrase_training_pairs = [
    ("emotionless finnish racer who avoids the media", "kimi_raikkonen"),
    ("driver who barely talks to journalists", "kimi_raikkonen"),
    ("aggressive defensive spanish veteran", "fernando_alonso"),
    ("driver who parties by drinking from his own shoe", "daniel_ricciardo"),
    ("hyper competitive dutch racer known for outbursts", "max_verstappen"),
    ("perfectionist calm mercedes legend", "lewis_hamilton"),
    ("honest self critical monaco based driver", "charles_leclerc"),
    ("calm consistent spanish strategist who loves spicy food", "carlos_sainz"),
    ("funny streamer mclaren driver who plays video games", "lando_norris"),
    ("unbothered rookie compared to an emotionless legend", "oscar_piastri"),
    ("methodical british qualifier who prepares detailed arguments", "george_russell"),
    ("humble german driver who protects bees", "sebastian_vettel"),
    ("patient reliable mexican racer great at saving tyres", "sergio_perez"),
    ("blunt finnish driver with a mullet who loves coffee", "valtteri_bottas"),
    ("driver who unexpectedly quit right after winning his title", "nico_rosberg"),
    ("relaxed champion who mastered wet weather driving", "jenson_button"),
    ("relentless dominant german legend with a strong work ethic", "michael_schumacher"),
    ("intense brazilian legend famous in the rain", "ayrton_senna"),
    ("calculated strategic french champion nicknamed the professor", "alain_prost"),
    ("quiet fast respectful finnish champion", "mika_hakkinen"),
    ("unlucky honest german driver who never got a podium", "nico_hulkenberg"),
    ("emotional resilient french driver who bounced back", "pierre_gasly"),
    ("loyal brazilian driver who always finished second", "rubens_barrichello"),
    ("son of a racing legend who became a british champion himself", "damon_hill"),
    ("outspoken canadian son of a famous racer", "jacques_villeneuve"),
    ("fearless mustached british champion", "nigel_mansell"),
    # Championship-focused pairs
    ("world champion multiple titles dominant", "michael_schumacher"),
    ("seven time world champion greatest of all time", "lewis_hamilton"),
    ("seven time world champion", "lewis_hamilton"),
    ("7 time world champion", "lewis_hamilton"),
    ("7 world championships", "lewis_hamilton"),
    ("most world championships", "michael_schumacher"),
    ("triple world champion youngest champion", "max_verstappen"),
    ("three time world champion", "max_verstappen"),
    ("3 time world champion", "max_verstappen"),
    ("3 world championships red bull", "max_verstappen"),
    ("four time world champion red bull", "sebastian_vettel"),
    ("four time world champion", "sebastian_vettel"),
    ("4 time world champion", "sebastian_vettel"),
    ("4 world championships", "sebastian_vettel"),
    ("4-time world champion", "sebastian_vettel"),
    ("two time world champion spanish fighter", "fernando_alonso"),
    ("two time world champion", "fernando_alonso"),
    ("2 time world champion", "fernando_alonso"),
    ("world champion ice cold finnish", "kimi_raikkonen"),
    ("one time world champion finland", "kimi_raikkonen"),
    ("2007 world champion", "kimi_raikkonen"),
    ("world champion retired immediately after winning", "nico_rosberg"),
    ("2016 world champion", "nico_rosberg"),
    ("world champion smooth british driver", "jenson_button"),
    ("2009 world champion brawn", "jenson_button"),
    ("world champion brazilian spiritual legend", "ayrton_senna"),
    ("three time world champion brazilian", "ayrton_senna"),
    ("world champion french strategic professor", "alain_prost"),
    ("four time world champion french", "alain_prost"),
    ("world champion flying finn mclaren", "mika_hakkinen"),
    ("two time world champion finnish", "mika_hakkinen"),
    ("world champion fearless british lion mustache", "nigel_mansell"),
    ("1992 world champion", "nigel_mansell"),
    ("world champion canadian outspoken son", "jacques_villeneuve"),
    ("1997 world champion", "jacques_villeneuve"),
    ("world champion son of graham hill british", "damon_hill"),
    ("1996 world champion", "damon_hill"),
    ("world champion south african ferrari last", "jody_scheckter"),
    ("1979 world champion", "jody_scheckter"),
    ("world champion australian tough hard racer", "alan_jones"),
    ("1980 world champion", "alan_jones"),
    ("world champion american born ferrari", "phil_hill"),
    ("1961 world champion", "phil_hill"),
    ("world champion built his own car australian", "jack_brabham"),
    ("three time world champion constructor", "jack_brabham"),
    ("world champion scottish natural talent lotus", "jim_clark"),
    ("two time world champion scottish", "jim_clark"),
    ("world champion triple crown monaco", "graham_hill"),
    ("two time world champion triple crown", "graham_hill"),
    ("world champion motorcycle and car", "john_surtees"),
    ("1964 world champion", "john_surtees"),
    ("world champion posthumous austrian died", "jochen_rindt"),
    ("1970 world champion", "jochen_rindt"),
    ("world champion american racing legend versatile", "mario_andretti"),
    ("1978 world champion", "mario_andretti"),
    # Generic "world champion" should surface top champions
    ("world champion", "michael_schumacher"),
    ("world champion", "lewis_hamilton"),
    ("world champion", "max_verstappen"),
    ("world champion", "sebastian_vettel"),
    ("world champion", "fernando_alonso"),
    ("f1 champion", "lewis_hamilton"),
    ("f1 champion", "michael_schumacher"),
    ("f1 champion", "max_verstappen"),
    ("formula 1 world champion", "lewis_hamilton"),
    ("formula 1 world champion", "michael_schumacher"),
    ("who won the championship", "max_verstappen"),
    ("championship winner", "lewis_hamilton"),
    ("championship winner", "michael_schumacher"),
    # Non-champions explicitly (negative signal via different wording)
    ("never won a championship close but not enough", "heinz_harald_frentzen"),
    ("never champion despite speed", "nico_hulkenberg"),
    ("no championship but fast", "brendon_hartley"),
    # Team associations
    ("red bull driver winning everything", "max_verstappen"),
    ("ferrari driver emotional Italian races", "charles_leclerc"),
    ("mclaren young british talent streamer", "lando_norris"),
    ("mercedes dominant era silver arrows", "lewis_hamilton"),
    ("aston martin veteran never gives up", "fernando_alonso"),
    # Era / rivalry pairs
    ("senna prost rivalry", "ayrton_senna"),
    ("hamilton rosberg rivalry teammates", "nico_rosberg"),
    ("schumacher era ferrari dominance", "michael_schumacher"),
    ("vettel red bull dominance finger celebration", "sebastian_vettel"),
    # Personality / off-track
    ("fashion icon social media activism champion", "lewis_hamilton"),
    ("piano playing emotional ferrari hopeful", "charles_leclerc"),
    ("twitch streamer gamer young british driver", "lando_norris"),
    ("environmental activist bees post-racing career", "sebastian_vettel"),
    ("naked cycling mullet finnish humor", "valtteri_bottas"),
    ("cat lover painting gentle soul", "alex_albon"),
    # Iconic moments
    ("brazil 2008 last corner championship decider", "timo_glock"),
    ("bahrain fireball survival miracle halo", "romain_grosjean"),
    ("monza shock win underdog alphatauri", "pierre_gasly"),
    ("spain 2012 shock williams win fire", "pastor_maldonado"),
    ("abu dhabi 2010 title decider blocker", "vitaly_petrov"),
    # Driving style
    ("late braking divebomb overtaker", "daniel_ricciardo"),
    ("smooth consistent tyre saver", "sergio_perez"),
    ("aggressive first lap chaos torpedo", "daniil_kvyat"),
    ("defensive driving master impossible to pass", "fernando_alonso"),
    ("qualifying specialist race pace slower", "jarno_trulli"),
    ("wet weather rain specialist", "ayrton_senna"),
    # Negative / meme reputation
    ("pay driver crashing spinning worst driver", "nikita_mazepin"),
    ("crash prone first lap incident specialist", "romain_grosjean"),
    ("always crashing chaotic unpredictable", "pastor_maldonado"),
    # Commentary / media
    ("tv pundit commentator former driver british", "paul_di_resta"),
    ("sky sports analysis indian former driver", "karun_chandhok"),
    # Comeback / resilience
    ("rally crash arm injury comeback inspiration", "robert_kubica"),
    ("paralympics handbike legs amputated racing hero", "alessandro_zanardi"),
    ("demoted from red bull then won a race", "pierre_gasly"),
    ("dropped by team came back stronger", "alex_albon"),
    # --- Team associations (longest/most iconic first, repeated for weight) ---
    # Ferrari
    ("ferrari", "michael_schumacher"),
    ("ferrari", "michael_schumacher"),
    ("ferrari driver", "michael_schumacher"),
    ("ferrari legend", "michael_schumacher"),
    ("ferrari", "rubens_barrichello"),
    ("ferrari", "kimi_raikkonen"),
    ("ferrari", "sebastian_vettel"),
    ("ferrari", "charles_leclerc"),
    ("ferrari", "jean_alesi"),
    ("ferrari", "michele_alboreto"),
    ("ferrari", "jody_scheckter"),
    ("ferrari", "luca_badoer"),
    ("ferrari team", "michael_schumacher"),
    ("scuderia ferrari", "michael_schumacher"),
    ("who drove for ferrari", "michael_schumacher"),
    ("who drove for ferrari the longest", "michael_schumacher"),
    # McLaren
    ("mclaren", "lewis_hamilton"),
    ("mclaren", "lewis_hamilton"),
    ("mclaren driver", "lewis_hamilton"),
    ("mclaren", "ayrton_senna"),
    ("mclaren", "mika_hakkinen"),
    ("mclaren", "alain_prost"),
    ("mclaren", "jenson_button"),
    ("mclaren", "lando_norris"),
    ("mclaren", "oscar_piastri"),
    ("mclaren", "daniel_ricciardo"),
    ("mclaren", "heikki_kovalainen"),
    ("mclaren", "pedro_de_la_rosa"),
    ("mclaren", "stoffel_vandoorne"),
    ("who drove for mclaren", "lewis_hamilton"),
    ("mclaren legend", "ayrton_senna"),
    # Red Bull
    ("red bull", "sebastian_vettel"),
    ("red bull", "sebastian_vettel"),
    ("red bull driver", "sebastian_vettel"),
    ("red bull", "max_verstappen"),
    ("red bull", "max_verstappen"),
    ("red bull", "daniel_ricciardo"),
    ("red bull", "mark_webber"),
    ("red bull", "sergio_perez"),
    ("red bull", "daniil_kvyat"),
    ("red bull", "alex_albon"),
    ("red bull", "pierre_gasly"),
    ("red bull racing", "max_verstappen"),
    ("who drove for red bull", "sebastian_vettel"),
    ("red bull legend", "sebastian_vettel"),
    # Mercedes
    ("mercedes", "lewis_hamilton"),
    ("mercedes", "lewis_hamilton"),
    ("mercedes driver", "lewis_hamilton"),
    ("mercedes", "nico_rosberg"),
    ("mercedes", "valtteri_bottas"),
    ("mercedes", "george_russell"),
    ("mercedes", "michael_schumacher"),
    ("who drove for mercedes", "lewis_hamilton"),
    ("mercedes legend", "lewis_hamilton"),
    ("silver arrows", "lewis_hamilton"),
    # Williams
    ("williams", "nigel_mansell"),
    ("williams", "nigel_mansell"),
    ("williams driver", "nigel_mansell"),
    ("williams", "alain_prost"),
    ("williams", "damon_hill"),
    ("williams", "jacques_villeneuve"),
    ("williams", "ricardo_patrese"),
    ("williams", "alex_albon"),
    ("williams", "lance_stroll"),
    ("williams", "george_russell"),
    ("williams", "logan_sargeant"),
    ("who drove for williams", "nigel_mansell"),
    ("williams legend", "nigel_mansell"),
    # Renault / Alpine
    ("renault", "fernando_alonso"),
    ("renault", "fernando_alonso"),
    ("renault driver", "fernando_alonso"),
    ("alpine", "fernando_alonso"),
    ("alpine", "esteban_ocon"),
    ("alpine", "pierre_gasly"),
    ("alpine", "oscar_piastri"),
    ("who drove for renault", "fernando_alonso"),
    ("renault legend", "fernando_alonso"),
    # Lotus (classic)
    ("lotus", "jim_clark"),
    ("lotus", "jim_clark"),
    ("lotus driver", "jim_clark"),
    ("lotus", "graham_hill"),
    ("lotus", "ayrton_senna"),
    ("lotus", "jochen_rindt"),
    ("who drove for lotus", "jim_clark"),
    # Aston Martin / Force India
    ("aston martin", "fernando_alonso"),
    ("aston martin", "fernando_alonso"),
    ("aston martin driver", "fernando_alonso"),
    ("aston martin", "lance_stroll"),
    ("aston martin", "sebastian_vettel"),
    ("force india", "sergio_perez"),
    ("force india", "adrian_sutil"),
    ("force india", "paul_di_resta"),
    ("force india", "esteban_ocon"),
    ("force india", "nico_hulkenberg"),
    # Haas
    ("haas", "kevin_magnussen"),
    ("haas", "kevin_magnussen"),
    ("haas driver", "kevin_magnussen"),
    ("haas", "romain_grosjean"),
    ("haas", "mick_schumacher"),
    ("haas", "nikita_mazepin"),
    ("who drove for haas", "kevin_magnussen"),
    # Toro Rosso / AlphaTauri / Racing Bulls
    ("toro rosso", "sebastian_vettel"),
    ("toro rosso", "daniil_kvyat"),
    ("toro rosso", "pierre_gasly"),
    ("alphatauri", "pierre_gasly"),
    ("alphatauri", "yuki_tsunoda"),
    ("racing bulls", "yuki_tsunoda"),
    ("racing bulls", "daniel_ricciardo"),
    # Sauber / Alfa Romeo
    ("sauber", "kimi_raikkonen"),
    ("sauber", "felipe_nasr"),
    ("sauber", "marcus_ericsson"),
    ("alfa romeo", "kimi_raikkonen"),
    ("alfa romeo", "valtteri_bottas"),
    ("alfa romeo", "antonio_giovinazzi"),
    ("alfa romeo", "zhou_guanyu"),
    # Benetton
    ("benetton", "michael_schumacher"),
    ("benetton", "michael_schumacher"),
    ("benetton driver", "michael_schumacher"),
    ("benetton", "fernando_alonso"),
    ("benetton", "gerhard_berger"),
    ("benetton", "jos_verstappen"),
    # Jordan
    ("jordan", "heinz_harald_frentzen"),
    ("jordan", "damon_hill"),
    ("jordan", "giancarlo_fisichella"),
    # Brawn GP
    ("brawn gp", "jenson_button"),
    ("brawn gp", "rubens_barrichello"),
    ("brawn", "jenson_button"),
]

for query_text, driver_id in paraphrase_training_pairs:
    if driver_id in id_to_vibe_text:
        anchors.append(query_text)
        positives.append(id_to_vibe_text[driver_id])
    else:
        print(f"WARNING: driver_id '{driver_id}' not found, skipping")

print(f"Total training pairs: {len(anchors)}")


# --- Custom Dataset to bypass HuggingFace datasets ---
class PairDataset(Dataset):
    def __init__(self, anchors, positives):
        self.anchors = anchors
        self.positives = positives

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        return {"anchor": self.anchors[idx], "positive": self.positives[idx]}


def collate_fn(batch):
    return {
        "anchor": [item["anchor"] for item in batch],
        "positive": [item["positive"] for item in batch],
    }


# Training setup
EPOCHS = 8
BATCH_SIZE = 16
WARMUP_STEPS = 10
LR = 2e-5

train_dataset = PairDataset(anchors, positives)
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)

loss_fn = MultipleNegativesRankingLoss(model)

total_steps = len(train_dataloader) * EPOCHS
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# Linear warmup scheduler
def get_lr(step):
    if step < WARMUP_STEPS:
        return step / max(WARMUP_STEPS, 1)
    return max(0.0, 1.0 - (step - WARMUP_STEPS) / (total_steps - WARMUP_STEPS))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

# Training loop
print(f"Starting fine-tuning: {EPOCHS} epochs, {len(train_dataloader)} batches/epoch, {total_steps} total steps")
model.train()

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for step, batch in enumerate(train_dataloader):
        # Tokenize both columns
        anchor_features = model.tokenize(batch["anchor"])
        positive_features = model.tokenize(batch["positive"])

        # Move to device
        anchor_features = {k: v.to(model.device) for k, v in anchor_features.items()}
        positive_features = {k: v.to(model.device) for k, v in positive_features.items()}

        # Forward pass
        anchor_embeddings = model(anchor_features)["sentence_embedding"]
        positive_embeddings = model(positive_features)["sentence_embedding"]

        # Compute MNR loss (in-batch negatives)
        # Cosine similarity matrix
        anchor_norm = torch.nn.functional.normalize(anchor_embeddings, p=2, dim=1)
        positive_norm = torch.nn.functional.normalize(positive_embeddings, p=2, dim=1)
        scores = torch.mm(anchor_norm, positive_norm.t()) * 20.0  # scale=20 is default

        # Labels: diagonal (each anchor matches its own positive)
        labels = torch.arange(scores.shape[0], device=scores.device)
        loss = torch.nn.functional.cross_entropy(scores, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

        global_step = epoch * len(train_dataloader) + step + 1
        if global_step % 20 == 0:
            avg = epoch_loss / (step + 1)
            print(f"  Epoch {epoch+1}/{EPOCHS} | Step {step+1}/{len(train_dataloader)} | Loss: {avg:.4f}")

    avg_epoch_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS} complete — avg loss: {avg_epoch_loss:.4f}")

# Save
model.save("f1_finetuned_model")
print("\n✅ Saved fine-tuned model to ./f1_finetuned_model/")
print("Restart the app (python app.py) and it will auto-detect the model.")
