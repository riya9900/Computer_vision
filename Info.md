# OrbitWatch: What We Are Building

## In one line
A website that watches the Earth from space and tells you when land is being changed, such as a new mine, a shrinking lake, or a building where there was none.

## The problem
Land is being dug up, built on and cleared every day, often without anyone noticing for months. Checking by hand means sending people to the spot, which is slow and costly. Free satellite photos of every place on Earth are taken every few days, but nobody has time to look through them.

## What our system does
1. **You point at one place.** Draw a shape around a spot you already know about, such as a quarry.
2. **It finds look-alikes.** An AI looks at satellite photos of the whole district and shows every other place that looks similar. Find one, find them all.
3. **It watches them every week.** Behind the scenes, background workers fetch new satellite photos and compare them with the old ones.
4. **It tells you what changed and when.** You get a before/after photo, the area that changed, and roughly when it started.
5. **You can just ask.** A built-in AI assistant can look into a site and write a short report for you.

## What makes it special
- **Free satellite images.** We use public photos from the European Sentinel satellites.
- **An AI that has seen millions of satellite photos.** It is called *Clay* and is open source. It gives every patch of land a kind of fingerprint, so look-alike places can be found without us teaching it each type.
- **Works through clouds.** When clouds block the view, we switch to radar satellite data, which sees through them.
- **Runs by itself.** The weekly checks happen automatically, even if a server restarts.

## What you will see on screen
- A 3D globe that zooms into the chosen area
- A before/after slider to compare two dates
- A timeline showing when the changes happened
- A map of look-alike places, with alerts when something new appears

## Honest limits
- Satellite photos are not sharp. We can spot large changes, not small ones like a single house.
- We report that **land changed**. We do **not** decide whether it is illegal. A person must check that.

## Project details
Course: CSE471, Lovely Professional University.
Built in 8 hours using only free tools and datasets.

==========================================================================
Datasets

OSCD for change detection. It has 24 Sentinel-2 image pairs from 2015–2018, 13 bands each. There are 14 training and 10 test pairs, and the labelled changes are urban ones such as new buildings and roads. The change labels are CC BY-NC-SA, which is fine for a course project. 
ieee-dataport
github
EuroSAT for land-cover classification. It has 27,000 labelled, geo-referenced Sentinel-2 images in 10 classes across 13 bands. It’s MIT licensed. 
alphaxiv
github

Training

Clay stays frozen. The standard recipe is a frozen Clay encoder with a small trainable head on top, so only the head learns. 
developmentseed
Two heads:
Head	Input	Output	Trained on
Change head	Clay features of the before and after images, plus their difference	Where it changed, as a pixel mask	OSCD train (14 pairs)
Land-cover head	Clay features of one patch	What it is: forest, industrial, water and so on	EuroSAT
Together they give “where it changed” and “from what to what”, for example forest → industrial. That is the dossier line.
Clay’s own repo already has this pattern. It ships a segmentation head and a EuroSAT classifier, which is a 2-layer MLP on the class token. You’re adapting a documented recipe, not inventing one. 
deepwiki

Evaluation

Change head: F1, IoU, precision and recall on OSCD test (10 pairs).
Land-cover head: accuracy and a confusion matrix on a held-out EuroSAT split.
Baselines: spectral-index differencing, TinyCD, and a linear probe on Clay.
Three honest gaps to settle before the viva
OSCD only labels urban change. The heads won’t be trained on mines or lakes, and no free dataset covers those. For them we use the land-cover head, spectral indices and lookalike search, and show them as qualitative results with a clear statement of that limit.
Clay is heavy. The encoder is about 311M parameters, and the official weights are about 1.25 GB. That doesn’t fit the free-CPU, small-instance plan I gave earlier. The fix: 
deepwiki
huggingface
Run Clay on a Kaggle GPU once and cache its features for OSCD, EuroSAT and your demo areas.
Train the heads on the cached features, which takes minutes.
The live app then only runs the small heads and the vector search.
Fresh scans of new areas go to a Temporal worker on a bigger EC2 instance. Set a budget alarm.
14 training pairs is small. Use flips, rotations and before/after swaps, and report results across a few random seeds. This is also why a frozen foundation model is the right choice here.