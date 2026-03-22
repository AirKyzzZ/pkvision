# How to Propose a New Trick

This guide is for anyone who wants to add a new parkour trick to the PkVision detection catalog. No coding experience is required. A maintainer will handle the technical implementation based on your submission.

---

## Overview

PkVision detects tricks by matching video frames against a catalog of trick definitions. Each trick is defined as a JSON configuration file that describes:

- The trick name (in English and optionally other languages)
- Its category (flip, vault, twist, combo, spin, precision)
- A difficulty rating (0 to 10)
- The phases of the trick (approach, takeoff, execution, landing)
- What joint angles and body positions define each phase

If a trick is not in the catalog, the system cannot detect it. Your proposal helps expand what PkVision can recognize.

---

## Step 1: Open a GitHub Issue

Go to the repository and open a new issue using the **"Propose New Trick"** template:

[Open a Trick Proposal Issue](../../issues/new?template=trick_submission.yml)

---

## Step 2: Fill Out the Form

The issue template asks for the following information:

### Trick Name (English) -- Required

The standard English name for the trick as it is commonly known in the parkour community.

Examples: "Lazy Vault", "Dash Vault", "Aerial", "B-Twist"

### Trick Name (French) -- Optional

The French name, if one exists. This supports the bilingual nature of the project.

Examples: "Saut de Paresseux", "Saut Dash", "Aerien"

### Category -- Required

Choose the category that best describes the trick:

| Category | Description | Examples |
|----------|-------------|----------|
| Flip | Rotations around a horizontal axis | Front flip, back flip, webster |
| Vault | Movements over obstacles using hands | Kong vault, lazy vault, dash vault |
| Twist | Rotations around a vertical axis | B-twist, 360, corkscrew |
| Combo | Combinations of multiple movements | Double kong, front-full |
| Spin | Rotational movements without full inversion | Tornado kick, 540 |
| Precision | Jumps to specific landing points | Precision jump, standing precision |

### Difficulty Estimate (1-10) -- Required

Rate the difficulty based on the skill required to perform the trick cleanly:

| Range | Level | Examples |
|-------|-------|----------|
| 1-2 | Beginner | Basic vaults, safety rolls |
| 3-4 | Intermediate | Single flips, standard vaults |
| 5-6 | Advanced | Webster, gainer, 360 flip |
| 7-8 | Expert | Double rotations, complex combos |
| 9-10 | Elite | Triple cork, multi-axis rotations |

This is an estimate. The maintainer team will calibrate the final value based on FIG guidelines and community consensus.

### Description of the Movement -- Required

Describe how the trick is performed. Focus on:

- **Starting position** -- How does the athlete begin? Running, standing, from an obstacle?
- **Takeoff** -- How do they leave the ground? Two feet, one foot, hands?
- **Execution** -- What happens in the air or during the main movement? Rotations, tucks, twists?
- **Landing** -- How do they finish? Two feet, one foot, rolling?
- **Key body positions** -- What joints are bent or extended at each stage?

The more detail you provide about body mechanics, the easier it is to translate your description into detection rules.

Example:

> "The athlete runs toward an obstacle. They dive forward, placing both hands on the obstacle with arms fully extended. As the hands make contact, they tuck their legs through the gap between their arms, flexing at the hips. The legs pass through and extend forward. The athlete pushes off with their hands and lands on both feet on the other side."

### Reference Video Link -- Optional

A link to a video showing the trick being performed. This is extremely helpful for the maintainer who will define the detection rules. YouTube, Instagram, or any publicly accessible video link works.

### Tags -- Optional

Comma-separated tags that describe the trick. These help with searching and filtering in the catalog.

Examples: "acrobatic, aerial, rotation, advanced", "vault, obstacle, one-hand", "twist, spinning, standing"

---

## Step 3: What Happens Next

After you submit the issue:

1. **Review** -- A maintainer reviews your proposal for completeness and accuracy.
2. **Discussion** -- If anything is unclear, the maintainer may ask questions in the issue thread. Community members may also weigh in on difficulty ratings or naming.
3. **Implementation** -- The maintainer creates a JSON configuration file for the trick based on your description and any reference video.
4. **Testing** -- The new trick definition is tested against available clips to verify detection accuracy.
5. **Merge** -- The trick is added to the catalog in both English and French (if a French name was provided).
6. **Notification** -- The issue is closed with a reference to the commit that added the trick.

Typical turnaround time depends on maintainer availability and the complexity of the trick. Simple tricks with clear descriptions and reference videos are fastest to implement.

---

## Tips for a Good Submission

- **Be specific about body positions.** "Arms extended" is more useful than "arms up."
- **Mention joint angles if you can.** "Knees bent at about 90 degrees during tuck" directly translates to detection rules.
- **Include a reference video.** A 5-second clip is worth more than a paragraph of description.
- **Note variations.** If the trick has common variations (e.g., tucked vs. layout), mention them so the maintainer can account for the range of valid forms.
- **Check the existing catalog first.** The current tricks are listed in `data/tricks/catalog/en/`. If the trick already exists, consider submitting a clip to improve its detection instead.

---

## Current Catalog

As of the latest release, the following tricks are in the catalog:

| Trick | Category | Difficulty |
|-------|----------|------------|
| Kong Vault | Vault | 2.0 |
| Double Kong | Vault | 7.0 |
| Front Flip | Flip | 3.5 |
| Back Flip | Flip | 3.5 |
| Side Flip | Flip | 4.0 |
| Webster | Flip | 5.5 |
| Gainer | Flip | 5.5 |
| Double Front Flip | Flip | 7.5 |
| 360 Flip | Twist | 6.0 |
| Triple Cork | Twist | 9.5 |

If the trick you want to propose is not on this list, we want to hear about it.
