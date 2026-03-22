# Clip Filming Guidelines

This guide is for athletes, coaches, and anyone filming parkour clips intended for PkVision training data or analysis. Following these guidelines will produce clips that the AI can process accurately.

---

## Quick Checklist

- [ ] Resolution: 720p or higher
- [ ] Frame rate: 30fps or higher
- [ ] Camera angle: side or diagonal
- [ ] Framing: full body visible at all times
- [ ] Buffer: 1-2 seconds before and after the trick
- [ ] Clothing: contrasting colors against background
- [ ] Camera: stable (tripod preferred)
- [ ] Content: one trick per clip
- [ ] Lighting: well-lit environment

---

## Resolution and Frame Rate

**Minimum resolution: 720p (1280x720)**

Higher resolution gives the pose detector more detail to work with. 1080p is ideal. 4K works but provides diminishing returns for pose estimation and increases processing time.

**Minimum frame rate: 30fps**

Parkour tricks happen fast. At 30fps, a backflip lasting 0.5 seconds is captured in approximately 15 frames. 60fps is better for very fast tricks (twists, corks) where subtle body positions change between frames. Avoid slow-motion recordings unless you also provide a normal-speed version.

---

## Camera Angle

**Preferred: side view or diagonal (45 degrees)**

The pose detector tracks 17 body keypoints. Side and diagonal angles reveal the most joint information:

- **Side view** -- Best for flips, vaults, and forward/backward rotations. All major joints (knee, hip, shoulder, elbow) are clearly visible.
- **Diagonal view (45 degrees)** -- Good general-purpose angle. Captures both lateral and forward movement.
- **Front view** -- Acceptable for some tricks but obscures depth and makes it harder to distinguish joint angles.
- **Overhead/top-down** -- Avoid. The pose detector performs poorly when the body is foreshortened.

If filming for training data, recording the same trick from multiple angles is valuable.

---

## Framing

**The athlete's full body must be visible throughout the entire trick.**

This means:

- Head to feet in frame from start to finish
- No cropping of limbs at the edge of the frame
- Allow extra space around the athlete for movement (especially height for flips)
- If the athlete moves horizontally (vaults, running tricks), pan the camera to follow or frame wide enough to capture the full movement

The pose detector cannot estimate keypoints for body parts that are out of frame.

---

## Timing and Buffer

**Include 1-2 seconds of footage before and after the trick.**

The detection system analyzes the approach phase (before the trick) and the landing/recovery phase (after). Clips that start mid-trick or cut off immediately after landing will lose context needed for accurate phase detection.

Good structure for a clip:
1. 1-2 seconds: athlete in starting position or approach
2. Trick execution
3. 1-2 seconds: landing and recovery

---

## Clothing

**Wear clothing that contrasts with the background.**

- Dark clothing on light backgrounds (or vice versa)
- Avoid patterns that match the environment (e.g., grey clothing against concrete)
- Avoid very loose or baggy clothing that obscures joint positions
- Tight-fitting or athletic clothing works best for keypoint detection
- Avoid white-on-white (white clothing against white walls or sky)

---

## Camera Stability

**Use a tripod or stable surface whenever possible.**

- A stable camera makes it easier for the system to isolate the athlete's movement from camera shake.
- If filming handheld, brace against a wall or use both hands.
- Avoid zooming in and out during the trick.
- A fixed, wide-angle shot is better than a shaky close-up.

---

## Content

**One trick per clip is ideal.**

- If a clip contains multiple tricks in sequence, note the timestamp of each trick when submitting.
- The system can analyze multi-trick videos, but individual clips are easier to label and produce cleaner training data.
- If filming a run (multiple tricks in sequence), that is also valuable for testing the full pipeline.

---

## Lighting

**Film in well-lit environments.**

- Natural daylight or bright indoor lighting works best.
- Avoid strong backlighting (e.g., athlete silhouetted against the sky).
- Avoid deep shadows that obscure parts of the body.
- Indoor gyms with even lighting produce the most consistent results.
- If filming outdoors, overcast days provide even lighting without harsh shadows.

---

## Environment

- Clear, uncluttered backgrounds improve detection accuracy.
- If other people are in frame, the system selects the person with the highest keypoint confidence (usually the most visible/centered person).
- Filming in a gym, park, or open space with minimal obstructions is ideal.

---

## File Format

- Any format supported by OpenCV: MP4, AVI, MOV, MKV, WEBM
- MP4 with H.264 encoding is recommended for compatibility and file size
- Avoid heavily compressed or low-bitrate videos

---

## Submitting Your Clips

Once you have a clip that meets these guidelines:

1. **GitHub Issue** -- Open a [Clip Submission issue](../../issues/new?template=clip_submission.yml) with a link to the video (YouTube, Google Drive, Dropbox, or any accessible URL).
2. **API** -- If you have API access, submit via `POST /api/v1/submissions`.

Include:
- The trick name
- The timestamp where the trick occurs (if the clip is longer than the trick itself)
- The camera angle used
- Any additional notes (lighting conditions, skill level, etc.)

---

## Summary of Requirements

| Parameter | Minimum | Recommended |
|-----------|---------|-------------|
| Resolution | 720p | 1080p |
| Frame rate | 30fps | 60fps |
| Camera angle | Side | Side or diagonal |
| Body visibility | Full body | Full body with margin |
| Buffer | 1 second | 2 seconds |
| Stability | Handheld (steady) | Tripod |
| Lighting | Adequate | Bright, even |
| Tricks per clip | Any | One |
