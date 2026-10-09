# Hand-comparison videos

Three paired hand animations visualize saved RecurFlex predictions and recorded glove labels for the BCI Competition IV Dataset 4 test recordings. Each video plays the **fixed 60–80 s interval at real time**, at 25 fps and 960 × 600 pixels. The same interval is used for all subjects; it was not selected by optimizing prediction agreement. The complete 200-second trajectory plots remain in the main README.

Left: recorded label. Right: RecurFlex prediction. The five inputs are ordered thumb, index, middle, ring, little. Both hands use the same per-finger mapping from the original training label ranges:

```text
visual_flexion = clip((glove_value - training_target_min) / training_target_range, 0, 1)
```

The rendered hands are original procedural 3D **schematic** geometry. Glove units determine a visual joint-angle control; they are not calibrated anatomical joint angles. Bounding the visual controls does not change stored predictions or scores. No smoothing, time shift, separate prediction rescaling, or image-generated motion is used. Correlations in the video title are the full-test four-finger scores.

`video_manifest.json` records input/output hashes and the mapping. The renderer independently recomputes all full-test finger correlations before creating any video.

## Recreate

Install the visualization dependencies in your Python environment:

```bash
python -m pip install numpy scipy Pillow imageio-ffmpeg
python tools/render_hand_videos.py --paper-dir paper --output-dir assets/videos
```

The original `paper/data/BCICIV_4_mat/subN_testlabels.mat` label files must be obtained separately. Saved predictions and checkpoint scaling statistics are included in this repository.

## GitHub display

The main README embeds animated GIF previews and links each preview to its MP4. These work with ordinary repository files after a commit and push.

For a native inline player like FingerFlex's, GitHub needs a video attachment URL. Open the GitHub README editor, drag each `sN_hand_comparison.mp4` into the corresponding subject section, wait for the generated `https://github.com/user-attachments/assets/...` URL, and use that URL on a separate line. Preview before saving. The local MP4s are H.264, silent, and below 1 MB each.

The paired layout was inspired by [FingerFlex's public demo](https://github.com/Irautak/FingerFlex). The hand geometry and renderer were created independently; no model assets or video content from that project are redistributed.
