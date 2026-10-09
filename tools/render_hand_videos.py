"""Render paired schematic 3D hands from saved ECoG predictions and glove labels.

The same training-derived per-finger mapping is used for both hands. Only the
visual joint angles are bounded; original predictions and scoring stay intact.
No external hand model or FingerFlex asset is used.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.io import loadmat
import imageio_ffmpeg

WIDTH, HEIGHT = 960, 600
BACKGROUND = (246, 247, 249)
COLORS = [(53, 94, 132), (197, 104, 54)]
FINGER_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font(size, bold=False):
    candidates = [Path('C:/Windows/Fonts') / ('segoeuib.ttf' if bold else 'segoeui.ttf'),
                  Path('/usr/share/fonts/truetype/dejavu') / ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def sphere(center, radii, rings=10, sides=16):
    u = np.linspace(-np.pi / 2, np.pi / 2, rings + 1)
    v = np.linspace(0, 2 * np.pi, sides + 1)
    grid = np.stack(np.broadcast_arrays(np.cos(u[:, None]) * np.cos(v),
                     np.sin(u[:, None]) + np.zeros_like(v),
                     np.cos(u[:, None]) * np.sin(v)), -1)
    grid = grid * radii + center
    return np.stack([grid[:-1, :-1], grid[1:, :-1], grid[1:, 1:], grid[:-1, 1:]], 2).reshape(-1, 4, 3)


def finger(base, lengths, radius, flex, splay=0, thumb=False):
    # Continuous tube across three phalanges; rounded joints and a fingertip cap.
    points, angles = [], []
    position = np.asarray(base, float)
    angle = 0.
    joints = np.array([1.30, 1.45, .8]) * flex
    if thumb:
        joints = np.array([.65, .95, .45]) * flex
    for segment, length in enumerate(lengths):
        before = angle
        angle += joints[segment]
        # Small-angle interpolation rounds the bend instead of creating a hinge.
        for k in range(5):
            a = before + (angle - before) * min(1, (k + 1) / 2)
            direction = np.array([np.sin(splay) * np.cos(a), np.cos(splay) * np.cos(a), np.sin(a)])
            points.append(position.copy())
            angles.append(direction)
            position += direction * length / 5
    points.append(position.copy())
    angles.append(angles[-1])
    points, directions = np.asarray(points), np.asarray(angles)
    around = np.linspace(0, 2*np.pi, 25)
    side = np.tile([np.cos(splay), -np.sin(splay), 0.], (len(points), 1))
    normal = np.cross(side, directions)
    taper = np.linspace(1.08, .78, len(points)) * radius
    rings = points[:, None, :] + taper[:, None, None] * (
        np.cos(around)[None, :, None]*side[:, None, :] + np.sin(around)[None, :, None]*normal[:, None, :])
    quads = np.stack([rings[:-1, :-1], rings[1:, :-1], rings[1:, 1:], rings[:-1, 1:]], 2).reshape(-1, 4, 3)
    # A smooth spherical end avoids flat cut-off fingers.
    tip = sphere(position, [radius*.8]*3, rings=12, sides=24)
    return np.concatenate([quads, tip])


def palm_mesh():
    palm = sphere([0., .61, -.04], [1.03, 1.12, .31], rings=30, sides=48)
    # Blend the thumb pad into the palm instead of overlaying another sphere.
    palm[:,:,2] += .16*np.exp(-((palm[:,:,0]+.62)/.38)**2-((palm[:,:,1]-.45)/.65)**2)*np.clip(palm[:,:,2]/.2,0,1)
    v = np.linspace(0,2*np.pi,49)
    y = np.linspace(-1.12,-.28,9)
    widths = np.linspace(.48,.64,len(y))
    depths = np.linspace(.22,.27,len(y))
    cuff = np.stack(np.broadcast_arrays(widths[:,None]*np.cos(v),y[:,None]+np.zeros_like(v),depths[:,None]*np.sin(v)-.03),-1)
    cuff = np.stack([cuff[:-1,:-1],cuff[1:,:-1],cuff[1:,1:],cuff[:-1,1:]],2).reshape(-1,4,3)
    return np.concatenate([palm,cuff])


PALM = palm_mesh()


def hand_mesh(values):
    pieces = [PALM]
    # Data order: thumb, index, middle, ring, little.
    pieces.append(finger([-.83, .58, .06], [.48, .48, .32], .23, values[0], splay=-.95, thumb=True))
    for i, (x, y, length, radius, splay) in enumerate([
        (-.66, 1.43, 1.32, .205, -.085),
        (-.20, 1.60, 1.48, .21, -.015),
        (.27, 1.51, 1.34, .19, .055),
        (.70, 1.29, 1.08, .165, .12),
    ]):
        pieces.append(finger([x, y, -.02], np.array([.46, .32, .22])*length, radius, values[i+1], splay))
    return np.concatenate(pieces)


def project_hand(draw, flex, center_x, base_color):
    faces = hand_mesh(flex)
    # A modest fixed view tilt exposes finger bending and palm thickness.
    ax, ay = -.16, -.20
    rx = np.array([[1,0,0],[0,np.cos(ax),-np.sin(ax)],[0,np.sin(ax),np.cos(ax)]])
    ry = np.array([[np.cos(ay),0,np.sin(ay)],[0,1,0],[-np.sin(ay),0,np.cos(ay)]])
    faces = faces @ (ry @ rx).T
    normals = np.cross(faces[:, 1]-faces[:, 0], faces[:, 3]-faces[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    # Quads are consistently parameterized, but shading both sides also handles
    # the closed palm surface without exposing artificial seam directions.
    normals *= np.where(normals[:, 2:3] < 0, -1, 1)
    light = np.array([-.48, .60, .65]); light /= np.linalg.norm(light)
    diffuse = np.clip(normals @ light, 0, 1)
    specular = np.clip(normals @ np.array([-.25, .35, .903]), 0, 1)**22
    shade = .36 + .63*diffuse
    rgb = np.clip(shade[:,None]*np.array(base_color) + 42*specular[:,None], 0, 255).astype(np.uint8)
    perspective = 1.0 / (1 - faces[:,:,2]*.048)
    screen = np.stack([center_x + 82*faces[:,:,0]*perspective,
                       395 - 82*faces[:,:,1]*perspective], -1)
    for i in np.argsort(faces[:,:,2].mean(1)):
        draw.polygon([tuple(p) for p in screen[i]], fill=tuple(rgb[i]))


def text_center(draw, x, y, text, size, color, bold=False):
    draw.text((x, y), text, fill=color, font=font(size, bold), anchor='mt')


def render_frame(subject, time, end, truth, prediction, lo, span, r4, start):
    image = Image.new('RGB', (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    text_center(draw, WIDTH/2, 18, f'RecurFlex  |  Subject {subject}', 24, (35,43,55), True)
    text_center(draw, WIDTH/2, 54, f'Official test: {start:.0f}–{end:.0f} s   ·   t = {time:05.2f} s   ·   Full-test r4 = {r4:.4f}', 16, (96,106,119))
    draw.line((38,87,922,87),fill=(216,222,229),width=1)
    draw.line((480,111,480,520),fill=(224,228,233),width=1)
    text_center(draw, 244, 101, 'Recorded label', 23, COLORS[0], True)
    text_center(draw, 716, 101, 'RecurFlex prediction', 23, COLORS[1], True)
    normalised = [np.clip((truth-lo)/span, 0, 1), np.clip((prediction-lo)/span, 0, 1)]
    for x, value, color in zip([244,716], normalised, [(139,161,185),(207,173,144)]):
        project_hand(draw, value, x, color)
    # Matched per-finger meters make the scalar-to-pose correspondence explicit.
    for hand, x0 in enumerate([59,531]):
        for i, name in enumerate(FINGER_NAMES):
            x=x0+i*81
            draw.text((x+31,520),name,fill=(103,112,124),font=font(11),anchor='mt')
            draw.rounded_rectangle((x,540,x+63,546),radius=3,fill=(222,227,233))
            if normalised[hand][i]>.01:
                draw.rounded_rectangle((x,540,x+63*normalised[hand][i],546),radius=2,fill=COLORS[hand])
    draw.line((38,564,922,564),fill=(216,222,229),width=1)
    text_center(draw,480,572,'Real-time playback · Same training-derived pose mapping · Schematic hand animation',12,(103,112,124))
    return image


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--paper-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--subjects',nargs='+',type=int,default=[1,2,3])
    ap.add_argument('--start',type=float,default=60)
    ap.add_argument('--duration',type=float,default=20)
    ap.add_argument('--fps',type=int,default=25)
    ap.add_argument('--preview-only',action='store_true')
    args=ap.parse_args()
    if not 0<=args.start<args.start+args.duration<=200: ap.error('Interval must be inside the official 200-second test.')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    p=args.paper_dir
    avg=p/'result/main experiment average r'
    manifest=json.loads((avg/'prediction_manifest.json').read_text())
    summary=json.loads((avg/'summary.json').read_text())
    all_info=[]
    for subject in args.subjects:
        item=next(x for x in manifest['predictions'] if x['subject']==subject)
        row=next(x for x in summary['subjects'] if x['subject']==subject)
        pred_path=avg/item['prediction']; label_path=p/'data/BCICIV_4_mat'/f'sub{subject}_testlabels.mat'
        scaler_path=p/'code/Main Experiment/artifacts/pretrained'/f's{subject}'/'scalers.json'
        assert sha256(pred_path)==item['prediction_sha256']
        assert sha256(label_path)==row['label_sha256']
        with np.load(pred_path) as data: pred=data['prediction_1000hz']
        truth=loadmat(label_path,variable_names=['test_dg'])['test_dg']
        assert truth.shape==pred.shape==(200000,5)
        actual=np.array([np.corrcoef(truth[:,i],pred[:,i])[0,1] for i in range(5)])
        assert np.allclose(actual,row['r_fingers'],atol=1e-10,rtol=0)
        scales=json.loads(scaler_path.read_text())
        lo,span=np.array(scales['target_min']),np.array(scales['target_range'])
        gif_frames=[]; end=args.start+args.duration
        video=args.output_dir/f's{subject}_hand_comparison.mp4'
        if args.preview_only:
            for t in [args.start,args.start+4.8,args.start+12]:
                j=round(t*1000)
                render_frame(subject,t,end,truth[j],pred[j],lo,span,row['r4'],args.start).save(args.output_dir/f's{subject}_preview_{t:g}.png')
            continue
        encoder=imageio_ffmpeg.write_frames(str(video),(WIDTH,HEIGHT),fps=args.fps,codec='libx264',
                     pix_fmt_in='rgb24',pix_fmt_out='yuv420p',quality=7,macro_block_size=2,
                     output_params=['-crf','21','-preset','fast','-movflags','+faststart'])
        encoder.send(None)
        count=round(args.duration*args.fps)
        for frame in range(count):
            t=args.start+frame/args.fps; j=round(t*1000)
            img=render_frame(subject,t,end,truth[j],pred[j],lo,span,row['r4'],args.start)
            encoder.send(np.asarray(img))
            if frame%2==0:
                gif_frames.append(img.resize((640,400),Image.Resampling.LANCZOS).quantize(colors=96))
            if frame in [0,count//3,2*count//3]: img.save(args.output_dir/f's{subject}_preview_{frame}.png')
            if frame%125==0: print(f'S{subject}: {frame}/{count} frames',flush=True)
        encoder.close()
        gif_path=args.output_dir/f's{subject}_hand_comparison.gif'
        gif_frames[0].save(gif_path,save_all=True,append_images=gif_frames[1:],loop=0,duration=round(2000/args.fps),optimize=False,disposal=2)
        info=dict(subject=subject,interval_seconds=[args.start,end],interval_rule='User-specified fixed interval, shared across subjects; README examples use 60–80 s, not optimized on predictions.',
                  fps=args.fps,frames=count,playback_speed=1.,prediction_sha256=sha256(pred_path),label_sha256=sha256(label_path),
                  scaler_sha256=sha256(scaler_path),pose_mapping='clip((glove - training_target_min) / training_target_range, 0, 1), same mapping for both hands',
                  visualisation_only=True,scoring_samples=200000,r4=row['r4'],r5=row['r5'],
                  no_prediction_smoothing=True,no_time_shift=True,hand_geometry='Original procedural schematic 3D hand; no anatomical angle calibration',
                  mp4_sha256=sha256(video),gif_sha256=sha256(gif_path),mp4_bytes=video.stat().st_size,gif_bytes=gif_path.stat().st_size)
        all_info.append(info); print(f'S{subject}: MP4 {info["mp4_bytes"]} bytes; GIF {info["gif_bytes"]} bytes',flush=True)
    if not args.preview_only:
        (args.output_dir/'video_manifest.json').write_text(json.dumps(dict(renderer_sha256=sha256(__file__),subjects=all_info),indent=2)+'\n')


if __name__=='__main__': main()
