"""Native acoustic/facial features with decoded PTS provenance.

The common clock is min(first retained audio sample PTS, first decoded video PTS).
Frames retain their actual display support; optional subsampling never extends
a frame across a sampling period. Missing faces have zero values and a
False mask. Quality values are declared engineering proxies, not calibrated
model probabilities. No sentiment labels are used here.
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import av
import cv2
import mediapipe as mp
import opensmile


BLENDSHAPE_NAMES = [
    '_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp',
    'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft',
    'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft',
    'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft',
    'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft',
    'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft',
    'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight',
    'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel', 'mouthLeft',
    'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft',
    'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower',
    'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft',
    'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight',
    'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight',
]


def _pts_seconds(frame):
    if frame.pts is None or frame.time_base is None:
        raise ValueError('Decoded frame has no PTS/time_base; cannot audit common clock')
    return float(frame.pts * frame.time_base)


def _stream_metadata(stream):
    tb = stream.time_base
    return dict(index=stream.index, time_base=[tb.numerator, tb.denominator],
                start_pts=stream.start_time, duration_pts=stream.duration,
                codec=stream.codec_context.name)


def _put_audio_chunks(chunks, sample_rate, end_limit=None, start_limit=None):
    """Place resampled chunks on PTS clock, preserving gaps and overlap audit.

    The first chunk's PTS, cropped to a declared presentation start when given,
    defines sample zero. Later timestamps are rounded to the nearest sample;
    earlier samples win in true overlaps. A gap remains zero
    with False sample mask. Returns waveform, sample mask, placement records.
    """
    if not chunks:
        raise ValueError('No decoded/resampled audio')
    t0 = min(t for t, _ in chunks)
    if start_limit is not None:
        t0 = max(t0,float(start_limit))
    placements = [(int(round((t - t0) * sample_rate)), np.asarray(x, np.float32))
                  for t, x in chunks]
    count = max(start + len(x) for start, x in placements)
    if end_limit is not None and end_limit > t0:
        count = min(count, max(1, int(round((end_limit - t0) * sample_rate))))
    output = np.zeros(count, np.float32)
    valid = np.zeros(count, bool)
    records = []
    for (t, _), (start, x) in zip(chunks, placements):
        left, right = max(0, start), min(count, start + len(x))
        if right <= left:
            records.append([t, start, len(x), 0, 0])
            continue
        occupied = valid[left:right].copy()
        output[left:right][~occupied] = x[left-start:right-start][~occupied]
        valid[left:right] = True
        records.append([t, start, len(x), right-left, int(occupied.sum())])
    return output, valid, t0, np.asarray(records, np.float64)


def _bbox(landmarks):
    coords = np.array([(p.x, p.y) for p in landmarks], np.float64)
    return np.clip(np.r_[coords.min(0), coords.max(0)], 0, 1)


def _iou(a, b):
    left = np.maximum(a[:2], b[:2]); right = np.minimum(a[2:], b[2:])
    intersection = float(np.prod(np.maximum(right-left, 0)))
    area_a = float(np.prod(np.maximum(a[2:]-a[:2], 0)))
    area_b = float(np.prod(np.maximum(b[2:]-b[:2], 0)))
    return intersection / max(area_a + area_b - intersection, 1e-12)


class NativeMediaExtractor:
    def __init__(self, model_path, config):
        self.model_path = str(model_path)
        # Python opens Unicode Windows paths reliably; native model_asset_path
        # fails on some MediaPipe Windows builds even when the file exists.
        self.model_asset_buffer = Path(model_path).read_bytes()
        self.cfg = config
        self.sr = int(config.get('sample_rate', 16000))
        self.fps = float(config.get('video_sample_fps', 0))
        if self.sr != 16000 or self.fps < 0:
            raise ValueError('Use 16-kHz audio; video_sample_fps=0 means every source frame')
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
            num_workers=1,
        )
        if len(self.smile.feature_names) != 25:
            raise ValueError('Expected eGeMAPSv02 25 LLDs')

    def _audio(self, path):
        chunks, source = [], []
        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            meta = _stream_metadata(stream)
            declared_start = (float(stream.start_time*stream.time_base)
                              if stream.start_time is not None else None)
            meta.update(container_start_seconds=(container.start_time/av.time_base if container.start_time is not None else None),
                        container_duration_seconds=(container.duration/av.time_base if container.duration is not None else None),
                        declared_start_seconds=declared_start,
                        original_sample_rate=stream.codec_context.sample_rate,
                        original_channels=stream.codec_context.channels)
            end_limit = (float((stream.start_time + stream.duration) * stream.time_base)
                         if stream.start_time is not None and stream.duration is not None else None)
            resampler = av.AudioResampler(format='fltp', layout='mono', rate=self.sr)
            for index, frame in enumerate(container.decode(stream)):
                absolute = _pts_seconds(frame)
                source.append([index, frame.pts, frame.time_base.numerator,
                               frame.time_base.denominator, frame.samples,
                               frame.sample_rate, absolute, absolute+frame.samples/frame.sample_rate])
                for out in resampler.resample(frame):
                    chunks.append((_pts_seconds(out), out.to_ndarray().reshape(-1)))
            for out in resampler.resample(None):
                chunks.append((_pts_seconds(out), out.to_ndarray().reshape(-1)))
        waveform, observed, t0, placements = _put_audio_chunks(chunks, self.sr, end_limit,declared_start)
        if not np.isfinite(waveform).all():
            raise ValueError('Decoded audio has nonfinite samples')
        meta.update(decoded_frame_count=len(source), resampled_chunk_count=len(chunks),
                    sample_rate=self.sr, channels=1, end_limit_seconds=end_limit,
                    gap_samples=int((~observed).sum()), overlapping_samples=int(placements[:,4].sum()),
                    first_decoded_seconds=float(source[0][6]),
                    first_resampled_seconds=float(chunks[0][0]),
                    presentation_start_seconds=t0,
                    discarded_preroll_seconds=max(0.,t0-min(t for t,_ in chunks)),
                    timing_note='PyAV decoded PTS already reflects codec skip metadata; any residual samples before declared stream presentation start are cropped; source/chunk PTS retained',
                    peak_absolute_amplitude=float(np.max(abs(waveform))))
        return waveform, observed, t0, np.asarray(source, np.float64), placements, meta

    def _video(self, path):
        """Stream every frame (fps=0), or first frame at/after a sampling grid.

        The full frame clock is always retained. Images are inferred and
        discarded one at a time, so full-frame extraction has bounded memory.
        """
        all_records, sampled, rows = [], [], []
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=self.model_asset_buffer),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=int(self.cfg.get('face_num_faces', 3)),
            min_face_detection_confidence=float(self.cfg.get('face_min_detection_confidence', .5)),
            min_face_presence_confidence=float(self.cfg.get('face_min_presence_confidence', .5)),
            min_tracking_confidence=float(self.cfg.get('face_min_tracking_confidence', .5)),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        previous_box = None; missing_run = 0
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as model, av.open(str(path)) as container:
            stream = container.streams.video[0]
            meta = _stream_metadata(stream)
            rate = float(stream.average_rate or stream.guessed_rate or 25.)
            grid = None; first = None; previous = None
            for index, frame in enumerate(container.decode(stream)):
                absolute = _pts_seconds(frame)
                if previous is not None and absolute <= previous:
                    raise ValueError('Nonincreasing decoded video PTS')
                previous = absolute
                if first is None:
                    first = absolute; grid = first
                duration = float(frame.duration * frame.time_base) if frame.duration else 1/rate
                all_records.append([index,frame.pts,frame.time_base.numerator,
                                    frame.time_base.denominator,absolute,duration])
                if self.fps and absolute+1e-9 < grid:
                    continue
                sampled.append([index,grid if self.fps else absolute,absolute])
                if self.fps:
                    grid = first+(math.floor((absolute-first)*self.fps+1e-7)+1)/self.fps
                row = dict(features=np.zeros(52,np.float32),valid=False,quality=0.,
                           bbox=np.zeros(4,np.float32),components=np.zeros(6,np.float32),reset=False)
                rows.append(row)
                rgb = frame.to_ndarray(format='rgb24')
                result = model.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                boxes = [_bbox(lm) for lm in result.face_landmarks]
                if not boxes:
                    missing_run += 1
                    if missing_run >= 3:
                        previous_box = None
                    continue
                areas = np.array([np.prod(b[2:]-b[:2]) for b in boxes])
                ious = np.array([_iou(b,previous_box) if previous_box is not None else 0. for b in boxes])
                if previous_box is not None and ious.max() >= .1:
                    choice = int(np.argmax(ious))
                else:
                    choice = int(np.argmax(areas)); row['reset'] = True
                box = boxes[choice]
                if choice >= len(result.face_blendshapes):
                    raise ValueError('Face landmarks/blendshapes row mismatch')
                scores = {s.category_name: float(s.score) for s in result.face_blendshapes[choice]}
                if set(scores) != set(BLENDSHAPE_NAMES):
                    raise ValueError('Unexpected MediaPipe blendshape vocabulary')
                row['features'] = np.array([scores[name] for name in BLENDSHAPE_NAMES],np.float32)
                height, width = rgb.shape[:2]
                x0,y0 = np.floor(box[:2]*[width,height]).astype(int)
                x1,y1 = np.ceil(box[2:]*[width,height]).astype(int)
                crop = rgb[max(0,y0):min(height,y1),max(0,x0):min(width,x1)]
                if not crop.size:
                    row['features'][:] = 0
                    continue
                gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                exposure = float(np.mean((gray>5)&(gray<250)))
                area = float(areas[choice])
                area_score = min(1., math.sqrt(area/.1))
                sharpness_score = blur/(blur+100.)
                row['quality'] = np.clip(area_score*sharpness_score*exposure,.01,1.)
                row['bbox'] = box; row['valid'] = True
                row['components'] = [area,blur,exposure,ious[choice],len(boxes),choice]
                previous_box = box; missing_run = 0
        if not all_records:
            raise ValueError('No decoded video frames')
        all_records = np.asarray(all_records,np.float64)
        ends = np.r_[all_records[1:,4],all_records[-1,4]+all_records[-1,5]]
        if meta['start_pts'] is not None and meta['duration_pts'] is not None:
            end_limit = (meta['start_pts']+meta['duration_pts'])*meta['time_base'][0]/meta['time_base'][1]
            if end_limit>all_records[-1,4]:
                ends[-1] = min(ends[-1],end_limit)
        all_records[:,5] = ends-all_records[:,4]
        n = len(rows)
        features = np.asarray([r['features'] for r in rows],np.float32)
        valid = np.asarray([r['valid'] for r in rows],bool)
        quality = np.asarray([r['quality'] for r in rows],np.float32)
        bbox = np.asarray([r['bbox'] for r in rows],np.float32)
        components = np.asarray([r['components'] for r in rows],np.float32)
        resets = np.asarray([r['reset'] for r in rows],bool)
        sampled = np.asarray(sampled, np.float64)
        selected_indices = sampled[:,0].astype(np.int64)
        intervals = np.column_stack((all_records[selected_indices,4],ends[selected_indices]))
        meta.update(decoded_frame_count=len(all_records),sampled_frame_count=n,
                    nominal_fps=rate,sample_fps=self.fps,face_observed_count=int(valid.sum()),
                    model_mode='IMAGE; independent inference followed by bbox continuity rule',
                    selector='prefer maximum bbox IoU >= 0.1 with last selected face; else largest face; reset after 3 missing sampled frames',
                    selector_limit='primary visible face is not guaranteed to be the speaking person',
                    quality_proxy='clip(sqrt(min(area_fraction/0.1,1))*laplacian_variance/(laplacian_variance+100)*unsaturated_fraction,0.01,1); absent face=0',
                    sample_rule=('all original decoded frames; exact original display support' if self.fps==0 else 'first decoded frame at/after each grid time; never duplicate; exact original frame display support'),
                    last_display_end=float(ends[-1]))
        return dict(visual_native=features,visual_intervals=intervals,
                    visual_quality=quality,visual_native_mask=valid,
                    visual_source_frame_index=selected_indices,
                    visual_source_pts=all_records[selected_indices,1].astype(np.int64),
                    visual_source_time_base=all_records[selected_indices,2:4].astype(np.int64),
                    visual_target_grid_time=sampled[:,1],
                    visual_all_frame_records=all_records,
                    visual_bbox=bbox,visual_quality_components=components,
                    visual_track_reset=resets),meta,float(all_records[0,4]),float(ends[-1])

    def extract(self, path):
        waveform, sample_mask, audio_start, source, placements, ameta = self._audio(path)
        arrays,vmeta,video_start,video_end = self._video(path)
        audio_end = audio_start+len(waveform)/self.sr
        origin = min(audio_start,video_start)
        df = self.smile.process_signal(waveform,self.sr)
        starts = np.asarray(df.index.get_level_values('start').total_seconds())
        ends = np.asarray(df.index.get_level_values('end').total_seconds())
        features = df.to_numpy(dtype=np.float32,copy=True)
        if len(features) == 0 or features.shape[1] != 25:
            raise ValueError('No 25-dimensional eGeMAPSv02 LLD sequence')
        # openSMILE index intervals are retained verbatim, then clipped at the
        # physical audio end. Some LLDs use larger internal analysis windows;
        # the config/hash, not this nominal output interval, defines context.
        ends = np.minimum(ends,len(waveform)/self.sr)
        quality = np.zeros(len(features),np.float32)
        valid = np.isfinite(features).all(1) & (ends>starts)
        components = np.zeros((len(features),3),np.float32)
        for i,(a,b) in enumerate(zip(starts,ends)):
            l = max(0,int(math.floor(a*self.sr)));r = min(len(waveform),int(math.ceil(b*self.sr)))
            if r<=l:
                valid[i]=False;continue
            coverage = float(sample_mask[l:r].mean())
            clipping = float(np.mean(abs(waveform[l:r])>=.999))
            rms = float(np.sqrt(np.mean(waveform[l:r].astype(np.float64)**2)))
            components[i] = [coverage,clipping,20*np.log10(max(rms,1e-12))]
            valid[i] &= coverage >= 1.-1e-7
            quality[i] = coverage*(1.-clipping) if valid[i] else 0.
        features[~valid]=0
        arrays.update(audio_native=features,audio_intervals=np.column_stack((starts,ends))+audio_start-origin,
                      audio_quality=quality,audio_native_mask=valid,audio_quality_components=components,
                      audio_source_frame_records=source,audio_resampled_chunk_records=placements,
                      audio_sample_observed_mask=sample_mask)
        arrays['visual_intervals'] -= origin
        arrays['visual_target_grid_time'] -= origin
        metadata = dict(duration=max(audio_end,video_end)-origin,
                        media_origin_absolute_seconds=origin,
                        audio_start=audio_start-origin,audio_end=audio_end-origin,
                        video_start=video_start-origin,video_end=video_end-origin,
                        audio_effective_duration=audio_end-audio_start,
                        video_effective_duration=video_end-video_start,
                        clock='seconds relative to min(first resampled audio PTS,first decoded video PTS); raw PTS and time bases retained',
                        audio=ameta,video=vmeta,
                        feature_names={'audio':list(self.smile.feature_names),'visual':BLENDSHAPE_NAMES},
                        source_record_columns={
                            'audio_source_frame_records':['decode_index','pts','time_base_numerator','time_base_denominator','samples','sample_rate','absolute_start_seconds','absolute_end_seconds'],
                            'audio_resampled_chunk_records':['absolute_start_seconds','waveform_start_sample','chunk_samples','retained_samples','overlap_samples'],
                            'visual_all_frame_records':['decode_index','pts','time_base_numerator','time_base_denominator','absolute_start_seconds','display_duration_seconds'],
                            'audio_quality_components':['observed_sample_fraction','clipped_sample_fraction','rms_dbfs'],
                            'visual_quality_components':['bbox_area_fraction','laplacian_variance','unsaturated_fraction','previous_bbox_iou','face_count','selected_face_index']},
                        acoustic_quality_proxy='nominal output interval observed_fraction*(1-clipped_fraction); silence remains an observation; nonfinite/incomplete rows have mask False',
                        opensmile_time_notes='Timedelta output intervals (typically 20 ms with 10 ms hop) are nominal scheduler supports; internal pitch/jitter analysis can use longer context; no claim of exact acoustic receptive field')
        declared_video_end = ((vmeta['start_pts']+vmeta['duration_pts'])*vmeta['time_base'][0]/vmeta['time_base'][1]
                              if vmeta['start_pts'] is not None and vmeta['duration_pts'] is not None else None)
        metadata['duration_audit'] = dict(
            container_declared_seconds=ameta['container_duration_seconds'],
            observed_presentation_span_seconds=metadata['duration'],
            container_minus_observed_seconds=(ameta['container_duration_seconds']-metadata['duration'] if ameta['container_duration_seconds'] is not None else None),
            audio_declared_minus_observed_end_seconds=(ameta['end_limit_seconds']-audio_end if ameta['end_limit_seconds'] is not None else None),
            video_declared_minus_observed_end_seconds=(declared_video_end-video_end if declared_video_end is not None else None),
            audio_video_end_difference_seconds=audio_end-video_end)
        metadata['timing_flags'] = []
        if ameta['discarded_preroll_seconds']>1/self.sr:
            metadata['timing_flags'].append('decoded_preroll_cropped_to_declared_presentation_start')
        if ameta['gap_samples']:
            metadata['timing_flags'].append('audio_pts_gap_zero_filled_with_false_mask')
        if ameta['overlapping_samples']:
            metadata['timing_flags'].append('audio_pts_overlap_first_observation_retained')
        if abs(metadata['duration_audit']['container_minus_observed_seconds'] or 0)>1e-3:
            metadata['timing_flags'].append('container_and_observed_duration_differ')
        return arrays,metadata,waveform
