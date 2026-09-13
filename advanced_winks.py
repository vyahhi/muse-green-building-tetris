"""Filtered, personalized EOG epoch classifier for Muse frontal channels."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt, welch


FS = 256
LABELS = ("left", "right", "both")


def _filter(data: np.ndarray, low: float, high: float) -> np.ndarray:
    sos = butter(4, (low, high), btype="bandpass", fs=FS, output="sos")
    return sosfiltfilt(sos, data, axis=1)


def _features(epoch: np.ndarray, scales: np.ndarray) -> np.ndarray:
    low = _filter(epoch, 1.0, 10.0) / scales[:, None]
    muscle = _filter(epoch, 18.0, 45.0) / scales[:, None]
    left, right = low
    common = (left + right) * 0.5
    lateral = (left - right) * 0.5
    values: list[float] = []
    for signal in (left, right, common, lateral):
        values.extend((
            float(np.max(signal)), float(np.min(signal)), float(np.ptp(signal)),
            float(np.sqrt(np.mean(signal ** 2))), float(np.mean(np.abs(signal))),
            float(np.argmax(np.abs(signal)) / max(len(signal) - 1, 1)),
        ))
    values.extend((
        float(np.corrcoef(left, right)[0, 1]),
        float(np.sqrt(np.mean(muscle[0] ** 2))),
        float(np.sqrt(np.mean(muscle[1] ** 2))),
        float(np.sqrt(np.mean(np.diff(left) ** 2))),
        float(np.sqrt(np.mean(np.diff(right) ** 2))),
    ))
    # Relative low-frequency spectral shape helps separate EOG from motion.
    for signal in (left, right):
        freqs, power = welch(signal, fs=FS, nperseg=min(128, len(signal)))
        total = max(float(np.sum(power[(freqs >= 1) & (freqs <= 10)])), 1e-9)
        values.append(float(np.sum(power[(freqs >= 1) & (freqs < 4)]) / total))
        values.append(float(np.sum(power[(freqs >= 4) & (freqs <= 10)]) / total))
    return np.nan_to_num(np.asarray(values), nan=0.0, posinf=0.0, neginf=0.0)


def _robust_scale(baseline: np.ndarray) -> np.ndarray:
    filtered = _filter(baseline, 1.0, 10.0)
    median = np.median(filtered, axis=1, keepdims=True)
    mad = np.median(np.abs(filtered - median), axis=1) * 1.4826
    return np.maximum(mad, np.std(filtered, axis=1) * 0.25 + 1e-6)


def _candidate_epochs(data: np.ndarray, scales: np.ndarray, count: int = 6):
    low = _filter(data, 1.0, 10.0) / scales[:, None]
    envelope = np.maximum(np.abs(low[0]), np.abs(low[1]))
    peaks, props = find_peaks(envelope, distance=int(0.55 * FS), prominence=0.8)
    if not len(peaks):
        return [], np.empty(0)
    order = np.argsort(props["prominences"])[::-1]
    half = int(0.35 * FS)
    epochs, strengths = [], []
    for idx in order:
        peak = int(peaks[idx])
        if peak - half < 0 or peak + half > data.shape[1]:
            continue
        epoch = data[:, peak - half:peak + half]
        # Reject BLE discontinuities: a single near-full-scale jump is not EOG.
        if np.max(np.abs(np.diff(epoch, axis=1))) > 1700:
            continue
        epochs.append(epoch)
        strengths.append(float(envelope[peak]))
        if len(epochs) >= count:
            break
    return epochs, np.asarray(strengths)


@dataclass
class WinkProfile:
    scales: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    centroids: np.ndarray
    event_threshold: float
    accuracy: float
    max_distance: float = 6.5

    def predict(self, epoch: np.ndarray) -> tuple[str, np.ndarray]:
        feature = (_features(epoch, self.scales) - self.feature_mean) / self.feature_scale
        distances = np.sqrt(np.sum((self.centroids - feature) ** 2, axis=1))
        return LABELS[int(np.argmin(distances))], distances

    @classmethod
    def load(cls, path: str | Path) -> "WinkProfile":
        values = json.loads(Path(path).read_text())
        return cls(
            scales=np.asarray(values["scales"]),
            feature_mean=np.asarray(values["feature_mean"]),
            feature_scale=np.asarray(values["feature_scale"]),
            centroids=np.asarray(values["centroids"]),
            event_threshold=float(values["event_threshold"]),
            accuracy=float(values["accuracy"]),
            max_distance=float(values.get("max_distance", 6.5)),
        )


class OnlineWinkDetector:
    """Detect completed filtered EOG peaks and classify their full epochs."""

    def __init__(
        self,
        profile: WinkProfile,
        confidence: float = 0.14,
        refractory_seconds: float = 0.7,
        enforce_profile_distance: bool = True,
    ) -> None:
        self.profile = profile
        self.confidence = confidence
        self.refractory_samples = max(1, int(refractory_seconds * FS))
        self.enforce_profile_distance = enforce_profile_distance
        self.data = np.empty((2, 0))
        self.total_samples = 0
        self.last_peak = -FS

    def add(self, data: np.ndarray) -> None:
        self.total_samples += data.shape[1]
        self.data = np.concatenate((self.data, data), axis=1)[:, -FS * 4:]

    def detect(self) -> tuple[str | None, np.ndarray | None, float]:
        half = int(0.35 * FS)
        if self.data.shape[1] < FS:
            return None, None, 0.0
        low = _filter(self.data, 1.0, 10.0) / self.profile.scales[:, None]
        envelope = np.maximum(np.abs(low[0]), np.abs(low[1]))
        peaks, props = find_peaks(
            envelope,
            height=self.profile.event_threshold,
            prominence=max(0.65, self.profile.event_threshold * 0.25),
            distance=self.refractory_samples,
        )
        for peak in reversed(peaks.tolist()):
            if peak < half or peak + half >= self.data.shape[1]:
                continue
            global_peak = self.total_samples - self.data.shape[1] + peak
            if global_peak - self.last_peak < self.refractory_samples:
                continue
            epoch = self.data[:, peak - half:peak + half]
            if np.max(np.abs(np.diff(epoch, axis=1))) > 1700:
                self.last_peak = global_peak
                return None, None, 0.0
            label, distances = self.profile.predict(epoch)
            ordered = np.sort(distances)
            margin = float((ordered[1] - ordered[0]) / max(ordered[1], 1e-6))
            self.last_peak = global_peak
            # Direction alone is not enough: ordinary facial motion can lean
            # toward a centroid while remaining unlike every trained wink.
            if (
                (self.enforce_profile_distance
                 and ordered[0] > self.profile.max_distance)
                or margin < self.confidence
            ):
                return None, distances, margin
            return label, distances, margin
        return None, None, 0.0


def train_profile(
    baseline: np.ndarray, labeled: dict[str, np.ndarray]
) -> tuple[WinkProfile, dict[str, int]]:
    scales = _robust_scale(baseline)
    features, labels, counts, strengths = [], [], {}, []
    for label_index, label in enumerate(LABELS):
        epochs, class_strengths = _candidate_epochs(labeled[label], scales)
        counts[label] = len(epochs)
        strengths.extend(class_strengths.tolist())
        for epoch in epochs:
            features.append(_features(epoch, scales))
            labels.append(label_index)
    if min(counts.values(), default=0) < 3:
        raise RuntimeError(f"not enough clean wink events: {counts}")
    x = np.asarray(features)
    y = np.asarray(labels)
    mean = np.mean(x, axis=0)
    scale = np.maximum(np.std(x, axis=0), 1e-5)
    z = (x - mean) / scale
    centroids = np.asarray([np.mean(z[y == idx], axis=0) for idx in range(3)])
    predictions = np.argmin(
        np.sqrt(np.sum((z[:, None, :] - centroids[None, :, :]) ** 2, axis=2)), axis=1
    )
    accuracy = float(np.mean(predictions == y))
    own_distances = np.sqrt(np.sum((z - centroids[y]) ** 2, axis=1))
    max_distance = float(np.clip(np.percentile(own_distances, 90) * 1.25, 3.0, 7.5))
    base_low = _filter(baseline, 1.0, 10.0) / scales[:, None]
    base_envelope = np.maximum(np.abs(base_low[0]), np.abs(base_low[1]))
    threshold = max(float(np.percentile(base_envelope, 99.5) * 1.15), 2.5)
    if strengths:
        threshold = min(threshold, float(np.percentile(strengths, 20) * 0.75))
    return WinkProfile(
        scales, mean, scale, centroids, threshold, accuracy, max_distance
    ), counts
