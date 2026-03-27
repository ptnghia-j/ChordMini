from __future__ import annotations

import os
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.utils import info, warning


PAPER_QUALITY_ORDER = ['Major', 'Minor', 'Dom7', 'Maj7', 'Min7', 'Dim', 'Dim7', 'Aug', 'Sus']
ALL_MAPPED_QUALITY_ORDER = [
    'Major',
    'No Chord',
    'Minor',
    'Min7',
    'Dom7',
    'Maj7',
    'Dim7',
    'Half-Dim',
    'Min6',
    'Dim',
    'Sus',
    'Aug',
    'Maj6',
    'Min-Maj7',
    'Unknown',
    'Other',
]


def extract_chord_quality(chord):
    if not chord:
        return 'N'
    if chord in ['N', 'None', 'NC']:
        return 'N'
    if chord in ['X', 'Unknown']:
        return 'X'

    if ':' in chord:
        parts = chord.split(':')
        if len(parts) > 1:
            quality = parts[1].split('/')[0] if '/' in parts[1] else parts[1]
            return quality

    match = re.match(r'^[A-G][#b]?', chord)
    if match:
        quality = chord[match.end():]
        if quality:
            return quality.split('/')[0] if '/' in quality else quality

    return 'maj'


def map_chord_to_quality(chord_name):
    if not isinstance(chord_name, str):
        return 'Other'
    if chord_name in ['N', 'X', 'None', 'Unknown', 'NC']:
        return 'No Chord'

    quality = extract_chord_quality(chord_name)
    quality_mapping = {
        'maj': 'Major', '': 'Major', 'M': 'Major', 'major': 'Major',
        'min': 'Minor', 'm': 'Minor', 'minor': 'Minor',
        '7': 'Dom7', 'dom7': 'Dom7', 'dominant': 'Dom7',
        'maj7': 'Maj7', 'M7': 'Maj7', 'major7': 'Maj7',
        'min7': 'Min7', 'm7': 'Min7', 'minor7': 'Min7',
        'dim': 'Dim', 'o': 'Dim', 'diminished': 'Dim',
        'dim7': 'Dim7', 'o7': 'Dim7', 'diminished7': 'Dim7',
        'hdim7': 'Half-Dim', 'm7b5': 'Half-Dim', 'half-diminished': 'Half-Dim',
        'aug': 'Aug', '+': 'Aug', 'augmented': 'Aug',
        'sus2': 'Sus', 'sus4': 'Sus', 'sus': 'Sus', 'suspended': 'Sus',
        'min6': 'Min6', 'm6': 'Min6',
        'maj6': 'Maj6', '6': 'Maj6',
        'minmaj7': 'Min-Maj7', 'mmaj7': 'Min-Maj7', 'min-maj7': 'Min-Maj7',
        'N': 'No Chord',
        'X': 'Unknown',
    }
    return quality_mapping.get(quality, 'Other')


def compute_chord_quality_accuracy(reference_labels, prediction_labels):
    raw_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    mapped_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    for ref, pred in zip(reference_labels, prediction_labels):
        if not ref or not pred:
            continue

        q_ref_raw = extract_chord_quality(ref)
        q_pred_raw = extract_chord_quality(pred)
        q_ref_mapped = map_chord_to_quality(ref)
        q_pred_mapped = map_chord_to_quality(pred)

        raw_stats[q_ref_raw]['total'] += 1
        if q_ref_raw == q_pred_raw:
            raw_stats[q_ref_raw]['correct'] += 1

        mapped_stats[q_ref_mapped]['total'] += 1
        if q_ref_mapped == q_pred_mapped:
            mapped_stats[q_ref_mapped]['correct'] += 1

    raw_acc = {}
    mapped_acc = {}

    for quality, vals in raw_stats.items():
        raw_acc[quality] = vals['correct'] / vals['total'] if vals['total'] > 0 else 0.0

    for quality, vals in mapped_stats.items():
        mapped_acc[quality] = vals['correct'] / vals['total'] if vals['total'] > 0 else 0.0

    info('\nRaw Chord Quality Distribution:')
    total_raw = sum(stats['total'] for stats in raw_stats.values())
    for quality, stats in sorted(raw_stats.items(), key=lambda x: x[1]['total'], reverse=True):
        if stats['total'] > 0:
            pct = 100.0 * stats['total'] / max(1, total_raw)
            info(f'  {quality}: {stats["total"]} samples ({pct:.2f}%)')

    info('\nMapped Chord Quality Distribution (matches validation):')
    total_mapped = sum(stats['total'] for stats in mapped_stats.values())
    for quality, stats in sorted(mapped_stats.items(), key=lambda x: x[1]['total'], reverse=True):
        if stats['total'] > 0:
            pct = 100.0 * stats['total'] / max(1, total_mapped)
            info(f'  {quality}: {stats["total"]} samples ({pct:.2f}%)')

    info('\nRaw Accuracy by chord quality:')
    for quality, accuracy_val in sorted(raw_acc.items(), key=lambda x: x[1], reverse=True):
        if raw_stats[quality]['total'] >= 10:
            info(f'  {quality}: {accuracy_val:.4f}')

    info('\nMapped Accuracy by chord quality (matches validation):')
    for quality, accuracy_val in sorted(mapped_acc.items(), key=lambda x: x[1], reverse=True):
        if mapped_stats[quality]['total'] >= 10:
            info(f'  {quality}: {accuracy_val:.4f}')

    return mapped_acc, mapped_stats


def compute_macro_frame_metrics(reference_labels, prediction_labels):
    min_len = min(len(reference_labels), len(prediction_labels))
    if min_len == 0:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    ref = list(reference_labels[:min_len])
    pred = list(prediction_labels[:min_len])
    labels = sorted(set(ref) | set(pred))

    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0
    total_support = 0
    correct = sum(int(ref_label == pred_label) for ref_label, pred_label in zip(ref, pred))

    for label in labels:
        tp = 0
        fp = 0
        fn = 0
        for ref_label, pred_label in zip(ref, pred):
            if pred_label == label and ref_label == label:
                tp += 1
            elif pred_label == label and ref_label != label:
                fp += 1
            elif ref_label == label and pred_label != label:
                fn += 1

        support = tp + fn
        if support == 0:
            continue

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support
        total_support += support

    return {
        'accuracy': correct / float(min_len),
        'precision': (weighted_precision / float(total_support)) if total_support > 0 else 0.0,
        'recall': (weighted_recall / float(total_support)) if total_support > 0 else 0.0,
        'f1': (weighted_f1 / float(total_support)) if total_support > 0 else 0.0,
    }


def compute_paper_quality_metrics(per_song, quality_order=None):
    if quality_order is None:
        observed = set()
        for song in per_song:
            observed.update(map_chord_to_quality(label) for label in song['ref_labels'])
        quality_order = [quality for quality in ALL_MAPPED_QUALITY_ORDER if quality in observed]
        quality_order.extend(sorted(observed.difference(quality_order)))
    else:
        quality_order = list(quality_order)

    quality_accuracy = {}
    quality_stats = {}
    wcsr_by_quality = {}

    for quality in quality_order:
        total = 0
        correct = 0

        for song in per_song:
            ref_quality = [map_chord_to_quality(label) for label in song['ref_labels']]
            pred_quality = [map_chord_to_quality(label) for label in song['pred_labels']]

            local_total = 0
            local_correct = 0
            for ref_label, pred_label in zip(ref_quality, pred_quality):
                if ref_label != quality:
                    continue
                local_total += 1
                if pred_label == quality:
                    local_correct += 1

            total += local_total
            correct += local_correct

        quality_accuracy[quality] = (correct / float(total)) if total > 0 else 0.0
        quality_stats[quality] = {'total': int(total), 'correct': int(correct)}
        wcsr_by_quality[quality] = quality_accuracy[quality]

    present = [quality for quality in quality_order if quality_stats[quality]['total'] > 0]
    total_weight = sum(quality_stats[quality]['total'] for quality in present)
    aggregate_wcsr = (
        sum(wcsr_by_quality[quality] * quality_stats[quality]['total'] for quality in present) / float(total_weight)
        if total_weight > 0 else 0.0
    )
    acqa = float(np.mean([wcsr_by_quality[quality] for quality in present])) if present else 0.0

    return {
        'quality_accuracy': quality_accuracy,
        'quality_stats': quality_stats,
        'wcsr_by_quality': wcsr_by_quality,
        'wcsr': aggregate_wcsr,
        'acqa': acqa,
    }


def generate_chord_distribution_accuracy_plot(quality_stats, quality_accuracy, output_path, title=None):
    qualities = []
    counts = []
    accuracies = []
    total_samples = sum(stats['total'] for stats in quality_stats.values())

    sorted_qualities = sorted(quality_stats.keys(), key=lambda q: quality_stats[q]['total'], reverse=True)
    for quality in sorted_qualities:
        if quality_stats[quality]['total'] >= 10:
            qualities.append(quality)
            counts.append(quality_stats[quality]['total'])
            accuracies.append(quality_accuracy.get(quality, 0.0))

    if not qualities:
        warning('No chord qualities with sufficient samples for plotting')
        return None

    percentages = [100.0 * c / max(1, total_samples) for c in counts]

    fig, ax1 = plt.subplots(figsize=(14, 8))
    bars = ax1.bar(qualities, percentages, alpha=0.7, color='steelblue', label='Distribution (%)')
    ax1.set_ylabel('Distribution (%)', color='steelblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_ylim(0, max(percentages) * 1.2 if percentages else 10)

    for bar, pct, count in zip(bars, percentages, counts):
        height = bar.get_height()
        ax1.annotate(
            f'{pct:.1f}%\n({count})',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords='offset points',
            ha='center',
            va='bottom',
            color='steelblue',
            fontsize=10,
        )

    ax2 = ax1.twinx()
    ax2.plot(qualities, accuracies, 'ro-', linewidth=2, markersize=8, label='Accuracy')
    ax2.set_ylabel('Accuracy', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 1.0)

    for i, acc in enumerate(accuracies):
        ax2.annotate(
            f'{acc:.2f}',
            xy=(i, acc),
            xytext=(0, 5),
            textcoords='offset points',
            ha='center',
            va='bottom',
            color='red',
            fontsize=10,
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title(title or 'Chord Quality Distribution and Accuracy', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    info(f'Saved chord distribution and accuracy plot to {output_path}')
    return output_path


def generate_confusion_matrix_heatmap(ref_labels, pred_labels, output_path, title=None):
    ref_qualities = [map_chord_to_quality(ch) for ch in ref_labels]
    pred_qualities = [map_chord_to_quality(ch) for ch in pred_labels]

    quality_counts = Counter(ref_qualities)
    unique_qualities = sorted(quality_counts.keys(), key=lambda q: quality_counts[q], reverse=True)
    filtered_qualities = [q for q in unique_qualities if quality_counts[q] >= 10]

    if not filtered_qualities:
        warning('No chord qualities with sufficient samples for confusion matrix')
        return None

    quality_to_idx = {q: i for i, q in enumerate(filtered_qualities)}
    filtered_indices = [
        i for i, (r, p) in enumerate(zip(ref_qualities, pred_qualities))
        if r in filtered_qualities and p in filtered_qualities
    ]

    filtered_ref = [ref_qualities[i] for i in filtered_indices]
    filtered_pred = [pred_qualities[i] for i in filtered_indices]
    ref_indices = [quality_to_idx[q] for q in filtered_ref]
    pred_indices = [quality_to_idx[q] for q in filtered_pred]

    cm = confusion_matrix(ref_indices, pred_indices, labels=range(len(filtered_qualities)))
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=filtered_qualities,
        yticklabels=filtered_qualities,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8},
    )

    plt.tight_layout()
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(title or 'Chord Quality Confusion Matrix', fontsize=14)

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    info(f'Saved chord quality confusion matrix to {output_path}')
    return output_path


__all__ = [
    'PAPER_QUALITY_ORDER',
    'ALL_MAPPED_QUALITY_ORDER',
    'compute_macro_frame_metrics',
    'compute_paper_quality_metrics',
    'compute_chord_quality_accuracy',
    'extract_chord_quality',
    'generate_chord_distribution_accuracy_plot',
    'generate_confusion_matrix_heatmap',
    'map_chord_to_quality',
]
