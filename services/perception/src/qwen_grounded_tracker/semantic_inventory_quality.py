"""Conservative post-filter for fresh grounding, retaining raw evidence for audit."""
from copy import deepcopy
import math


CONFLICT_PAIRS = {
    frozenset(('outlet', 'switch')),
    frozenset(('table', 'cabinet')),
    frozenset(('door', 'window')),
    frozenset(('door', 'sign')),
    frozenset(('window', 'sign')),
}


def iou(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.


def filter_observations(observations, min_confidence=.65):
    kept, rejected = [], []
    for original in observations:
        confidence = original.get('confidence', 0)
        if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or confidence < min_confidence:
            rejected.append({'observation_id': original.get('observation_id'), 'reason': 'low_confidence',
                             'name': original.get('name'), 'confidence': confidence})
            continue
        item = deepcopy(original)
        valid = []
        for view in item.get('views', []):
            box = view.get('bbox_1000')
            if (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float))
                    and math.isfinite(v) and 0 <= v <= 1000 for v in box)
                    and box[0] < box[2] and box[1] < box[3] and view.get('image')):
                valid.append(view)
        if not valid:
            rejected.append({'observation_id': item.get('observation_id'), 'reason': 'no_valid_image_box', 'name': item.get('name')})
            continue
        item['views'] = valid
        kept.append(item)
    # Grounding currently emits one view per observation. Index by photo so
    # across-photo comparisons cannot accidentally merge different objects.
    parent = list(range(len(kept)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    conflicts = []
    by_image = {}
    for index, item in enumerate(kept):
        if len(item['views']) != 1:
            continue
        view = item['views'][0]
        for other_index in by_image.get(view['image'], []):
            other = kept[other_index]
            overlap = iou(view['bbox_1000'], other['views'][0]['bbox_1000'])
            pair = frozenset((item['category'], other['category']))
            duplicate = item['category'] == other['category'] and overlap >= .85
            incompatible = pair in CONFLICT_PAIRS and overlap >= .75
            if duplicate or incompatible:
                parent[find(index)] = find(other_index)
                if incompatible:
                    conflicts.append({'image': view['image'], 'categories': sorted(pair),
                                      'overlap_iou': overlap, 'reason': 'same_region_incompatible_labels'})
        by_image.setdefault(view['image'], []).append(index)
    groups = {}
    for index in range(len(kept)):
        groups.setdefault(find(index), []).append(kept[index])
    result = []
    for group in groups.values():
        best = deepcopy(max(group, key=lambda item: (item['confidence'],
            -((item['views'][0]['bbox_1000'][2] - item['views'][0]['bbox_1000'][0])
              * (item['views'][0]['bbox_1000'][3] - item['views'][0]['bbox_1000'][1])))))
        categories = sorted({item['category'] for item in group})
        best['merged_observation_ids'] = sorted({item['observation_id'] for item in group})
        if len(categories) > 1:
            names = sorted({item['name'] for item in group})
            best.update(name=' / '.join(names) + '（类别待确认）', category='unknown_object',
                        semantic_ambiguous=True, alternative_categories=categories,
                        alternative_names=names, mobility='unknown')
        result.append(best)
    return result, {'raw_observation_count': len(observations), 'filtered_observation_count': len(result),
                    'rejected': rejected, 'conflicts': conflicts,
                    'ambiguous_regions': sum(item.get('semantic_ambiguous') is True for item in result),
                    'zero_or_low_confidence_is_not_a_confirmed_object': True}


def remove_ambiguous_world_positions(catalog):
    for item in catalog['objects']:
        if any(e.get('semantic_ambiguous') for e in item['evidence']):
            item.update(map_xy=None, approach_xy=None, position_status='unlocalized',
                        position_method='incompatible_image_labels_need_review',
                        position_uncertainty_m=None, semantic_ambiguous=True,
                        navigation_allowed=False, localization_allowed=False, grasp_allowed=False)
            item['aliases'] = sorted(set(item['aliases'] + [name for e in item['evidence'] for name in e.get('alternative_names', [])]))
    catalog['search_regions'] = [region for region in catalog['search_regions']
        if any(item['id'] == region['id'] and item['map_xy'] is not None for item in catalog['objects'])]
    return catalog
