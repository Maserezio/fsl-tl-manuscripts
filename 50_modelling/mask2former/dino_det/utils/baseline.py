import math

import cv2
import networkx as nx
import numpy as np

from skimage.morphology import skeletonize


def _neighbors(y, x, height, width):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny = y + dy
            nx_ = x + dx
            if 0 <= ny < height and 0 <= nx_ < width:
                yield ny, nx_


def _build_skeleton_graph(skeleton):
    graph = nx.Graph()
    height, width = skeleton.shape
    ys, xs = np.nonzero(skeleton)
    for y, x in zip(ys, xs):
        graph.add_node((int(y), int(x)))
        for ny, nx_ in _neighbors(y, x, height, width):
            if skeleton[ny, nx_]:
                graph.add_edge((int(y), int(x)), (int(ny), int(nx_)), weight=math.hypot(ny - y, nx_ - x))
    return graph


def _farthest_node(graph, source):
    lengths = nx.single_source_dijkstra_path_length(graph, source, weight="weight")
    return max(lengths, key=lengths.get)


def _extract_longest_path(graph):
    if graph.number_of_nodes() == 0:
        return []

    endpoints = [node for node in graph.nodes if graph.degree[node] == 1]
    if len(endpoints) >= 2:
        best_pair = None
        best_length = -1.0
        for source in endpoints:
            lengths = nx.single_source_dijkstra_path_length(graph, source, weight="weight")
            for target in endpoints:
                length = lengths.get(target, -1.0)
                if length > best_length:
                    best_length = length
                    best_pair = (source, target)
        return nx.shortest_path(graph, best_pair[0], best_pair[1], weight="weight")

    first = next(iter(graph.nodes))
    far = _farthest_node(graph, first)
    other = _farthest_node(graph, far)
    return nx.shortest_path(graph, far, other, weight="weight")


def _simplify_path(points):
    if len(points) <= 2:
        return [(int(x), int(y)) for y, x in points]
    contour = np.array([[[x, y]] for y, x in points], dtype=np.int32)
    epsilon = max(1.0, 0.01 * cv2.arcLength(contour, False))
    simplified = cv2.approxPolyDP(contour, epsilon, False)
    return [(int(point[0][0]), int(point[0][1])) for point in simplified]


def mask_to_baseline(mask):
    binary_mask = mask > 0
    if not binary_mask.any():
        return []

    skeleton = skeletonize(binary_mask).astype(np.uint8)
    graph = _build_skeleton_graph(skeleton)
    path = _extract_longest_path(graph)
    if path:
        return _simplify_path(path)

    ys, xs = np.nonzero(binary_mask)
    if ys.size == 0:
        return []
    y = int(np.percentile(ys, 90))
    return [(int(xs.min()), y), (int(xs.max()), y)]