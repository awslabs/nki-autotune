import networkx as nx


def analyze_data_reuse_opportunities(graph: nx.DiGraph) -> list[list[int]]:
    """
    Args:
        graph: Compute graph with load nodes to analyze

    Returns:
        List of merge groups, where each group contains node IDs that load identical data
    """
    load_groups: dict[tuple, list[int]] = {}

    for node_id in graph.nodes():
        node_data = graph.nodes[node_id]
        if node_data.get("type") != "load":
            continue

        tensor_name = node_data.get("tensor_name")
        tile_indices = node_data.get("tile_indices", {})

        tile_key = tuple(sorted(tile_indices.items()))
        group_key = (tensor_name, tile_key)

        if group_key not in load_groups:
            load_groups[group_key] = []
        load_groups[group_key].append(node_id)

    merge_opportunities = [group for group in load_groups.values() if len(group) > 1]

    return merge_opportunities


def apply_data_reuse_merge(graph: nx.DiGraph, merge_group: list[int]) -> nx.DiGraph:
    """
    Args:
        graph: Compute graph to transform
        merge_group: List of load node IDs to merge into a single shared load

    Returns:
        Modified graph with redundant loads removed and edges redirected
    """
    if len(merge_group) < 2:
        return graph

    representative_node = merge_group[0]
    redundant_nodes = merge_group[1:]

    for redundant_node in redundant_nodes:
        successors = list(graph.successors(redundant_node))

        for successor in successors:
            if not graph.has_edge(representative_node, successor):
                graph.add_edge(representative_node, successor)

        graph.remove_node(redundant_node)

    return graph


def apply_all_data_reuse(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Args:
        graph: Compute graph to transform

    Returns:
        Transformed graph with all data reuse opportunities applied
    """
    merge_opportunities = analyze_data_reuse_opportunities(graph)

    for merge_group in merge_opportunities:
        graph = apply_data_reuse_merge(graph, merge_group)

    return graph
