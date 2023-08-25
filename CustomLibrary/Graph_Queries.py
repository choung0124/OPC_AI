from __future__ import annotations
from typing import Any, Dict, List, Optional
from py2neo import Graph
from typing import Tuple, Set

def get_node_label(graph: Graph, node_name: str) -> str:
    query = """
    MATCH (node)
    WHERE toLower(node.name) = toLower($nodeName)
    RETURN head(labels(node)) AS FirstLabel, node.name AS NodeName
    LIMIT 1
    """
    result = graph.run(query, nodeName=node_name).data()
    if result:
        print(query)
        return result[0]['FirstLabel'], result[0]['NodeName']
    else:
        return None, None

def get_node_labels_dict(graph: Graph, node_names: List[str]) -> Dict[str, str]:
    query = """
    UNWIND $nodeNames AS nodeName
    MATCH (node)
    WHERE toLower(node.name) = toLower(nodeName)
    RETURN node.name AS NodeName, head(labels(node)) AS FirstLabel
    """
    results = graph.run(query, nodeNames=node_names).data()
    print(query)
    return {result['NodeName']: result['FirstLabel'] for result in results}

def get_node_labels(graph: Graph, node_names: List[str]) -> Dict[str, str]:
    query = """
    UNWIND $node_names AS node_name
    MATCH (node)
    WHERE toLower(node.name) = toLower(node_name)
    RETURN node.name AS NodeName, head(labels(node)) AS FirstLabel
    """
    results = graph.run(query, node_names=node_names).data()
    return {result['NodeName']: result['FirstLabel'] for result in results}

def construct_path_string(nodes: List[str], relationships: List[str]) -> str:
    path_elements = []
    for node, relationship in zip(nodes, relationships):
        if node is None or relationship is None:
            continue  # Skip this element if the node or the relationship is None
        path_elements.append(f"{node} -> {relationship}")
    if nodes[-1] is not None:
        path_elements.append(nodes[-1])  # add the last node
    return " -> ".join(path_elements)

def construct_relationship_string(nodes: List[str], relationships: List[str]) -> str:
    path_elements = []
    for i in range(len(nodes) - 1):
        if nodes[i] is None or relationships[i] is None or nodes[i + 1] is None:
            continue  # Skip this element if any of the nodes or the relationship is None
        path_elements.append(f"{nodes[i]} -> {relationships[i]} -> {nodes[i + 1]}")
    return ", ".join(path_elements)

def query_direct(graph: Graph, node:str, node_label:Optional[str]=None) -> List[Dict[str, Any]]:
    if not node_label:
        node_label, node_name = get_node_label(graph, node)
    else:
        node_name = node

    paths_list = []
    query = f"""
    MATCH path=(source:{node_label})-[rel*1..1]->(node)
    WHERE source.name = "{node_name}" AND node IS NOT NULL
    WITH DISTINCT path, relationships(path) AS rels, nodes(path) AS nodes
    WHERE NONE(n IN nodes WHERE n IS NULL)
    RETURN [node IN nodes | node.name] AS path_nodes, [rel IN rels | type(rel)] AS path_relationships
    UNION
    MATCH path=(node)-[rel*1..1]->(source:{node_label})
    WHERE source.name = "{node_name}" AND node IS NOT NULL
    WITH DISTINCT path, relationships(path) AS rels, nodes(path) AS nodes
    WHERE NONE(n IN nodes WHERE n IS NULL)
    RETURN [node IN nodes | node.name] AS path_nodes, [rel IN rels | type(rel)] AS path_relationships
    LIMIT 50000
    """
    result = graph.run(query)
    print(query)

    for record in result:
        path_nodes = record['path_nodes']
        path_relationships = record['path_relationships']
        paths_list.append({'nodes': path_nodes, 'relationships': path_relationships})

    return paths_list

def query_direct_constituents(graph: Graph, node:str, entity:str,  node_label:Optional[str]=None) -> List[Dict[str, Any]]:
    if not node_label:
        node_label, node_name = get_node_label(graph, node)
    else:
        node_name = node

    paths_list = []
    query = f"""
    MATCH path=(source:{node_label})-[rel*1..1]->(node)
    WHERE source.name = "{node_name}" AND node IS NOT NULL
    WITH DISTINCT path, relationships(path) AS rels, nodes(path) AS nodes
    WHERE NONE(n IN nodes WHERE n IS NULL)
    RETURN [node IN nodes | node.name] AS path_nodes, [rel IN rels | type(rel)] AS path_relationships
    UNION
    MATCH path=(node)-[rel*1..1]->(source:{node_label})
    WHERE source.name = "{node_name}" AND node IS NOT NULL
    WITH DISTINCT path, relationships(path) AS rels, nodes(path) AS nodes
    WHERE NONE(n IN nodes WHERE n IS NULL)
    RETURN [node IN nodes | node.name] AS path_nodes, [rel IN rels | type(rel)] AS path_relationships
    LIMIT 50000
    """
    result = graph.run(query)
    print(query)

    for record in result:
        path_nodes = []
        path_relationships = []
        path_nodes.append(entity)
        path_relationships.append("contains constituent")
        path_nodes.extend(record['path_nodes'])
        path_relationships.extend(record['path_relationships'])
        paths_list.append({'nodes': path_nodes, 'relationships': path_relationships})

    return paths_list


def query_between_direct(graph: Graph, direct_nodes, nodes:List[str]) -> str:
    all_node_names = list(nodes) + direct_nodes
    node_labels = get_node_labels(graph, all_node_names)
    unique_labels = list(set(node_labels.values()))
    
    query_parameters_2 = {"nodes": list(nodes) + direct_nodes, "unique_labels": unique_labels}
    total_nodes = list(nodes) + direct_nodes
    print("number of direct nodes")
    print(len(total_nodes))
    # Query for paths between the nodes from the original list

    inter_between_direct_query = """
    MATCH (n)
    WHERE n.name IN $nodes AND any(label in labels(n) WHERE label IN $unique_labels)
    CALL apoc.path.spanningTree(n, {minLevel: 1, limit: 200}) YIELD path
    WITH nodes(path) AS nodes, relationships(path) AS rels
    RETURN [node IN nodes | node.name] AS path_nodes, [rel IN rels | type(rel)] AS path_relationships
    """
    result_inter_direct = list(graph.run(inter_between_direct_query, **query_parameters_2))
    print(inter_between_direct_query)

    paths = [{'nodes': record['path_nodes'], 'relationships': record['path_relationships']} for record in result_inter_direct]
    print("number of inter direct relations:")
    print(len(paths))

    return paths
