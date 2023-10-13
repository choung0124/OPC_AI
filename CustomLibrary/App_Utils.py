import re
import ast
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
import webbrowser
import os
import colour
import json
from Bio import Entrez
import time
import requests

def get_umls_id(search_string: str) -> list:
    api_key = "7cc294c9-98ed-486b-add8-a60bd53de1c6"
    base_url = "https://uts-ws.nlm.nih.gov/rest/search/current"
    query = f"?string={search_string}&inputType=atom&returnIdType=concept&apiKey={api_key}"
    url = f"{base_url}{query}"
    
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        results = data["result"]["results"]
        if results:
            filtered_results = [result for result in results if search_string.lower() in result['name'].lower()]
            if filtered_results:
                top_result = filtered_results[0]
                result_string = f"Name: {top_result['name']} UMLS_CUI: {top_result['ui']}"
                return [result_string]
            else:
                return ["No results found."]
        else:
            return ["No results found."]
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

def extract_entities(question, entity_extraction_chain, additional_entity_extraction_chain):
    result = entity_extraction_chain.run(question)
    entities = result

    additional_entities = additional_entity_extraction_chain.run(input=question, entities=entities)

    return entities, additional_entities


def get_umls_info(entities):
    entities_umls_ids = {}
    for entity in entities:
        umls_id = get_umls_id(entity)
        entities_umls_ids[entity] = umls_id

    return entities_umls_ids


def get_names_list(entities_umls_ids):
    names_list = []
    for entity, umls_info_list in entities_umls_ids.items():
        if umls_info_list:
            umls_info = umls_info_list[0]
            match = re.search(r"Name: (.*?) UMLS_CUI: (\w+)", umls_info)
            if match:
                umls_name = match.group(1)
                umls_cui = match.group(2)
                names_list.append(umls_name)
            else:
                names_list.append(entity)
        else:
            names_list.append(entity)

    return names_list


def get_entity_types(Entity_type_chain, names_list):
    Entity_type_chain_result = Entity_type_chain.run(names_list)

    start = Entity_type_chain_result.index("[")
    end = Entity_type_chain_result.index("]") + 1
    list_str = Entity_type_chain_result[start:end]
    extracted_types = ast.literal_eval(list_str)

    entity_types = {entity_info[0]: entity_info[1] for entity_info in extracted_types}

    return entity_types


def get_additional_entity_umls_dict(additional_entities, Entity_type_chain_add):
    entities_umls_ids = get_umls_info(additional_entities)

    additional_entity_umls_dict = {}

    for entity, umls_info_list in entities_umls_ids.items():
        if umls_info_list:
            umls_info = umls_info_list[0]
            match = re.search(r"Name: (.*?) UMLS_CUI: (\w+)", umls_info)
            if match:
                umls_name = match.group(1)
                umls_cui = match.group(2)
                additional_entity_umls_dict[entity] = umls_cui
            else:
                additional_entity_umls_dict[entity] = None
        else:
            additional_entity_umls_dict[entity] = None

    for entity, umls_cui in additional_entity_umls_dict.items():
        if umls_cui:
            entity_type_result = Entity_type_chain_add.run(entity)
            start = entity_type_result.index("[")
            end = entity_type_result.index("]") + 1
            list_str = entity_type_result[start:end]
            extracted_types = ast.literal_eval(list_str)
            entity_type = extracted_types[0] if extracted_types else None
            additional_entity_umls_dict[entity] = {"umls_cui": umls_cui, "entity_type": entity_type}
        else:
            additional_entity_umls_dict[entity] = {"umls_cui": None, "entity_type": None}

    return additional_entity_umls_dict

def parse_relationships_pyvis(relationships):
    nodes = set()
    edges = []
    for relationship in relationships:
        elements = relationship.split(' -> ')
        for i in range(0, len(elements) - 2, 2):
            source = elements[i]
            relationship_type = elements[i + 1]
            target = elements[i + 2]
            nodes.add(source)
            nodes.add(target)
            edges.append((source, target, relationship_type))
    return list(nodes), edges

def create_and_display_network(nodes, edges, back_color, name, source, target):
    back_color = colour.Color(back_color)
    bg_color = back_color.get_hex()

    # darken the color by reducing the luminance
    # darken the color by reducing the luminance
    back_color.luminance *= 0.8  # reduce luminance by 20%
    border_color = back_color.get_hex()

    # darken the color even more for the nodes
    back_color.luminance *= 0.5  # reduce luminance by additional 50%
    node_color = back_color.get_hex()

    source_node_color = "#ff0000"
    target_node_color = "#00ff00"

    # Initialize Network with the hexadecimal color string
    net = Network(height='750px', width='100%', bgcolor=bg_color, font_color='black', directed=True)

    for node in nodes:
        if node == source:
            node_color = source_node_color
        elif node == target:
            node_color = target_node_color
        else:
            node_color = back_color.get_hex()
            
        net.add_node(node, label=node, title=node, color=node_color, url="http://example.com/{}".format(node))

    # add edges
    for edge in edges:
        net.add_edge(edge[0], edge[1], title=edge[2])
    net.toggle_physics(True)

    # save to HTML file
    net.save_graph(f'{name}network.html')

    # Create a border around the network using custom CSS
    st.markdown(
        f"""
        <style>
        .network {{
            border: 4px solid {border_color};
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    net.save_graph(f'{name}network.html')

    # Display in Streamlit within a div with the class "network"
    with st.spinner("Rendering network..."):
        html_string = open(f'{name}network.html', 'r').read()
        components.html(
            f"""
            <div class="network" style="display: flex; justify-content: center;">
                {html_string}
            </div>
            """, 
            width=630, 
            height=630
        )

    # Add a button to open the network in full size in a new tab
    st.markdown(
        f'<a href="file://{os.path.realpath(f"{name}network.html")}" target="_blank">Open Network in Full Size</a>', 
        unsafe_allow_html=True
    )

def search(query):
    Entrez.email = 'hschoung0124@gmail.com'  # Always tell NCBI who you are
    handle = Entrez.esearch(db='pubmed', 
                            sort='relevance',  
                            retmax='1', 
                            retmode='xml', 
                            term=query)
    results = Entrez.read(handle)
    return results

from http.client import IncompleteRead

def fetch_details(id_list):
    ids = ','.join(id_list)
    Entrez.email = 'hschoung0124@gmail.com'
    try:
        handle = Entrez.efetch(db='pubmed',
                               retmode='xml',
                               id=ids)
        results = Entrez.read(handle)
    except IncompleteRead as e:
        print("Incomplete read error, retrying...")
        return fetch_details(id_list)
    return results

def search_pubmed(query):
    results = search(query)
    id_list = results['IdList']
    papers = fetch_details(id_list)
    for i, paper in enumerate(papers['PubmedArticle']):
        return "https://pubmed.ncbi.nlm.nih.gov/{}".format(id_list[0])

def create_docs_from_results(results):
    # Flatten the lists
    documents = [doc for sublist in results["documents"] for doc in sublist]
    metadatas = [meta for sublist in results["metadatas"] for meta in sublist]

    # Create the list of dictionaries
    docs = []
    for document, metadata in zip(documents, metadatas):
        doc = {"document": document, "metadata": metadata}
        docs.append(doc)

    return docs