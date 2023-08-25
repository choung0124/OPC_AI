import streamlit as st
from transformers import logging
from langchain.llms import TextGen
from langchain.prompts import PromptTemplate
from langchain import LLMChain
import streamlit as st
import streamlit as st
from pyvis.network import Network
from CustomLibrary.Custom_Agent import CustomLLMChain, CustomLLMChainAdditionalEntities
from CustomLibrary.Custom_Prompts import (
    Entity_type_Template_add_Upstage,
    Entity_Extraction_Template_Upstage,
    Additional_Entity_Extraction_Template_Upstage,
    Entity_type_Template_Upstage,
    OPC_Entity_type_Template,
    OPC_Entity_Extraction_Template_Upstage,
    OPC_Entity_Extraction_Template_Guanaco,
    Entity_type_Template,
    Final_Answer_Template_Alpaca
)
from CustomLibrary.App_Utils import(
    get_umls_info, 
    extract_entities, 
    get_names_list, 
    get_names_list, 
    get_entity_types, 
    get_additional_entity_umls_dict,
    create_and_display_network
)
from itertools import combinations, product
from CustomLibrary.Graph_Visualize import parse_relationships_pyvis
from langchain.embeddings import HuggingFaceEmbeddings

# Could there be a synergistic interaction between sildenafil and ashwagandha to treat alzheimer's?
#Withaferin A, Withanolide A, Withanolide B, Withanolide C, Withanolide D, Withanone, Withanoside IV, Withanoside V

logging.set_verbosity(logging.CRITICAL)

from CustomLibrary.OPC_GraphQA import OPC_GraphQA

@st.cache_data()
def initialize_models():
    model_url = "https://provides-tomorrow-norman-amounts.trycloudflare.com/"
    local_model_url = "http://127.0.0.1:5000/"
    llm = TextGen(model_url=model_url, max_new_tokens=2048)
    local_llm = TextGen(model_url=local_model_url, max_new_tokens=2048)
    Entity_extraction_prompt = PromptTemplate(template=Entity_Extraction_Template_Upstage, input_variables=["input"])
    entity_extraction_chain = CustomLLMChain(prompt=Entity_extraction_prompt, llm=llm, output_key="output",)
    return llm, entity_extraction_chain, local_llm

@st.cache_data()
def initialize_knowledge_graph():
    uri = "neo4j://localhost:7687"
    username = "neo4j"
    password = "NeO4J"
    return uri, username, password

class SessionState(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

def get_state(**kwargs):
    if 'state' not in st.session_state:
        st.session_state['state'] = SessionState(**kwargs)
    return st.session_state['state']

# Define the progress bar
progress_bar = st.empty()

# Define the callback function to update the progress bar
def progress_callback(progress):
    progress_bar.progress(progress)

def initialize_all(state):
    if not hasattr(state, 'initialized'):
        state.llm, state.entity_extraction_chain, state.local_llm = initialize_models()
        state.uri, state.username, state.password = initialize_knowledge_graph()
        #OPC_Entity_Extraction_Prompt = PromptTemplate(template=OPC_Entity_Extraction_Template_Upstage, input_variables=["input"])
        OPC_Entity_Extraction_Prompt = PromptTemplate(template=OPC_Entity_Extraction_Template_Guanaco, input_variables=["input"])
        state.OPC_Entity_Extraction_Chain = CustomLLMChain(prompt=OPC_Entity_Extraction_Prompt, llm=state.llm, output_key="output")
        additional_entity_extraction_prompt = PromptTemplate(template=Additional_Entity_Extraction_Template_Upstage, input_variables=["input", "entities"])
        #Entity_type_prompt = PromptTemplate(template=Entity_type_Template_Upstage, input_variables=["input"])
        Entity_type_prompt = PromptTemplate(template=OPC_Entity_type_Template, input_variables=["input"])        
        Entity_type_prompt_add = PromptTemplate(template=Entity_type_Template_add_Upstage, input_variables=["input"])
        state.additional_entity_extraction_chain = CustomLLMChainAdditionalEntities(prompt=additional_entity_extraction_prompt, llm=state.llm, output_key="output",)
        state.Entity_type_chain = LLMChain(prompt=Entity_type_prompt, llm=state.llm)
        state.Entity_type_chain_add = LLMChain(prompt=Entity_type_prompt_add, llm=state.llm)
        state.initialized = True

# Get the state
state = get_state(user_options=[])

# Initialize all
initialize_all(state)

question = st.text_input("Enter your question")

# initialize your counter
if 'counter' not in st.session_state:
    st.session_state.counter = 0

# initialize the last processed question
if 'last_question' not in st.session_state:
    st.session_state.last_question = None

# initialize the entities_list
if 'entities_list' not in st.session_state:
    st.session_state.entities_list = []

# initialize the constituents_dict
if 'constituents_dict' not in st.session_state:
    st.session_state.constituents_dict = {}

if 'paths_list' not in st.session_state:
    st.session_state.paths_list = []

if 'names_list' not in st.session_state:
    st.session_state.names_list = []

if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False

# if a new question is entered, process it
if question and question != st.session_state.last_question:

    with st.spinner("Processing..."):
        # Entity extraction
        entities = state.OPC_Entity_Extraction_Chain.run(question)

        entities_umls_ids = get_umls_info(entities)

        names_list = get_names_list(entities_umls_ids)

        entity_types = get_entity_types(state.Entity_type_chain, names_list)

        print("names list", names_list)

        # get the list of entities
        st.session_state.entities_list = list(entity_types.items())

        # store the processed question
        st.session_state.last_question = question
        # reset the counter for the new question
        st.session_state.counter = 0
else:
    # if there's no new question, use the entities from the last question
    entities_list = st.session_state.entities_list

if st.session_state.counter < len(st.session_state.entities_list):
    entity, entity_type = st.session_state.entities_list[st.session_state.counter]
    if entity_type in ["Food", "Metabolite", "Drug"]:
        with st.form(key=str(entity)):
            st.write("Please input the chemical constituents of:", entity, "(Please separate each constituent with a comma)")
            entity_constituents = st.text_input(f"Input the chemical constituents of {entity}:", key=str(entity))
            none_checkbox = st.checkbox("None", key=str(entity).join("_NoneCheckbox"))  # Move none_checkbox inside the form
            submitted = st.form_submit_button('Submit')

        if submitted:
            if none_checkbox:
                st.session_state.counter += 1
            else:
                constituents = entity_constituents.split(",") if entity_constituents else []
                # Only add entity to constituents_dict if constituents is not empty and contains non-empty strings
                if constituents and any(constituent.strip() for constituent in constituents):
                    st.session_state.constituents_dict[entity] = constituents
                    for constituent in constituents:
                        path = {
                            'nodes': [entity, constituent],
                            'relationships': ['contains constituent']
                        }
                        st.session_state.paths_list.append(path)
                st.session_state.counter += 1
        print(st.session_state.constituents_dict)  # Debug print statement
    else:
        st.session_state.counter += 1

if st.session_state.counter == len(st.session_state.entities_list):
    st.write("All entities processed")
    nodes_list = []
    paths = st.session_state.paths_list
    for entity, entity_type in st.session_state.entities_list:
        if entity_type in ["Food", "Metabolite", "Drug"]:
            if entity in st.session_state.constituents_dict:
                constituents = st.session_state.constituents_dict[entity]
                nodes_list.extend([entity, *constituents])
            else:
                constituents = []
            with st.expander(f"Constituents of {entity}"):
                st.write(constituents)

    if st.button("Run OPC GraphQA"):
        with st.spinner("Running OPC GraphQA..."):
            # Assuming 'entities_list' is a list of all entities
            entities_list = st.session_state.entities_list

            # Dictionary to hold combinations for each entity
            entity_combinations = {}

            for entity, entity_type in entities_list:

                if entity in st.session_state.constituents_dict and st.session_state.constituents_dict [entity]:
                    # Get the constituents for the current entity
                    constituents = st.session_state.constituents_dict[entity]

                    if len(constituents) > 2:
                        # Generate all combinations of 4 constituents
                        combinations_of_constituents = list(combinations(constituents, 2))
                    else:
                        # If there are 4 or fewer constituents, use them directly
                        combinations_of_constituents = [constituents]

                    # Store the combinations in the dictionary
                    entity_combinations[entity] = combinations_of_constituents

            # Generate all combinations of combinations for each entity
            all_combinations = list(product(*entity_combinations.values()))

            answers = []
            counter = 0
            for combo in all_combinations:

                counter += 1
                # 'combo' is a tuple of combinations, one for each entity
                # Convert it to a dictionary
                constituents_dict = {}
                for i, entity in enumerate(entity_combinations.keys()):
                    constituents_dict[entity] = list(combo[i])  # The i-th combination for the i-th entity

                # Run OPC GraphQA for each combination
                KG = OPC_GraphQA(uri=state.uri, 
                                username=state.username, 
                                password=state.password,
                                llm=state.local_llm,
                                entities_list=list(entity_combinations.keys()),  # All entities
                                constituents_dict=constituents_dict,  # The current combination of constituents
                                constituents_paths=st.session_state.paths_list)

                graph_query = KG._call(question, progress_callback=progress_callback)
                if graph_query['result'] == None:
                    continue
                answer = graph_query['result']
                answers.append(answer)
                with st.expander(f"Combination {counter}"):
                    all_rels = graph_query['all_rels']
                    st.header("Answer:")
                    st.write(answer)

                    st.header("Graph:")
                    nodes, edges = parse_relationships_pyvis(all_rels)
                    create_and_display_network(nodes, edges, '#fff6fe', "Graph", entities_list[0], entities_list[1])

            hf = HuggingFaceEmbeddings(
            model_name='pritamdeka/S-Bluebert-snli-multinli-stsb',
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True})

            db = Chroma.from_texts(answers, hf)
            retriever = db.as_retriever(search_kwargs={"k": 2})
            docs = retriever.get_relevant_documents(question)[:2]
            final_result = [doc.page_content for doc in docs]
            selected_answers = final_result.join("\n")

            st.header("Final Answer:")
            Final_chain_prompt = PromptTemplate(template=Final_Answer_Template_Alpaca, input_variables=["input", "question"])
            Final_chain = LLMChain(llm=state.local_llm, prompt=Final_chain_prompt)
            final_answer = Final_chain.run(input=selected_answers, question=question)
            st.write(final_answer)