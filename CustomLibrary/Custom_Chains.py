from langchain.agents import Tool, AgentOutputParser, AgentType, initialize_agent
from langchain.schema import AgentAction, AgentFinish
from typing import List, Tuple, Any, Union, Callable, Type, Optional, Dict
from langchain.prompts import StringPromptTemplate
import re
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain.chains import LLMChain
import json
import ast

class CustomLLMChain(LLMChain):
    def run(self, query: str, *args, **kwargs) -> Dict[str, Any]:
        raw_output = super().run(query, *args, **kwargs)
        print(raw_output)
        return self.parse_output(raw_output)
    
    def parse_output(self, output: str) -> Dict[str, Any]:
        return parse_llm_output(output)

def parse_llm_output(output: str) -> list:
    entities_match = re.search(r"Entities: (\[[^\]]*\])", output)
    if entities_match:
        entities = entities_match.group(1)
        # Parse the entities with json.loads
        try:
            entities = json.loads(entities)
        except json.JSONDecodeError as e:
            try:
                entities = ast.literal_eval(entities)
            except: 
                print(f"Error parsing entities: {e}")
                return []

        return entities
    
class CustomLLMChainAdditionalEntities(LLMChain):
    def run(self, *args, query: str = "", **kwargs) -> Dict[str, Any]:
        raw_output = super().run(query=query, *args, **kwargs)
        print(raw_output)
        return self.parse_output_additional(raw_output)
    
    def parse_output_additional(self, output: str) -> Dict[str, Any]:
        return parse_llm_output_additional(output)

def parse_llm_output_additional(output: str) -> list:
    entities_match = re.search(r'Additional Entities: (\[[^\]]*\])', output)
    if entities_match:
        entities = entities_match.group(1)
        print(f"entities: {entities}")  # Debugging line
        try:
            entities = json.loads(entities)
            return entities
        except json.JSONDecodeError as e:
            print(f"Error parsing entities: {e}")
            return []
    else:
        return []
