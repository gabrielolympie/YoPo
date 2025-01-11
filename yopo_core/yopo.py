import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

from mirascope.core import litellm, Messages
import xml.etree.ElementTree as ET
from treelib import Node, Tree
import random

from yopo_core.config import API_KEY, API_BASE, MODEL_ID
from yopo_core.utils import parse_root, msg2mira, notebook_streamer

@litellm.call(model=MODEL_ID, stream=True, call_params={'temperature': 0.3, 'top_p': 0.95}, tools=[])
def run_chat(messages: list, header: str = "run_conversation", **kwargs) -> str:
    return [msg2mira(msg) for msg in messages]

class YOPO(nx.DiGraph):
    """Sub class of network x, that enables parallel DAG execution"""
    def __init__(self, n_jobs=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_jobs = n_jobs
        
    def __repr__(self):
        return "YOPO()\n" + self.to_treelib()
        
    def __str__(self):
        return self.to_treelib()
    
    def __call__(self, messages):
        for token in self.run(messages):
            yield token

    def to_treelib(self):
        tree = Tree()
        terminal_node = next(node for node in self.nodes if self.out_degree(node) == 0)
        tree.create_node(tag=self.nodes[terminal_node]['description'], identifier="root_node")
        used_nodes = set()

        for edge in self.edges:
            source_node, target_node = edge
            if target_node == terminal_node:
                target_node = "root_node"

            tag = self.nodes[source_node]['description']

            if source_node in used_nodes:
                source_node += str(random.randint(10000, 100000))
            tree.create_node(tag=tag, identifier=source_node, parent=target_node)
            used_nodes.add(source_node)
        return f"```yopo\n{tree.show(stdout=False, reverse=False)}```\n"

    def from_xml_file(self, path):
        tree = ET.parse(path)
        root = tree.getroot()
        return self.from_xml_root(root)

    def from_xml_prompt(self, xml_prompt):
        root = ET.fromstring(xml_prompt)
        return self.from_xml_root(root)

    def from_xml_root(self, root):
        nodes, edges = parse_root(root)

        for node in nodes:
            self.add_new_node(node_idx=node['idx'], description=node['description'], inputs={'prompt': node['content']})

        for edge in edges:
            self.add_new_edge(*edge)
        return self

    def add_new_node(self, node_idx, description, inputs):
        self.add_node(node_idx, description=description, inputs=inputs)

    def add_new_edge(self, src_idx, tar_idx):
        new_self = self.copy()
        new_self.add_edge(src_idx, tar_idx)
        assert nx.is_directed_acyclic_graph(new_self), f"Impossible to add edge ({src_idx},{tar_idx}) as the Graph won't be a DAG"
        self.add_edge(src_idx, tar_idx)

    def pos(self, level):
        s = len(level)
        return np.arange(s) - (s - 1) / 2

    def hierarchy(self):
        return list(nx.topological_generations(self))

    def depth(self):
        return len(self.hierarchy())

    def layout(self):
        layout = {}
        for i, level in enumerate(nx.topological_generations(self)):
            for idx, j in zip(level, self.pos(level)):
                layout[idx] = [i, j - 0.7 * i]
        return layout

    def show(self, figsize=(16, 6)):
        pos = self.layout()
        labels = {node: self.nodes[node]['description'] for node in self.nodes}
        plt.figure(figsize=figsize)
        nx.draw(self, pos=pos, labels={}, node_size=2000, font_color="black")
        for node, (x, y) in pos.items():
            plt.text(x, y - 0.25, labels[node], horizontalalignment='center', verticalalignment='top')

    def runtime(self):
        self.graph.visualize(include_state=True, include_conditions=True)

    def build_input(self, node, state, messages):
        node_context = messages.copy()
        inputs = state[f"{node}_inputs"]
        predecessors = state[f"{node}_predecessors"]
        predecessors_outputs = [state[f"{pred}_output"] for pred in predecessors]

        prompt = inputs['prompt']
        for pred_output in predecessors_outputs:
            node_context.append({"role": "user", "content": pred_output['prompt']})
            node_context.append({"role": "assistant", "content": pred_output['reply']})
        node_context.append({"role": "user", "content": prompt})

        heading = f"\n```prompt\n{prompt}\n```\n"
        if predecessors:
            heading += f"```context\n{', '.join(predecessors)}\n```\n"

        return prompt, node_context, heading

    def run(self, messages):
        state = {}
        for node in self.nodes:
            state[f"{node}_inputs"] = self.nodes[node]['inputs']
            state[f"{node}_predecessors"] = list(self.predecessors(node))
            state[f"{node}_output"] = {}


        yield self.to_treelib()
        for step, step_nodes in enumerate(self.hierarchy()):
            yield f"\n\n************** Stage {step} **************\n"
            if len(step_nodes) == 1:
                step_node = step_nodes[0]

                prompt, node_context, heading = self.build_input(step_node, state, messages)
                yield heading

                token_stream = run_chat(node_context)
                for token, _ in token_stream:
                    yield token.content
                reply = token_stream.construct_call_response().content

                state[f'{step_node}_output'] = {'prompt': prompt, 'reply': reply}

            else:
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    future_to_node = {
                        executor.submit(self._run_node, step_node, state, messages): step_node
                        for step_node in step_nodes
                    }

                    for future in as_completed(future_to_node):
                        step_node = future_to_node[future]
                        try:
                            heading, reply = future.result()
                            yield f"{heading}\n{reply}"
                            state[f'{step_node}_output'] = {'prompt': heading.strip().split('\n')[1], 'reply': reply}
                        except Exception as e:
                            print(f"Error running node {step_node}: {e}")
            yield "```"

    def _run_node(self, step_node, state, messages):
        prompt, node_context, heading = self.build_input(step_node, state, messages)
        token_stream = run_chat(node_context)
        for token, _ in token_stream:
            pass
        reply = token_stream.construct_call_response().content

        return heading, reply