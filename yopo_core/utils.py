import xml.etree.ElementTree as ET
import functools

def parse_root(element):
    element_node = element.attrib
    element_node['node_type'] = element.tag

    element_nodes = [element_node]
    element_edges = []

    if "predecessors" in element_node:
        element_edges = [(elt, element_node['idx']) for elt in element_node['predecessors'].split(',')]

    if len(element) == 0:
        return element_nodes, element_edges

    for child in element:
        element_edges.append((child.attrib['idx'], element_node['idx']))
        child_nodes, child_edges = parse_root(child)
        element_nodes.extend(child_nodes)
        element_edges.extend(child_edges)
    return element_nodes, element_edges

def msg2mira(message: dict):
    from mirascope.core import Messages
    if message['role'] == 'system':
        return Messages.System(message['content'])
    elif message['role'] == 'assistant':
        return Messages.Assistant(message['content'])
    else:
        return Messages.User(message['content'])

def stream_call(call):
    print(f"\n\n**************\n{call.fn_args['header']}\n**************")
    for chunk, _ in call:
        print(chunk.content, end="")
    return call.construct_call_response().content

def notebook_streamer(func):
    @functools.wraps(func)
    def stream_wrapper(*args, **kwargs):
        call = func(*args, **kwargs)
        return stream_call(call)
    return stream_wrapper