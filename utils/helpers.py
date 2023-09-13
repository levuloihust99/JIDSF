def recursive_apply(data, fn, ignore_keys=None):
    """Hàm áp dụng hồi quy function {fn} vào data

    Args:
        data (Dict/List): _description_
        fn (function): _description_
        ignore_keys (_type_, optional): Key của Dict không áp dụng function. Defaults to None.

    Returns:
        data: _description_
    """    
    stack = [(None, -1, data)]  # parent, idx, child: parent[idx] = child
    while stack:
        parent_node, index, node = stack.pop()
        if isinstance(node, list):
            stack.extend(list(zip([node] * len(node), range(len(node)), node)))
        elif isinstance(node, dict):
            stack.extend(
                list(zip([node] * len(node), node.keys(), node.values())))
        elif isinstance(node, str):
            if node and (ignore_keys is None or index not in ignore_keys):
                parent_node[index] = fn(node)
            else:
                parent_node[index] = node
        else:
            continue
    return data