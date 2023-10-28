from .characters import vietnamese_chars

character_sorting_score = {
    v: k for k, v in enumerate(vietnamese_chars)
}


class Trie:
    def __init__(self):
        self.root = {}
    
    def add(self, word):
        node = self.root
        for ch in word:
            if ch not in node:
                node[ch] = {}
            node = node[ch]
        node["<END>"] = None
    
    def exists(self, word):
        node = self.root
        for ch in word:
            if ch not in node:
                return False
            node = node[ch]
        if "<END>" not in node:
            return False
        return True

    def sort(self, key=None):
        if key is False:
            sort_key = lambda x: x[0] # sort by key
        elif key:
            sort_key = key
        else:
            sort_key = lambda x: character_sorting_score[x[0]]
        self.root = sort_dict(self.root, sort_key)

    def get_entities_by_prefix(self, prefix, key=None):
        if key is False:
            sort_key = lambda x: x[0] # sort by key
        elif key:
            sort_key = key
        else:
            sort_key = lambda x: character_sorting_score[x[0]]
        node = self.root
        for ch in prefix:
            if ch not in node:
                return []
            node = node[ch]
        L = self._get_entities(node, sort_key)
        L = [prefix + entity for entity in L]
        return L
    
    def get_entities(self, key=None):
        return self._get_entities(self.root, key)

    @staticmethod
    def _get_entities(obj: dict, key):
        L = []
        stack = [("", obj)]
        while stack:
            entity, node = stack.pop()
            children = sorted(node.items(), key=key, reverse=True)
            for k, v in children:
                if k == "<END>":
                    L.append(entity)
                else:
                    stack.append((entity + k, v))
        return L


def sort_dict(obj: dict, key):
    sorted_obj = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            v = sort_dict(v)
        sorted_obj[k] = v
    return dict(sorted(sorted_obj.items(), key=key))
