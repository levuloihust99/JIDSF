import functools
from typing import List, Text, Optional

from .trie import Trie
from .characters import vietnamese_chars

character_sorting_score = {
    v: k for k, v in enumerate(vietnamese_chars)
}


def comparator(a: Text, b: Text):
    for ch_a, ch_b in zip(a, b):
        if character_sorting_score[ch_a] > character_sorting_score[ch_b]:
            return 1
        elif character_sorting_score[ch_a] < character_sorting_score[ch_b]:
            return -1
    if len(a) > len(b):
        return 1
    elif len(a) < len(b):
        return -1
    return 0


class RuleType:
    EXPAND       = "EXPAND"
    MULTI_EXPAND = "MULTI_EXPAND"
    PREFIX       = "PREFIX"
    TRUNCATE     = "TRUNCATE"
    UNION        = "UNION"
    NOT          = "NOT"
    EMPTY        = "EMPTY"
    EXACT        = "EXACT"


class Rule:
    def __init__(self, trie: Trie, parent: Optional["Rule"] = None):
        self.trie = trie
        self.parent = parent
    
    def execute(self):
        raise NotImplementedError
    
    def get_prefix(self):
        prefix = ""
        if self.parent:
            prefix = self.parent.get_prefix()
        if hasattr(self, "prefix"):
            return prefix + getattr(self, "prefix")
        return prefix

    @classmethod
    def parse(cls, data, trie: Optional[Trie] = None) -> "Rule":
        if data["type"] == RuleType.NOT:
            child_rule = Rule.parse(data["rule"], trie)
            rule = NotRule(trie, child_rule)
            child_rule.parent = rule
            return rule
        if data["type"] == RuleType.PREFIX:
            child_rule = Rule.parse(data["rule"], trie)
            rule = PrefixRule(trie, prefix=data["prefix"], rule=child_rule)
            child_rule.parent = rule
            return rule
        if data["type"] == RuleType.EXPAND:
            return ExpandRule(trie)
        if data["type"] == RuleType.MULTI_EXPAND:
            return MultiExpandRule(trie, prefixes=data["prefixes"])
        if data["type"] == RuleType.TRUNCATE:
            child_rule = Rule.parse(data["rule"], trie)
            rule = TruncateRule(trie, data["pretrunc"], child_rule)
            child_rule.parent = rule
            return rule
        if data["type"] == RuleType.EMPTY:
            return EmptyRule(trie)
        if data["type"] == RuleType.UNION:
            child_rules = [Rule.parse(rule_data, trie) for rule_data in data["rules"]]
            rule = UnionRule(trie, child_rules)
            for r in child_rules:
                r.parent = rule
            return rule
        if data["type"] == RuleType.EXACT:
            return ExactRule(trie, data["entities"])


class UnionRule(Rule):
    type = RuleType.UNION

    def __init__(self, trie, rules: List[Rule], parent: Optional[Rule] = None):
        super(UnionRule, self).__init__(trie, parent)
        self.rules = rules
    
    def execute(self):
        all_entities = []
        tracker = set()
        for rule in self.rules:
            entities = rule.execute()
            for e in entities:
                if e not in tracker:
                    tracker.add(e)
                    all_entities.append(e)
        all_entities = sorted(all_entities, key=functools.cmp_to_key(comparator))
        return all_entities


class NotRule(Rule):
    type = RuleType.NOT

    def __init__(self, trie: Trie, rule: Rule, parent: Optional[Rule] = None):
        super(NotRule, self).__init__(trie, parent)
        self.rule = rule
    
    def execute(self):
        prefix = self.get_prefix()
        all_entities = self.trie.get_entities_by_prefix(prefix)
        excluded_entities = self.rule.execute()
        excluded_entities = set(excluded_entities)
        for i, e in enumerate(all_entities):
            if e in excluded_entities:
                all_entities[i] = None
        all_entities = [e for e in all_entities if e is not None]
        return all_entities


class ExpandRule(Rule):
    type = RuleType.EXPAND

    def execute(self):
        prefix = self.get_prefix()
        entities = self.trie.get_entities_by_prefix(prefix)
        entities = [e[len(prefix):] for e in entities]
        return entities


class PrefixRule(Rule):
    type = RuleType.PREFIX

    def __init__(self, trie: Trie, prefix: Text, rule: Rule, parent: Optional[Rule] = None):
        super(PrefixRule, self).__init__(trie, parent)
        self.prefix = prefix
        self.rule = rule
    
    def execute(self):
        entities = self.rule.execute()
        entities = [self.prefix + e for e in entities]
        return entities


class MultiExpandRule(Rule):
    type = RuleType.MULTI_EXPAND

    def __init__(self, trie: Trie, prefixes: List[Text], parent: Optional[Rule] = None):
        super(MultiExpandRule, self).__init__(trie, parent)
        self.prefixes = prefixes
    
    def execute(self):
        parent_prefix = self.get_prefix()
        all_entities = []
        tracker = set()
        for prefix in self.prefixes:
            entities = self.trie.get_entities_by_prefix(parent_prefix + prefix)
            for e in entities:
                if e not in tracker:
                    tracker.add(e)
                    all_entities.append(e)
        all_entities = [e[len(parent_prefix):] for e in all_entities]
        all_entities = sorted(all_entities, key=functools.cmp_to_key(comparator))
        return all_entities


class TruncateRule(Rule):
    type = RuleType.TRUNCATE

    def __init__(self, trie: Trie, pretrunc: Text, rule: Rule, parent: Optional[Rule] = None):
        super(TruncateRule, self).__init__(trie, parent)
        self.pretrunc = pretrunc
        self.rule = rule
    
    def execute(self):
        entities = self.rule.execute()
        entities = [e[len(self.pretrunc):] for e in entities]
        return entities


class EmptyRule(Rule):
    type = RuleType.EMPTY

    def execute(self):
        return []


class ExactRule(Rule):
    type = RuleType.EXACT

    def __init__(self, trie: Trie, entities: List[Text], parent: Optional[Rule] = None):
        super(ExactRule, self).__init__(trie, parent)
        self.entities = entities

    def execute(self):
        return self.entities
