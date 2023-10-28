import json
from pathlib import Path

from ..trie import Trie

sound_path = Path(__file__).parent.parent / "assets/sounds.json"
with sound_path.open(mode="r") as reader:
    sound_data = json.load(reader)

trie = Trie()
trie.root = sound_data
sounds = trie.get_entities()

out_sound_path = sound_path.parent / "sounds.txt"
with out_sound_path.open(mode="w") as writer:
    writer.write("\n".join(sounds) + "\n")