"""List at-least-two-wowels-plus-one-tone (altwpot) sounds.

E.g.
ưa -> 2 vowel, no tone -> False
ứa -> 2 vowels, having tone -> True
uỷu -> 3 vowels, having tone -> True
"""

import json
import unicodedata
from pathlib import Path

from ..trie import Trie

sound_path = Path(__file__).parent.parent / "assets/sounds.json"
with sound_path.open(mode="r") as reader:
    sound_data = json.load(reader)

trie = Trie()
trie.root = sound_data
sounds = trie.get_entities()

altwpot_sounds = []
vowels = ["u", "e", "o", "a", "i", "y"] # nguyên âm
tones = [769, 768, 777, 771, 803] # sắc, huyền, hỏi, ngã, nặng

for sound in sounds:
    NFD_sound = unicodedata.normalize("NFKD", sound)
    num_vowels = 0
    has_tone = False
    for ch in NFD_sound:
        if ch in vowels:
            num_vowels += 1
        if ord(ch) in tones:
            has_tone = True
    if has_tone is False:
        continue
    if num_vowels <= 1:
        continue
    altwpot_sounds.append(sound)

out_path = sound_path.parent / "altwpot_sounds.txt"
with out_path.open(mode="w") as writer:
    writer.write("\n".join(altwpot_sounds) + "\n")