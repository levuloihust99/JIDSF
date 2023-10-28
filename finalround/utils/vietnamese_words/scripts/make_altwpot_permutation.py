"""List all at-least-two-wowels-plus-one-tone (altwpot) permutations, with the correct form comes first.

E.g.
ứa -> ứa,ưá
oạo -> oạo,ọao,oaọ
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

vi_vowel_path = sound_path.parent / "vi_vowels.txt"
vi_vowels = []
with vi_vowel_path.open(mode="r") as reader:
    for v in reader:
        v = v.strip()
        if v:
            vi_vowels.append(v)

all_permutations = []
for sound in altwpot_sounds:
    NFKD_sound = unicodedata.normalize("NFKD", sound)
    tone = []
    NFKD_sound_no_tone = ""
    for ch in NFKD_sound:
        if ord(ch) not in tones:
            NFKD_sound_no_tone += ch
        else:
            tone.append(ch)
    assert len(tone) == 1
    tone = tone[0]
    NFKC_sound_no_tone = unicodedata.normalize("NFKC", NFKD_sound_no_tone)
    permutations = []
    for i, ch in enumerate(NFKC_sound_no_tone):
        if ch in vi_vowels:
            permutations.append(NFKC_sound_no_tone[:i + 1] + tone + NFKC_sound_no_tone[i + 1:])
    permutations = [unicodedata.normalize("NFKC", p) for p in permutations]
    permutations = [sound] + [p for p in permutations if p != sound]
    all_permutations.append(",".join(permutations))

permutation_path = sound_path.parent / "altwpot_sound_permutations.txt"
with permutation_path.open(mode="w") as writer:
    writer.write("\n".join(all_permutations) + "\n")
