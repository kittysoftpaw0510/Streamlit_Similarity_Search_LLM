import random
from typing import List

TOPICS = [
    "solar energy","machine learning","urban gardening","digital privacy",
    "remote work culture","sustainable transport","smart homes","mindfulness",
    "fitness routines","space exploration","culinary arts","classical music",
    "wildlife conservation","personal finance","photography","language learning",
    "autonomous vehicles","virtual reality","cybersecurity","renewable resources",
    "climate change","ocean currents","blockchain","electric cars",
    "project management","time management","habit tracking","open source software",
    "user experience","game design"
]

TEMPLATES = [
    "This sentence explores {topic} with a focus on practical applications.",
    "A brief overview of {topic} highlights both benefits and limitations.",
    "We often ignore how {topic} influences everyday decisions.",
    "In-depth analysis reveals new trends across {topic}.",
    "Experts debate whether {topic} can scale efficiently.",
    "A beginner can start learning {topic} by doing small projects.",
    "There are ethical questions that arise when discussing {topic}.",
    "Recent studies suggest {topic} is becoming more accessible.",
    "Many people misunderstand the complexity of {topic}.",
    "It's worth noting that {topic} has historical roots.",
]

def _rng(user_id: str, which: str) -> random.Random:
    # Deterministic per user & file
    seed = hash(f"{user_id}::{which}::v1")
    return random.Random(seed)

def generate_sentences(user_id: str, which: str, n: int = 30) -> List[str]:
    r = _rng(user_id, which)
    out = []
    for _ in range(n):
        topic = r.choice(TOPICS)
        tmpl = r.choice(TEMPLATES)
        out.append(tmpl.format(topic=topic))
    return out

def simulate_file_content(user_id: str, which: str) -> str:
    return "\n".join(generate_sentences(user_id, which, 30))
