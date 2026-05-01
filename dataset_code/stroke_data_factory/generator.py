import random
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .geometry_builder import build_geometry
from .metadata_annotator import annotate_metadata
from .spec_sampler import sample_scene_spec
from .stroke_compiler import compile_strokes
from .text_realizer import make_prompt


def sample_scene(
    rng: random.Random,
    recipe_name: str = "balanced_v1",
    difficulty: Optional[str] = None,
) -> Dict[str, Any]:
    scene_spec = sample_scene_spec(rng, recipe_name=recipe_name, difficulty=difficulty)
    shapes = build_geometry(scene_spec, rng)
    strokes = compile_strokes(shapes)
    prompt = make_prompt(
        shapes,
        relation_type=scene_spec.get("relation_type"),
        anchor_position=scene_spec.get("anchor_position"),
        scene_spec=scene_spec,
    )
    metadata = annotate_metadata(scene_spec, shapes, strokes)

    return {
        "scene_spec": scene_spec,
        "prompt": prompt,
        "shapes": [
            {
                "shape_type": s.shape_type,
                "prompt_fragment": s.prompt_fragment,
                "points": [asdict(p) for p in s.points],
                "bbox": s.bbox,
                "closed": s.closed,
            }
            for s in shapes
        ],
        "strokes": [asdict(step) for step in strokes],
        "metadata": metadata,
    }


def generate_dataset(
    num_samples: int,
    seed: int = 0,
    recipe_name: str = "balanced_v1",
    difficulty: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    return [sample_scene(rng, recipe_name=recipe_name, difficulty=difficulty) for _ in range(num_samples)]
