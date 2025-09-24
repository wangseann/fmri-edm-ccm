from pathlib import Path
from typing import Dict, List


def list_subjects(data_root: str) -> List[str]:
    base = Path(data_root)
    return sorted([d.name for d in base.glob("sub-*") if d.is_dir()])


def list_stories_for_subject(data_root: str, subject: str) -> List[Dict]:
    subject_dir = Path(data_root) / subject
    wavs = list(subject_dir.rglob("*.wav"))
    textgrids = list(subject_dir.rglob("*.TextGrid"))
    stories: List[Dict] = []
    for wav_path in wavs:
        stem = wav_path.stem.lower()
        match = next((tg for tg in textgrids if tg.stem.lower() == stem), None)
        stories.append(
            {
                "wav": str(wav_path),
                "textgrid": str(match) if match else None,
                "stem": wav_path.stem,
            }
        )
    return stories
