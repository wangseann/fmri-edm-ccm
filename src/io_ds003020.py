from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Union


def _require_directory(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Expected directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Expected directory but found file: {path}")
    return path


def list_subjects(data_root: Union[str, Path]) -> List[str]:
    base = _require_directory(Path(data_root))
    return sorted(d.name for d in base.glob("sub-*") if d.is_dir())


def _story_id_from_path(path: Path) -> Optional[str]:
    match = re.search(r"task-([a-zA-Z0-9]+)", path.name)
    if not match:
        return None
    return match.group(1).lower()


def _iter_story_runs(subject_dir: Path) -> Iterable[Path]:
    # Functional runs are stored under ses-*/func/ as BOLD NIfTI files.
    return subject_dir.glob("ses-*/func/*_bold.nii.gz")


def list_stories_for_subject(data_root: Union[str, Path], subject: str) -> List[Dict]:
    root = _require_directory(Path(data_root))
    subject_dir = _require_directory(root / subject)
    stimuli_dir = root / "stimuli"
    textgrid_dir = root / "derivative" / "TextGrids"

    stories: List[Dict] = []
    for run_path in _iter_story_runs(subject_dir):
        story_id = _story_id_from_path(run_path)
        if story_id is None:
            continue

        wav_path = stimuli_dir / f"{story_id}.wav"
        if not wav_path.exists():
            # Skip localizer or other tasks without matching audio.
            continue

        textgrid_path = stimuli_dir / f"{story_id}.TextGrid"
        if not textgrid_path.exists() and textgrid_dir.exists():
            textgrid_path = textgrid_dir / f"{story_id}.TextGrid"
        run_id = None
        run_match = re.search(r"_run-([0-9]+)", run_path.name)
        if run_match:
            run_id = run_match.group(1)

        stories.append(
            {
                "subject": subject,
                "session": run_path.parents[1].name,
                "run": run_id,
                "story_id": story_id,
                "bold": str(run_path),
                "wav": str(wav_path),
                "textgrid": str(textgrid_path) if textgrid_path.exists() else None,
            }
        )

    return stories
