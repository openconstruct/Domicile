"""M5 demo beat (hackathon segment): one Bash scenario and one M4 social
scenario run back-to-back from the curriculum runner -- the runner's
per-family / per-difficulty report is the on-camera output. Scripted
students stand in for the trained resident; swap in
students.lfm_student_bash for the live model.
"""
import sys
sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
from bedroom.multi import load_multi_dir, run_curriculum

base = __file__.rsplit("/tests/", 1)[0]
bash = [s for s in load_multi_dir(base + "/scenarios/bash") if s["id"] == "bash-0004"]
social = [s for s in load_multi_dir(base + "/scenarios/social") if s["id"] == "calendar-0001"]

SCRIPT = {
    "bash-0004": [
        {"tool": "Bash", "args": {"argv": ["find", "/data", "-name", "*.log"]}},
        {"text": "/data/exports/audit.log\n/data/old/access.log\n/data/server.log"},
    ],
    "calendar-0001": [
        {"tool": "CalendarList", "args": {}},
        {"tool": "CalendarCreate", "args": {"summary": "Quillmark sync",
                                            "start": "2026-10-23T12:30:00-04:00",
                                            "end": "2026-10-23T13:15:00-04:00"}},
        {"text": "Booked Quillmark sync for Friday 12:30-13:15."},
    ],
}

def scripted(obs):
    return SCRIPT[obs["scenario_id"]][obs["turn"]]

if __name__ == "__main__":
    rep = run_curriculum(bash + social, scripted)
    print(f"demo run: overall={rep['overall']} n={rep['n']}")
    print(f"  by_family={rep['by_family']}")
    print(f"  by_difficulty={rep['by_difficulty']}")
    assert rep["overall"] == 1.0
    print("DEMO OK")
