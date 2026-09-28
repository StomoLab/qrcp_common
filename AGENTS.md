# AGENTS.md

## HUCC collaboration

- Read `/Users/masa/hucc_home/AGENT-WORKFLOW.md` before work in this local HUCC mirror or its worktrees. This is an explicit file-read instruction, not an automatic Markdown import.
- Preserve existing uncommitted changes. Claude coordinates and reviews; one Codex worker edits a software directory at a time. Record each assignment under `.agent-runs/<task-id>/`.
- Remote HUCC operations use `/Users/masa/hucc_home/.opencode/bin/huccctl` from `/Users/masa/hucc_home`, following `/Users/masa/.agents/skills/hucc-hpc/SKILL.md`. No direct ssh/rsync/qsub/qstat/qdel or login-node builds/tests. Existing human authorization requirements still apply.
- Treat recorded measurements and current-state notes below as historical context until verified for this task. Keep numerical output contracts and agreed design constraints.

## Project guidance

Read `README.md` before changing this project.
Do not infer a successful build or supported test command from this setup. Validate the actual toolchain and report checks that could not run.
