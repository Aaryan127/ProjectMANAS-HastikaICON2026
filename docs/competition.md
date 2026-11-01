# Competition context

HASTIKA at ICON 2026 has two subtasks on Kannada–English YouTube comments: binary hate detection
(A) and six-way category classification of hateful comments (B). The ranking metric is macro-F1;
accuracy is also reported.

- [ICON shared-task listing and paper submission route](https://www.icon2026.org/shared-tasks)
- [Organizer starting kit, task definitions and citation](https://github.com/shankarb14/Hastika-ICON2026)
- [CodaBench competition](https://www.codabench.org/competitions/17784/)

This repository supports the system-description paper by Robin Punnen Alex and Aaryan Ganesh Salunke (Project MANAS, Manipal Institute of Technology). The final systems are the 12-model
Task A and 8-model Task B ensembles in [`results/final_scores.json`](../results/final_scores.json),
built as described in [methods.md](methods.md).

The final fits combine each task's original training labels with its released labelled validation
set. If you reuse this protocol, check the phase-specific permissions and external-data terms with
the organizers. Test labels are not distributed here. Organizer schedules and submission formats
change, so consult the competition pages for current instructions.
