# File card UI correction — 2026-09-13

User screenshots showed upload, history and unrelated catalog rows on the same
screen. Previous explanation proposed a file card before it was implemented.
This change implements it, superseding the prior UI layout description.

- Default view: search, names, separate folder labels and Open file card.
- New-file upload: its own screen, accessed from Add file.
- File card: selected name, folder, current size/download, Update this file and
  per-file history. No upload picker or unrelated file list.
- Update screen: only the selected file and upload action, with a return link.
- Successful upload/update redirects to that file's card with a status message.

Storage code and formats are unchanged. Existing duplicate-path entries remain;
folder labels distinguish them. No data is merged or removed automatically.

Verification command:

    PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s experiments/personal_memory_1tb_v2/tests -p test_memory_browser.py -v

FILE_CARD_TEST_OUTPUT.txt records 10 successful HTTP integration tests. Added
regression traverses list -> add -> save redirect -> card -> update, ensuring
unrelated controls/catalog entries are absent. Other tests retain version
restore, corruption, duplicate, stale-parent and access checks. No graphical
browser screenshot test; laptop visual acceptance remains pending.
