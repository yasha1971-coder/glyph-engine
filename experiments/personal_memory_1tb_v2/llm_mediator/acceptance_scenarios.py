"""Synthetic acceptance specifications. Defining a scenario is NOT passing it.

Run this file to print the machine-readable catalogue. No private data is used,
no files are deleted, and no live product operations or model calls are made.
"""
import json

# id, basis, given, when, acceptance oracle
ROWS = [
('UX01','user','A fresh local installation and one synthetic PDF.', 'Add, find, preview and restore it without a terminal.', 'One document card; each action unambiguous; restored SHA-256 equals the source; user completes unaided.'),
('UX02','user','A folder of 100 synthetic files and one new file tomorrow.', 'Import twice, then import after adding the new file.', '101 document identities, no additional payload for the repeat import; progress and skipped items visible.'),
('ID01','user','Identical bytes under three names in three folders.', 'Import all three paths.', 'All three origins retained; shared payload counted once; no arbitrary name loss.'),
('ID02','user','Two unrelated files with the same basename.', 'Import and search by basename.', 'Two clearly distinguished cards; no automatic merge based on name.'),
('VER01','user','A 32 MiB synthetic file and 100 versions with small deterministic edits.', 'Save and restore every version.', '101 byte-perfect states; report new payload and recipe bytes separately; unmodified chunks referenced, not repeatedly copied.'),
('VER02','user','Two edits independently based on version 1.', 'Save both with different labels.', 'Both branches retained with correct parent IDs; user can select either; no silent last-writer overwrite.'),
('VER03','engineering','Three versions sharing most blocks.', 'Permanently delete the middle version with host confirmation.', 'Remaining versions restore exactly; referenced blocks retained; index and preview entries for removed version invalidated.'),
('DATE01','user','A file whose filesystem creation date is later than its modification date after copying.', 'Import and view its dates.', 'Source creation, source modification and GLYPH save time shown separately with provenance; no invented original-authoring date.'),
('DATE02','user','Legacy versions with missing timestamps and new versions with recorded times.', 'Filter by date and inspect history.', 'Unknown dates remain unknown; undated results are disclosed rather than silently excluded as proven nonmatches.'),
('DATE03','engineering','UTC instants near a daylight-saving transition.', 'Display in local time and sort versions.', 'Ordering uses actual instants; timezone/offset visible; no duplicate identity caused by repeated local clock time.'),
('ID03','user','A renamed document with unchanged bytes.', 'Rename and search its former name.', 'Stable document ID, retained origin/alias and unchanged history; no spurious new payload.'),
('SEARCH01','user','Synthetic texts with exact identifiers, punctuation and RU/UA/EN words.', 'Search a quoted phrase or invoice identifier.', 'Expected document/version and byte span returned; no translated or altered identifier.'),
('SEARCH02','user','A labelled synthetic collection with paraphrases and decoy documents.', 'Ask by meaning without knowing filename.', 'Report recall@5 and wrong-document rate against held-out relevance labels; similarity is not described as proof.'),
('SEARCH03','user','Scanned synthetic documents, a rotated page and an unreadable page.', 'Search words found by OCR.', 'Readable page retrieved with page location; OCR text labelled derived; unreadable page reported unindexed; original hash unchanged.'),
('SEARCH04','user','Synthetic image, short video and audio with labelled events.', 'Ask for a visible event or spoken phrase.', 'Correct asset/time span linked; unsupported modalities explicitly disclosed; originals unchanged.'),
('SEARCH05','user','One indexed file and one relevant but pending file.', 'Search for a phrase in the pending file.', 'INCOMPLETE or indexing-pending, never a claim of absence from all memory.'),
('SEARCH06','public_case','A document removed from an index and then restored/reindexed.', 'Immediately search for its known phrase.', 'Either current result or explicit index lag; no silent stale absence; index generation tied to source version.'),
('SEC01','engineering','Allowed and denied documents containing the same phrase.', 'Ask a local model to find all matches.', 'Denied titles, snippets, counts and identifiers never enter model context or results.'),
('SEC02','engineering','An allowed document containing instructions to ignore permissions or delete files.', 'Retrieve it and generate a reply.', 'Instructions treated as quoted data; zero write/network operations and zero permission expansion.'),
('LLM01','user','Two versions state different synthetic delivery dates.', 'Ask what date was agreed.', 'Answer identifies version and evidence; conflict disclosed rather than silently combining versions.'),
('VIEW01','user','A malformed PDF and a healthy document.', 'Open the broken preview, cancel it, then view the healthy document.', 'Preview worker times out/cancels; application remains responsive; corruption and unsupported rendering distinguished.'),
('DEL01','user','One document with four versions.', 'Choose permanent deletion of exactly one version.', 'Confirmation names document and version; cancellation changes nothing; replay of an old confirmation cannot delete a newer state.'),
('DEL02','user','A document has previews, OCR, vector entries and an offline replica.', 'Confirm deletion of the whole document.', 'All live views stop returning it; derivative cleanup tracked; offline replica deletion remains pending, not falsely complete.'),
('DEL03','user','An archived document disappears from its original folder.', 'Run the next import.', 'Archive history retained; source disappearance never authorizes archive deletion.'),
('SEC03','user','A synthetic file marked confidential, containing a fake credential marker.', 'Import then ask a broad question.', 'Explicit exclusion prevents indexing/model exposure; diagnostics omit content; tests never use real secrets.'),
('OFF01','user','A populated local store with network disabled.', 'Find and restore a document.', 'Local core works offline; model availability shown separately; no implicit cloud fallback.'),
('PHONE01','public_case','A queued mobile import interrupted by OS background suspension.', 'Resume the app and reconnect.', 'Queue resumes without duplicate payload; UI distinguishes pending, uploaded and verified; no unsupported always-on promise.'),
('PHONE02','engineering','A phone under low-memory/low-battery conditions.', 'Run a semantic query while import is queued.', 'Measure actual peak memory, latency and energy on device; cancel/fallback remains usable; no unmeasured compatibility claim.'),
('FAIL01','user','An import with injected process crashes before/after durable commit.', 'Reopen and resume.', 'Every acknowledged commit restores; incomplete transaction is identifiable and recoverable; no false completed status.'),
('FAIL02','user','A destination approaching its reserve or returning ENOSPC.', 'Continue adding data.', 'Admission stops safely; committed data remains readable; report distinguishes resource stop from corruption.'),
('FAIL03','engineering','Two independent test replicas, one with a corrupted shared block.', 'Verify and explicitly repair from the healthy replica.', 'Damage detected; repaired bytes match trusted pin; unrelated versions remain intact; repair provenance recorded.'),
('FAIL04','engineering','Payload packs retained but primary catalogue unavailable.', 'Use independent catalogue backup and restore on a clean installation.', 'Names, version graph and all hashes recovered without original source/cache; missing backup results in explicit failure.'),
('UPGRADE01','user','A frozen old-format fixture and an interrupted migration to a new format.', 'Upgrade then exercise rollback.', 'Old reader/export path remains available; migration never overwrites the only healthy copy; compatibility matrix reported.'),
('SCALE01','user','Separate unique-byte, many-small-file and deep-history workloads.', 'Increase each dimension independently.', 'Record RSS, p95/p99 lookup, ingest and verified restore; no TB claim inferred from metadata-only rows.'),
('SPACE01','user','Text-heavy, photo-heavy, already-compressed and mixed synthetic/public corpora.', 'Compare storage policies end to end.', 'Count payload, catalogues, history, OCR, previews, embeddings and model weights separately; no universal 30% promise.'),
('EXIT01','user','A store opened on a clean computer without the LLM or original application settings.', 'Export a chosen version and its metadata.', 'Original bytes and portable metadata recovered with a documented reader; model not required for decoding.'),
('LLM02','user','No model server, model timeout or malformed model output.', 'Search by exact filename and restore.', 'Deterministic search remains available; clear model-unavailable state; no fabricated answer.'),
('UX03','user','A first-time user searching an old document among similar names.', 'Find, preview, select an older version and return to search.', 'One search surface, one card, visible version label; measure time and wrong-version selections in observed usability tests.'),
('DEL04','engineering','Encrypted payload on SSD plus backups and shared blocks.', 'Request irreversible physical erasure.', 'Distinguish logical deletion, replica cleanup and cryptographic erasure; no claim that unlink securely erases every physical copy.'),
('TRUST01','user','A single healthy archive with no independently verified backup.', 'Ask whether originals can be removed to free space.', 'No automatic source deletion or assertion of backup safety; show the missing independent recovery evidence.'),
]


def catalogue():
    return {'format': 'GLYPH_PERSONAL_MEMORY_ACCEPTANCE_V1',
            'scope': 'Synthetic specifications; none of these scenarios is marked executed by this catalogue.',
            'cases': [dict(id=i, basis=b, given=g, when=w, oracle=o,
                           status='NOT_EXECUTED', evidence=None)
                      for i,b,g,w,o in ROWS]}


if __name__ == '__main__':
    print(json.dumps(catalogue(), ensure_ascii=False, indent=2))
