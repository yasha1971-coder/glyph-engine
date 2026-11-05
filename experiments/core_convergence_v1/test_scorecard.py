import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scorecard import Invalid, frontier, validate

H = hashlib.sha256(b"synthetic source").hexdigest()
W = hashlib.sha256(b"synthetic fixed query workload").hexdigest()
E = hashlib.sha256(b"synthetic evidence receipt").hexdigest()


def fixture(name="synthetic"):
    return {
        "schema":"GLYPH_CORE_FRONTIER_V1",
        "dataset":{"name":"synthetic", "raw_bytes":1000, "source_sha256":H,
                   "workload_sha256":W,"document_order_id":"ordered-docs-v1"},
        "candidate":{"name":name,"commit":"a"*40,"scope":"complete_searchable_store","profile":"synthetic"},
        "storage_bytes":{"payload":450,"search_index":50,"locate":20,"manifests_evidence":10,"recovery_metadata":10,"other_mandatory":10},
        "correctness":{"count_cases":10,"locate_cases":10,"absent_cases":2,"boundary_cases":2,
                       "count_mismatches":0,"locate_mismatches":0,"boundary_mismatches":0,
                       "full_coverage":True,"independent_oracle":True,"restored_sha256":H},
        "resources":{"build_peak_rss_bytes":1000000,"open_peak_rss_bytes":100000,
                    "startup_ms":10.0,"build_time_ms":200.0,"index_build_workspace_bytes":1000,
                    "count_samples_ms":[1.0]*20,"locate_samples_ms":[3.0]*20,"extract_samples_ms":[5.0]*20,
                    "decoded_bytes_total":2000,"requested_bytes_total":1000},
        "provenance":{"commands":["synthetic command only"],"evidence_sha256":E,
                      "environment_id":"synthetic-env", "oracle_method":"synthetic", "clean_source_tree":True}
    }

class ScorecardTest(unittest.TestCase):
    def test_positive_is_consistent_not_proof(self):
        out=validate(fixture())
        self.assertEqual(out["full_store_bytes"],550)
        self.assertEqual(out["full_store_ratio"],.55)
        self.assertIn("NOT_REPLAY_VERIFIED",out["verdict"])

    def test_reject_kernel_only(self):
        r=fixture();r["candidate"]["scope"]="kernel"
        with self.assertRaises(Invalid): validate(r)

    def test_reject_missing_storage(self):
        r=fixture();del r["storage_bytes"]["search_index"]
        with self.assertRaises(Invalid): validate(r)

    def test_reject_nonzero_mismatch(self):
        r=fixture();r["correctness"]["locate_mismatches"]=1
        with self.assertRaises(Invalid):validate(r)

    def test_reject_incomplete_coverage(self):
        r=fixture();r["correctness"]["full_coverage"]=False
        with self.assertRaises(Invalid):validate(r)

    def test_reject_restore_hash(self):
        r=fixture();r["correctness"]["restored_sha256"]="0"*64
        with self.assertRaises(Invalid):validate(r)

    def test_reject_booleans_as_numbers(self):
        r=fixture();r["storage_bytes"]["payload"]=True
        with self.assertRaises(Invalid):validate(r)

    def test_reject_nonfinite_latency(self):
        r=fixture();r["resources"]["locate_samples_ms"][0]=float("nan")
        with self.assertRaises(Invalid):validate(r)

    def test_reject_dirty_tree(self):
        r=fixture();r["provenance"]["clean_source_tree"]=False
        with self.assertRaises(Invalid):validate(r)

    def test_reject_different_environment(self):
        r1=validate(fixture("a"));r2=validate(fixture("b"));r2["environment_id"]="other"
        with self.assertRaises(Invalid):frontier([r1,r2])

    def test_dominated_and_nondominated(self):
        r1=validate(fixture("a"));r2=validate(fixture("b"))
        r2["full_store_bytes"]+=10
        r2["locate_p95_ms"]+=1
        out=frontier([r1,r2])
        self.assertEqual([r["pareto_status"] for r in out],["NONDOMINATED","DOMINATED"])

    def test_tradeoff_keeps_both(self):
        r1=validate(fixture("a"));r2=validate(fixture("b"))
        r2["full_store_bytes"]-=10;r2["locate_p95_ms"]+=1
        self.assertEqual([r["pareto_status"] for r in frontier([r1,r2])],["NONDOMINATED","NONDOMINATED"])

    def test_cli_does_not_invent_result(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"example.json";p.write_text(json.dumps(fixture()))
            cli=Path(__file__).with_name("scorecard.py")
            result=subprocess.run([sys.executable,str(cli),str(p)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)["records"][0]["verdict"],"REPORTED_FIELDS_CONSISTENT_NOT_REPLAY_VERIFIED")

if __name__=="__main__":
    unittest.main()
